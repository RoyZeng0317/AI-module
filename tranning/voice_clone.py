"""Voice cloning — from-scratch Tacotron-style TTS scaffold (no ElevenLabs,
no cloud voice-cloning API, no pretrained vocoder — Rule 06).

This is stage 2 of the two-stage plan character_model.py's docstring left
open: stage 1 (character_model.py) turns a text description into a
temperament profile; this file turns a character's own recorded voice into a
"speak in this voice" model, and can then point that character's card
`voice_ref` field (previously always null) at the trained checkpoint via
`--attach-to-character`.

Point this at a manifest of (audio, text) pairs *recorded by one person* —
this is single-speaker TTS, not the harder zero-shot "clone from 5 seconds of
anyone's voice" problem. One checkpoint == one character's voice, the same
"one checkpoint per persona" shape chats.py already uses for
character_chat_runs. Right now, with no real recordings provided yet, this
only defines the architecture and training loop; it is smoke-tested against a
handful of synthetic sine-wave "utterances" in test_voice_clone.py (no
voice-quality claim whatsoever — just "the pipeline runs").

Manifest format — a JSON list, audio paths relative to --audio-root (defaults
to the manifest's own folder), PCM WAV files, same shape as
speech_to_text.py's manifest but read in the opposite direction (text -> audio
instead of audio -> text):

    [{"audio": "0001.wav", "text": "你好"}, {"audio": "0002.wav", "text": "再見"}]

Architecture: character-level text tokenizer (PAD/UNK only, same as
character_model.py — no SOS/EOS needed since this encodes source text, it
never generates text) feeds chats.py's existing `Encoder` (GRU) unchanged;
chats.py's existing Luong `Attention` is reused unchanged too. The only new
piece is `SpecDecoder`: a small "prenet" (2-layer MLP with dropout, the
standard Tacotron regularizer) turns the *previous* spectrogram frame into a
vector, attention pulls in a context vector from the encoder, a GRU combines
them into the next frame + a "stop" logit (should generation end here?),
autoregressively, one time frame at a time — teacher forcing during training,
free-running (each frame feeds the next) at inference, exactly mirroring how
chats.py's decoder generates characters one at a time except the output unit
is a spectrogram frame instead of a vocabulary id.

The target spectrogram is the *exact* log-magnitude linear STFT
speech_to_text.py already computes (`waveform_to_spectrogram`, same N_FFT/
HOP_LENGTH) — not a mel spectrogram — specifically so no neural vocoder is
needed: `griffin_lim()` below inverts a linear-magnitude spectrogram back to
a waveform via classical iterative phase estimation (Griffin & Lim, 1984),
adding zero extra trained parameters and comfortably fitting Rule 06's
RTX 4060 8GB budget. The honest tradeoff: Griffin-Lim's reconstructed phase
is only an estimate, so audio quality will sound noticeably more "robotic"/
buzzy than a neural vocoder (WaveGlow/HiFi-GAN) would produce — accepted here
because training a vocoder from scratch needs far more audio than one
character's spoken lines will ever provide.

Design choices made to avoid overfitting AND underfitting, adapted from the
same reasoning road_sign_train.py / OCR.py / speech_to_text.py /
character_model.py already use:

  Underfitting:
    - Attention over every encoder position (not one fixed bottleneck vector)
      so the decoder can align to long sentences instead of collapsing.
    - ReduceLROnPlateau instead of a fixed decay schedule.

  Overfitting:
    - No time-reversal augmentation — reversed speech is unintelligible and
      changes speaker identity entirely, the same directional-signal
      reasoning speech_to_text.py/OCR.py apply to audio/text.
    - Dropout inside the prenet + weight decay (AdamW).
    - Early stopping on validation loss, restoring the best checkpoint.
    - Per-epoch validation frame-L1 (mean absolute error per spectrogram bin,
      the TTS analogue of OCR.py's val_cer / character_model.py's val_mae) is
      printed alongside the loss gap, with a warning on a wide train/val gap.
    - **Honest caveat this project's other modules don't have to make as
      strongly**: single-speaker TTS with what will likely be only a handful
      of recorded sentences per character is an extreme overfitting regime —
      there usually isn't enough data for validation loss to mean much until
      a real recording set (dozens+ of lines) exists. Treat early results as
      "did the pipeline learn *something*", not "does it sound right".

Usage:
    python voice_clone.py --train-manifest train.json --val-manifest val.json --out-dir characters/character_voice_runs/周柯宇 --epochs 300
    python voice_clone.py --clone --text "你好" --out-wav out.wav --out-dir characters/character_voice_runs/周柯宇
    python voice_clone.py --attach-to-character 周柯宇 --out-dir characters/character_voice_runs/周柯宇
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chats import Attention, Encoder
from character_model import _safe_filename
from speech_to_text import FREQ_BINS, HOP_LENGTH, N_FFT, SAMPLE_RATE, load_waveform, waveform_to_spectrogram
from train_utils import EarlyStopper, loss_gap_warning, plateau_scheduler, save_checkpoint

PAD, UNK = 0, 1
SPECIAL_TOKENS = {"<pad>": PAD, "<unk>": UNK}

TRANNING_DIR = Path(__file__).resolve().parent
DEFAULT_CHARACTERS_DIR = TRANNING_DIR / "characters"
DEFAULT_OUT_DIR = DEFAULT_CHARACTERS_DIR / "character_voice_runs"


def tokenize(text: str) -> list[str]:
    return list(text.strip())


def build_vocab(records: list[dict]) -> dict[str, int]:
    vocab = dict(SPECIAL_TOKENS)
    for record in records:
        for token in tokenize(record["text"]):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict, max_len: int) -> list[int]:
    ids = [vocab.get(t, UNK) for t in tokenize(text)][:max_len]
    ids += [PAD] * (max_len - len(ids))
    return ids


def _pad_frames(spec: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Zero-pad (silence) or crop along the time axis to a fixed width —
    same reasoning as speech_to_text.py's own frame padding, kept as a
    separate small copy here rather than imported since it operates on this
    module's own frame-count bookkeeping (see VoiceCloneDataset below).
    """
    freq_bins, frames = spec.shape
    if frames >= max_frames:
        return spec[:, :max_frames]
    pad = torch.zeros(freq_bins, max_frames - frames)
    return torch.cat([spec, pad], dim=1)


def save_waveform(waveform: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a mono 16-bit PCM WAV — the save-side counterpart of
    speech_to_text.load_waveform(), stdlib `wave` only, no soundfile dep.
    """
    import wave

    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _match_frames(spec: torch.Tensor, frames: int) -> torch.Tensor:
    current = spec.size(-1)
    if current == frames:
        return spec
    if current > frames:
        return spec[..., :frames]
    pad = torch.zeros(spec.shape[:-1] + (frames - current,), dtype=spec.dtype)
    return torch.cat([spec, pad], dim=-1)


def griffin_lim(magnitude: torch.Tensor, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                n_iter: int = 32) -> torch.Tensor:
    """Griffin & Lim (1984) iterative phase reconstruction. speech_to_text.py's
    spectrogram feature is already an invertible linear-magnitude STFT (no mel
    filterbank collapsing frequency information), so a classical DSP algorithm
    can reconstruct a waveform with zero trained parameters instead of needing
    a neural vocoder — the deliberate choice that keeps this module inside
    Rule 06's self-built / RTX-4060-8GB budget.

    magnitude: (freq_bins, frames), LINEAR magnitude (not log — caller must
    undo the log1p from waveform_to_spectrogram first).
    """
    window = torch.hann_window(n_fft)
    frames = magnitude.size(-1)
    waveform_length = max(1, (frames - 1) * hop_length)
    angles = torch.exp(1j * torch.rand_like(magnitude) * 2 * np.pi)
    spec = magnitude.to(torch.complex64) * angles
    for _ in range(n_iter):
        waveform = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, window=window,
                                center=True, length=waveform_length)
        rebuilt = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window,
                              center=True, return_complex=True)
        rebuilt = _match_frames(rebuilt, frames)
        angles = rebuilt / (rebuilt.abs() + 1e-8)
        spec = magnitude.to(torch.complex64) * angles
    return torch.istft(spec, n_fft=n_fft, hop_length=hop_length, window=window,
                        center=True, length=waveform_length)


class VoiceCloneDataset(Dataset):
    def __init__(self, records: list[dict], audio_root: Path, vocab: dict[str, int],
                 max_len: int, max_frames: int):
        self.records = records
        self.audio_root = Path(audio_root)
        self.vocab = vocab
        self.max_len = max_len
        self.max_frames = max_frames

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text_ids = torch.tensor(encode_text(record["text"], self.vocab, self.max_len))
        waveform = load_waveform(self.audio_root / record["audio"])
        spec = waveform_to_spectrogram(waveform)  # (freq_bins, frames), log1p magnitude
        n_frames = min(spec.size(1), self.max_frames)
        spec = _pad_frames(spec, self.max_frames)
        frames = spec.transpose(0, 1).contiguous()  # (max_frames, freq_bins), time-major
        return text_ids, frames, n_frames


class Prenet(nn.Module):
    """Tacotron-style bottleneck: squeezes the previous spectrogram frame
    through two dropout-regularized linear layers before it reaches the
    decoder GRU — the main overfitting guard for a model that will likely see
    very little audio per character.
    """

    def __init__(self, in_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SpecDecoder(nn.Module):
    """Autoregressive spectrogram decoder — same shape as chats.py's Decoder
    (attention context + GRU + linear head, one output unit per step) except
    the input/output unit is a continuous spectrogram frame instead of a
    discrete vocabulary id, and there's a second head predicting when to stop.
    """

    def __init__(self, freq_bins: int, hidden_size: int, prenet_size: int, dropout: float):
        super().__init__()
        self.freq_bins = freq_bins
        self.prenet = Prenet(freq_bins, prenet_size, dropout)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(prenet_size + hidden_size, hidden_size, batch_first=True)
        self.frame_out = nn.Linear(hidden_size * 2, freq_bins)
        self.stop_out = nn.Linear(hidden_size * 2, 1)

    def forward(self, prev_frame, hidden, encoder_outputs, src_mask):
        # prev_frame: (batch, freq_bins)
        prenet_out = self.prenet(prev_frame).unsqueeze(1)  # (batch, 1, prenet_size)
        context = self.attention(hidden.squeeze(0), encoder_outputs, src_mask)  # (batch, hidden_size)
        gru_input = torch.cat([prenet_out, context.unsqueeze(1)], dim=2)
        output, hidden = self.gru(gru_input, hidden)
        combined = torch.cat([output.squeeze(1), context], dim=1)
        frame = self.frame_out(combined)
        stop_logit = self.stop_out(combined).squeeze(1)
        return frame, stop_logit, hidden


class VoiceCloneModel(nn.Module):
    """Thin container so train_utils.EarlyStopper/save_checkpoint (built for
    a single nn.Module) can track the encoder+decoder pair as one unit and
    write one best_model.pt, instead of chats.py's separate encoder.pt/
    decoder.pt files — this module needs early stopping on a real val split
    (chats.py doesn't validate/early-stop at all), so it fits train_utils.py's
    existing single-model convention instead.
    """

    def __init__(self, encoder: Encoder, decoder: SpecDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def _run_epoch(model: VoiceCloneModel, loader: DataLoader, optimizer, device: str,
                train: bool, max_frames: int, teacher_forcing_ratio: float, stop_loss_weight: float):
    model.train(mode=train)
    total_frame_l1, total_stop_bce, total_valid_frames = 0.0, 0.0, 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for text_ids, target_frames, n_frames in loader:
            text_ids = text_ids.to(device)
            target_frames = target_frames.to(device)  # (batch, max_frames, freq_bins)
            n_frames = n_frames.to(device).long()
            batch_size = text_ids.size(0)

            if train:
                optimizer.zero_grad()

            encoder_outputs, hidden = model.encoder(text_ids)
            src_mask = text_ids == PAD

            frame_mask = (torch.arange(max_frames, device=device).unsqueeze(0)
                          < n_frames.unsqueeze(1)).float()  # (batch, max_frames)
            stop_targets = torch.zeros(batch_size, max_frames, device=device)
            stop_targets[torch.arange(batch_size, device=device), (n_frames - 1).clamp(min=0)] = 1.0

            # training always teacher-forces by default (--teacher-forcing-ratio 1.0) since
            # free-running a continuous regression target early in training compounds error far
            # more violently than chats.py's discrete-token decoder does; validation always
            # teacher-forces so val_loss stays a comparable reconstruction metric across epochs.
            teacher_forcing = (not train) or (random.random() < teacher_forcing_ratio)
            prev_frame = torch.zeros(batch_size, model.decoder.freq_bins, device=device)
            frame_l1_sum = torch.tensor(0.0, device=device)
            stop_bce_sum = torch.tensor(0.0, device=device)

            for t in range(max_frames):
                frame_pred, stop_logit, hidden = model.decoder(prev_frame, hidden, encoder_outputs, src_mask)
                mask_t = frame_mask[:, t]
                frame_err = F.l1_loss(frame_pred, target_frames[:, t], reduction="none").mean(dim=1)
                frame_l1_sum = frame_l1_sum + (frame_err * mask_t).sum()
                stop_err = F.binary_cross_entropy_with_logits(stop_logit, stop_targets[:, t], reduction="none")
                stop_bce_sum = stop_bce_sum + (stop_err * mask_t).sum()
                prev_frame = target_frames[:, t] if teacher_forcing else frame_pred

            valid_frames = frame_mask.sum().clamp(min=1)
            loss = (frame_l1_sum + stop_loss_weight * stop_bce_sum) / valid_frames
            if train:
                loss.backward()
                optimizer.step()

            total_frame_l1 += frame_l1_sum.item()
            total_stop_bce += stop_bce_sum.item()
            total_valid_frames += valid_frames.item()

    avg_frame_l1 = total_frame_l1 / max(total_valid_frames, 1.0)
    avg_loss = (total_frame_l1 + stop_loss_weight * total_stop_bce) / max(total_valid_frames, 1.0)
    return avg_loss, avg_frame_l1


def train(train_manifest: Path, val_manifest: Path, audio_root: Path, out_dir: Path,
          epochs: int, batch_size: int, lr: float, patience: int, max_len: int, max_frames: int,
          embed_size: int, hidden_size: int, prenet_size: int, dropout: float, weight_decay: float,
          teacher_forcing_ratio: float, stop_loss_weight: float, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_records = json.loads(Path(train_manifest).read_text(encoding="utf-8"))
    val_records = json.loads(Path(val_manifest).read_text(encoding="utf-8"))
    audio_root = Path(audio_root) if audio_root else Path(train_manifest).resolve().parent

    vocab = build_vocab(train_records)

    train_loader = DataLoader(
        VoiceCloneDataset(train_records, audio_root, vocab, max_len, max_frames),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        VoiceCloneDataset(val_records, audio_root, vocab, max_len, max_frames),
        batch_size=batch_size, shuffle=False,
    )

    encoder = Encoder(len(vocab), embed_size, hidden_size)
    decoder = SpecDecoder(FREQ_BINS, hidden_size, prenet_size, dropout)
    model = VoiceCloneModel(encoder, decoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = plateau_scheduler(optimizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({
            "embed_size": embed_size, "hidden_size": hidden_size, "prenet_size": prenet_size,
            "dropout": dropout, "max_len": max_len, "max_frames": max_frames,
        }, indent=2)
    )

    stopper = EarlyStopper(patience)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_frame_l1 = _run_epoch(
            model, train_loader, optimizer, device, True, max_frames, teacher_forcing_ratio, stop_loss_weight
        )
        val_loss, val_frame_l1 = _run_epoch(
            model, val_loader, optimizer, device, False, max_frames, teacher_forcing_ratio, stop_loss_weight
        )
        scheduler.step(val_loss)

        warning = loss_gap_warning(train_loss, val_loss, epoch, epochs)

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_frame_l1={train_frame_l1:.4f}  "
              f"val_loss={val_loss:.4f} val_frame_l1={val_frame_l1:.4f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_frame_l1": train_frame_l1,
                         "val_loss": val_loss, "val_frame_l1": val_frame_l1})

        if stopper.step(val_loss, model, epoch):
            break

    stopper.restore_best(model)
    save_checkpoint(model, out_dir, history)
    return model, vocab, history


_loaded_models: dict = {}


def _load(out_dir: Path):
    key = str(out_dir)
    if key in _loaded_models:
        return _loaded_models[key]

    vocab_path = out_dir / "vocab.json"
    config_path = out_dir / "config.json"
    model_path = out_dir / "best_model.pt"
    if not (vocab_path.exists() and config_path.exists() and model_path.exists()):
        return None

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    encoder = Encoder(len(vocab), config["embed_size"], config["hidden_size"])
    decoder = SpecDecoder(FREQ_BINS, config["hidden_size"], config["prenet_size"], config["dropout"])
    model = VoiceCloneModel(encoder, decoder)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loaded = (model.encoder, model.decoder, vocab, config)
    _loaded_models[key] = loaded
    return loaded


def clone_voice(text: str, out_dir: Path = DEFAULT_OUT_DIR):
    """Run the self-trained encoder/decoder to synthesize `text` in this
    checkpoint's cloned voice, autoregressively (each generated frame feeds
    the next — there is no ground truth to teacher-force with at inference).

    Returns a placeholder message (str, rather than crashing) if no
    checkpoint has been trained yet at out_dir; otherwise a 1-D float32 numpy
    waveform in roughly [-1, 1] at speech_to_text.SAMPLE_RATE.
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        return ("聲音克隆模型尚未訓練，請先提供這個角色的語音資料集並執行 "
                "`python voice_clone.py --train-manifest ... --val-manifest ...` 進行訓練。")

    encoder, decoder, vocab, config = loaded
    src = torch.tensor([encode_text(text, vocab, config["max_len"])])
    src_mask = src == PAD

    frames = []
    with torch.no_grad():
        encoder_outputs, hidden = encoder(src)
        prev_frame = torch.zeros(1, decoder.freq_bins)
        for _ in range(config["max_frames"]):
            frame, stop_logit, hidden = decoder(prev_frame, hidden, encoder_outputs, src_mask)
            frames.append(frame)
            if torch.sigmoid(stop_logit).item() > 0.5:
                break
            prev_frame = frame

    log_spec = torch.cat(frames, dim=0).transpose(0, 1)  # (freq_bins, frames)
    magnitude = torch.expm1(log_spec.clamp(min=0))
    waveform = griffin_lim(magnitude, N_FFT, HOP_LENGTH)
    return waveform.numpy()


def synthesize_to_file(text: str, out_wav: Path, out_dir: Path = DEFAULT_OUT_DIR) -> str:
    """clone_voice() + write the result as a WAV file. Returns the model's
    placeholder message instead of writing a file if no checkpoint exists yet.
    """
    result = clone_voice(text, out_dir)
    if isinstance(result, str):
        return result
    save_waveform(result, Path(out_wav))
    return f"已輸出至 {out_wav}"


def attach_voice_ref(name: str, voice_out_dir: Path, characters_dir: Path | None = None) -> Path:
    """Point an existing character card's `voice_ref` (character_model.py
    always leaves this null) at this trained voice-clone checkpoint
    directory — the wiring character_model.py's docstring explicitly left for
    "a future voice-cloning module" to fill in.

    Raises FileNotFoundError if the character card doesn't exist yet — build
    one first with `python character_model.py --build --name ... --description ...`.
    """
    characters_dir = Path(characters_dir) if characters_dir else DEFAULT_CHARACTERS_DIR
    card_path = characters_dir / f"{_safe_filename(name)}.json"
    if not card_path.exists():
        raise FileNotFoundError(
            f"character card not found: {card_path} — 請先用 "
            f"`python character_model.py --build --name {name} --description ...` 建立這個角色的 card"
        )

    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["voice_ref"] = str(Path(voice_out_dir))
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return card_path


def main():
    parser = argparse.ArgumentParser(description="Train or run a from-scratch single-speaker voice-cloning model")
    parser.add_argument("--train-manifest", type=Path, help='JSON file of [{"audio": ..., "text": ...}, ...] for ONE character\'s voice')
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="base dir audio paths are relative to (default: --train-manifest's own folder)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="dedicate one out-dir per character, e.g. characters/character_voice_runs/<name>")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--prenet-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=1.0)
    parser.add_argument("--stop-loss-weight", type=float, default=1.0)
    parser.add_argument("--clone", action="store_true",
                        help="skip training; synthesize --text into --out-wav using --out-dir's checkpoint")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--out-wav", type=Path, default=None)
    parser.add_argument("--attach-to-character", type=str, default=None,
                        help="skip training; write --out-dir into this character's card (built by "
                             "character_model.py --build) as voice_ref")
    parser.add_argument("--characters-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.attach_to_character:
        path = attach_voice_ref(args.attach_to_character, args.out_dir, args.characters_dir)
        print(f"已更新 {path}")
        return

    if args.clone:
        if not (args.text and args.out_wav):
            parser.error("--text and --out-wav are required with --clone")
        print(synthesize_to_file(args.text, args.out_wav, out_dir=args.out_dir))
        return

    if not (args.train_manifest and args.val_manifest):
        parser.error("--train-manifest and --val-manifest are required unless --clone/--attach-to-character is given")

    train(args.train_manifest, args.val_manifest, args.audio_root, args.out_dir, args.epochs,
          args.batch_size, args.lr, args.patience, args.max_len, args.max_frames, args.embed_size,
          args.hidden_size, args.prenet_size, args.dropout, args.weight_decay,
          args.teacher_forcing_ratio, args.stop_loss_weight)


if __name__ == "__main__":
    main()
