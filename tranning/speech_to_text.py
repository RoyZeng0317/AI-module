"""Speech recognition (STT) — from-scratch CTC scaffold (no Whisper, no cloud
speech API — Rule 06).

Reuses the exact CRNN + CTC machinery already built and validated in OCR.py
(that architecture is generic over any single-channel 2D input — it has
nothing OCR-specific in it — so a log-magnitude spectrogram slots in in
place of a text-line image without duplicating the model/decoder code).

Point this at a manifest of (audio, transcript) pairs once a real dataset is
available. Right now, with no data provided yet, this only defines the
feature extraction + training loop; it is smoke-tested against a handful of
synthetic sine-wave "utterances" in test_speech_to_text.py (no
recognition-accuracy claim — just "the pipeline runs").

Manifest format — a JSON list, audio paths relative to --audio-root
(defaults to the manifest's own folder), PCM WAV files:

    [{"audio": "0001.wav", "text": "你好"}, {"audio": "0002.wav", "text": "goodbye"}]

Pipeline: mono waveform -> torch.stft log-magnitude spectrogram (freq x
time, no torchaudio/librosa dependency needed) -> same CRNN (collapses
frequency to one row, keeps a column per time frame) -> 2-layer
bidirectional LSTM -> per-frame character logits -> CTC loss, exactly like
OCR.py's per-column character logits. Character-level vocabulary built the
same way (chats.py / OCR.py) so Chinese and English both work.

Design choices to avoid overfitting AND underfitting (same reasoning as
OCR.py / road_sign_train.py, adapted to audio):

  Underfitting:
    - Same 2-layer bidirectional LSTM for left/right context per frame.
    - ReduceLROnPlateau instead of a fixed decay schedule.

  Overfitting:
    - Augmentation deliberately excludes time-reversal ("flipping" audio
      backwards) — reversed speech is unintelligible / a different sound
      entirely, the same directional-signal reasoning road_sign_train.py /
      OCR.py use for signs and text. Instead uses SpecAugment-style
      frequency/time masking plus mild additive noise.
    - Dropout between LSTM layers + weight decay (AdamW), both reused as-is
      from OCR.py's CRNN.
    - Early stopping on validation loss, restoring the best checkpoint.
    - Per-epoch validation CER (character error rate) printed alongside the
      loss gap, plus a warning on a wide train/val loss gap.

Usage:
    python speech_to_text.py --train-manifest train.json --val-manifest val.json --epochs 50
    python speech_to_text.py --recognize path/to/clip.wav   # run the last checkpoint on one file
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from OCR import CRNN, BLANK, build_charset, char_error_rate, greedy_decode
from train_utils import EarlyStopper, loss_gap_warning, plateau_scheduler, save_checkpoint

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "speech_runs"
SAMPLE_RATE = 16000
N_FFT = 254          # -> 128 frequency bins (multiple of 16, same constraint as CRNN's img_height)
HOP_LENGTH = 128      # 8ms hop at 16kHz
FREQ_BINS = N_FFT // 2 + 1


def load_waveform(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a PCM WAV file with the stdlib `wave` module (no soundfile/
    librosa dependency), mixed down to mono float32 in [-1, 1].

    Simple linear-interpolation resample if the file's rate differs from
    `sample_rate` — good enough for this project's own recordings; swap for
    a real polyphase resampler if a dataset arrives with far-off rates.
    """
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frame_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported, got {sampwidth * 8}-bit")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if frame_rate != sample_rate:
        duration = len(samples) / frame_rate
        target_len = max(1, round(duration * sample_rate))
        old_idx = np.linspace(0, len(samples) - 1, num=len(samples))
        new_idx = np.linspace(0, len(samples) - 1, num=target_len)
        samples = np.interp(new_idx, old_idx, samples).astype(np.float32)

    return samples


def waveform_to_spectrogram(waveform: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> torch.Tensor:
    """Log-magnitude STFT spectrogram, shape (freq_bins, time_frames).

    Plain torch.stft — no torchaudio/librosa mel-filterbank needed, since a
    linear-frequency spectrogram is enough signal for the CNN to learn from
    at this project's scale.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float()
    if tensor.numel() < n_fft:
        tensor = nn.functional.pad(tensor, (0, n_fft - tensor.numel()))
    spec = torch.stft(
        tensor, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft),
        return_complex=True, center=True,
    )
    return torch.log1p(spec.abs())


def _fit_time_frames(spec: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Pad with silence (zeros) or crop along the time axis to a fixed
    width — same reasoning as OCR.load_image's right-pad, but zero (silence)
    is the natural "blank" filler for audio the way white background is for
    a scanned line of text.
    """
    freq_bins, frames = spec.shape
    if frames >= max_frames:
        return spec[:, :max_frames]
    pad = torch.zeros(freq_bins, max_frames - frames)
    return torch.cat([spec, pad], dim=1)


def _spec_augment(spec: torch.Tensor, freq_mask: int = 12, time_mask: int = 12, noise_std: float = 0.02) -> torch.Tensor:
    """SpecAugment-style masking + mild noise. Deliberately no time-reversal
    (see module docstring) — masking/noise perturb the recording without
    turning it into a different or unintelligible utterance.
    """
    spec = spec.clone()
    freq_bins, frames = spec.shape

    f0 = torch.randint(0, max(1, freq_bins - freq_mask), (1,)).item()
    spec[f0:f0 + freq_mask, :] = 0.0

    t0 = torch.randint(0, max(1, frames - time_mask), (1,)).item()
    spec[:, t0:t0 + time_mask] = 0.0

    spec = spec + torch.randn_like(spec) * noise_std
    return spec


class SpeechDataset(Dataset):
    def __init__(self, records: list[dict], audio_root: Path, charset: dict[str, int], max_frames: int, train: bool):
        self.records = records
        self.audio_root = Path(audio_root)
        self.charset = charset
        self.max_frames = max_frames
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        waveform = load_waveform(self.audio_root / record["audio"])
        spec = waveform_to_spectrogram(waveform)
        spec = _fit_time_frames(spec, self.max_frames)
        if self.train:
            spec = _spec_augment(spec)
        text = record["text"]
        label = torch.tensor([self.charset[c] for c in text if c in self.charset], dtype=torch.long)
        return spec.unsqueeze(0), label, text


def collate(batch):
    specs, labels, texts = zip(*batch)
    specs = torch.stack(specs)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    targets = torch.cat(labels)
    return specs, targets, target_lengths, list(texts)


def train(train_manifest: Path, val_manifest: Path, audio_root: Path, out_dir: Path,
          epochs: int, batch_size: int, lr: float, patience: int, max_frames: int,
          hidden_size: int, dropout: float, weight_decay: float, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_records = json.loads(Path(train_manifest).read_text(encoding="utf-8"))
    val_records = json.loads(Path(val_manifest).read_text(encoding="utf-8"))
    audio_root = Path(audio_root) if audio_root else Path(train_manifest).resolve().parent

    charset = build_charset(train_records)
    idx_to_char = {i: c for c, i in charset.items()}

    train_loader = DataLoader(
        SpeechDataset(train_records, audio_root, charset, max_frames, train=True),
        batch_size=batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        SpeechDataset(val_records, audio_root, charset, max_frames, train=False),
        batch_size=batch_size, shuffle=False, collate_fn=collate,
    )

    model = CRNN(len(charset) + 1, FREQ_BINS, hidden_size, dropout).to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = plateau_scheduler(optimizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "charset.json").write_text(json.dumps(charset, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({"max_frames": max_frames, "hidden_size": hidden_size}, indent=2)
    )

    stopper = EarlyStopper(patience)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_samples = 0.0, 0
        for specs, targets, target_lengths, _ in train_loader:
            specs, targets, target_lengths = specs.to(device), targets.to(device), target_lengths.to(device)
            optimizer.zero_grad()
            log_probs = model(specs)
            input_lengths = torch.full((specs.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * specs.size(0)
            train_samples += specs.size(0)
        train_loss = train_loss_sum / train_samples

        model.eval()
        val_loss_sum, val_samples, cer_sum = 0.0, 0, 0.0
        with torch.no_grad():
            for specs, targets, target_lengths, texts in val_loader:
                specs, targets, target_lengths = specs.to(device), targets.to(device), target_lengths.to(device)
                log_probs = model(specs)
                input_lengths = torch.full((specs.size(0),), log_probs.size(0), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss_sum += loss.item() * specs.size(0)
                val_samples += specs.size(0)
                preds = greedy_decode(log_probs.cpu(), idx_to_char)
                cer_sum += sum(char_error_rate(pred, target) for pred, target in zip(preds, texts))
        val_loss = val_loss_sum / val_samples
        val_cer = cer_sum / val_samples
        scheduler.step(val_loss)

        warning = loss_gap_warning(train_loss, val_loss, epoch, epochs)

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_cer={val_cer:.3f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_cer": val_cer})

        if stopper.step(val_loss, model, epoch):
            break

    stopper.restore_best(model)
    save_checkpoint(model, out_dir, history)
    return model, charset, history


_loaded_models: dict = {}


def _load(out_dir: Path):
    key = str(out_dir)
    if key in _loaded_models:
        return _loaded_models[key]

    charset_path, config_path, model_path = out_dir / "charset.json", out_dir / "config.json", out_dir / "best_model.pt"
    if not (charset_path.exists() and config_path.exists() and model_path.exists()):
        return None

    charset = json.loads(charset_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    idx_to_char = {i: c for c, i in charset.items()}

    model = CRNN(len(charset) + 1, FREQ_BINS, config["hidden_size"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loaded = (model, config, idx_to_char)
    _loaded_models[key] = loaded
    return loaded


def recognize_waveform(waveform: np.ndarray, out_dir: Path = DEFAULT_OUT_DIR) -> str:
    """Run the self-trained CRNN+CTC model on raw samples already in memory
    (e.g. straight from a microphone capture) — no temp file needed.

    Returns a placeholder message (rather than crashing) if no checkpoint
    has been trained yet at out_dir.
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        return "語音辨識模型尚未訓練，請先提供語音資料集並執行 `python speech_to_text.py --train-manifest ... --val-manifest ...` 進行訓練。"

    model, config, idx_to_char = loaded
    spec = waveform_to_spectrogram(waveform)
    spec = _fit_time_frames(spec, config["max_frames"]).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        log_probs = model(spec)
    return greedy_decode(log_probs, idx_to_char)[0]


def recognize(audio_path: Path, out_dir: Path = DEFAULT_OUT_DIR) -> str:
    """Same as recognize_waveform(), reading the audio from a WAV file first."""
    return recognize_waveform(load_waveform(Path(audio_path)), out_dir)


def main():
    parser = argparse.ArgumentParser(description="Train or run a from-scratch CRNN+CTC speech recognition model")
    parser.add_argument("--train-manifest", type=Path, help='JSON file of [{"audio": ..., "text": ...}, ...]')
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="base dir audio paths are relative to (default: --train-manifest's own folder)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--recognize", type=Path, default=None,
                        help="skip training; run STT on this WAV file using --out-dir's checkpoint")
    args = parser.parse_args()

    if args.recognize:
        print(recognize(args.recognize, out_dir=args.out_dir))
        return

    if not (args.train_manifest and args.val_manifest):
        parser.error("--train-manifest and --val-manifest are required unless --recognize is given")

    train(args.train_manifest, args.val_manifest, args.audio_root, args.out_dir,
          args.epochs, args.batch_size, args.lr, args.patience, args.max_frames,
          args.hidden_size, args.dropout, args.weight_decay)


if __name__ == "__main__":
    main()
