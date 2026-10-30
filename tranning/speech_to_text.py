"""Speech recognition (STT) -- from-scratch syllable CTC model (no Whisper, no
cloud speech API -- Rule 06).

2026-10-29 rewrite. The first version predicted CHARACTERS from a linear
spectrogram cropped to max_frames=200 x 8ms = the FIRST 1.6 SECONDS of every
clip, while each Common Voice transcript describes a ~4-5 s sentence: the
model was asked to output a whole sentence from its first second and a half,
could not align the two, and collapsed into printing one character ("路",
see diagnose_stt.py). This version:

  1. keeps the WHOLE utterance (clips longer than --max-seconds are skipped,
     never cut), batches by length, and gives CTC each clip's REAL length,
     so padding is never mistaken for speech;
  2. uses 80-band log-mel features (filterbank computed here in numpy, no
     torchaudio/librosa) normalised per utterance -- loudness and
     microphone differences no longer look like different sounds;
  3. predicts ~400 toneless pinyin SYLLABLES instead of ~3000+ characters
     (pinyin_units.py), then syllable_lm.py -- a language model counted
     from the project's own Chinese text -- turns syllables into characters.

Manifest format (unchanged) -- a JSON list, audio relative to --audio-root
(default: the manifest's folder), 16-bit PCM WAV:

    [{"audio": "0001.wav", "text": "你好"}, {"audio": "0002.wav", "text": "今天天氣很好"}]

Model: 2 strided convolutions (time and frequency / 4, i.e. one output per
40 ms -- a syllable lasts ~150-300 ms) -> 3-layer bidirectional GRU ->
per-frame syllable logits -> CTC. ~4M parameters, comfortable on an RTX
4060. Overfitting guards: SpecAugment frequency/time masks (never time
reversal -- reversed speech is a different sound), dropout, AdamW weight
decay, ReduceLROnPlateau, early stopping on validation loss; the per-epoch
line also prints validation syllable error rate (SER).

Usage (on the machine with the data / GPU):
    python speech_to_text.py --train-manifest commonvoice_zhTW/train.json --val-manifest commonvoice_zhTW/val.json
    python speech_to_text.py --recognize clip.wav
    python syllable_lm.py --build --extra-text commonvoice_zhTW/train.json   # optional: LM also learns the transcripts
"""

import argparse
import json
import random
import re
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from OCR import char_error_rate
from pinyin_units import to_syllables
from train_utils import EarlyStopper, loss_gap_warning, plateau_scheduler, save_checkpoint

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / "speech_syllable_runs"
SAMPLE_RATE = 16000

# linear STFT settings: still used by voice_clone.py (its Griffin-Lim vocoder
# inverts exactly this spectrogram), so they stay as they were
N_FFT = 254
HOP_LENGTH = 128
FREQ_BINS = N_FFT // 2 + 1

# recognizer features: 25 ms windows every 10 ms, 80 mel bands
MEL_N_FFT = 400
MEL_HOP = 160
N_MELS = 80
BLANK = 0


# --- audio i/o ----------------------------------------------------------------

def load_waveform(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a 16-bit PCM WAV with the stdlib `wave` module, mono float32 in
    [-1, 1], linearly resampled to `sample_rate` if needed."""
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
    """Log-magnitude linear STFT, (freq_bins, time_frames). Kept for
    voice_clone.py, whose Griffin-Lim vocoder inverts this exact transform."""
    tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float()
    if tensor.numel() < n_fft:
        tensor = nn.functional.pad(tensor, (0, n_fft - tensor.numel()))
    spec = torch.stft(
        tensor, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft),
        return_complex=True, center=True,
    )
    return torch.log1p(spec.abs())


def _fit_time_frames(spec: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Pad with silence or crop to a fixed number of frames (voice_clone.py)."""
    freq_bins, frames = spec.shape
    if frames >= max_frames:
        return spec[:, :max_frames]
    return torch.cat([spec, torch.zeros(freq_bins, max_frames - frames)], dim=1)


# --- recognizer features ---------------------------------------------------------

def _mel_filterbank(n_mels: int = N_MELS, n_fft: int = MEL_N_FFT, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Triangular filters equally spaced on the mel scale (HTK formula),
    shape (n_mels, n_fft // 2 + 1)."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(20.0), hz_to_mel(sample_rate / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sample_rate).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return torch.from_numpy(fb)


_MEL_FB = _mel_filterbank()


def waveform_to_logmel(waveform: np.ndarray) -> torch.Tensor:
    """(time_frames, N_MELS) log-mel features, normalised per utterance
    (each band to zero mean / unit variance) so that a quiet laptop mic and
    a loud headset look alike to the model."""
    tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
    if tensor.numel() < MEL_N_FFT:
        tensor = nn.functional.pad(tensor, (0, MEL_N_FFT - tensor.numel()))
    spec = torch.stft(tensor, n_fft=MEL_N_FFT, hop_length=MEL_HOP, window=torch.hann_window(MEL_N_FFT),
                      return_complex=True, center=True)
    mel = torch.log(_MEL_FB @ spec.abs().pow(2) + 1e-6).T          # (T, n_mels)
    return (mel - mel.mean(dim=0)) / (mel.std(dim=0) + 1e-5)


def _spec_augment(feats: torch.Tensor) -> torch.Tensor:
    """Two frequency masks (<= 15 bands) and two time masks (<= 5% of the
    clip). Never time reversal (see module docstring)."""
    feats = feats.clone()
    frames, bands = feats.shape
    for _ in range(2):
        width = random.randint(0, 15)
        f0 = random.randint(0, max(0, bands - width))
        feats[:, f0:f0 + width] = 0.0
        width = random.randint(0, max(1, frames // 20))
        t0 = random.randint(0, max(0, frames - width))
        feats[t0:t0 + width, :] = 0.0
    return feats


def subsampled_length(frames: int) -> int:
    """Frames left after the model's two stride-2 convolutions (k=3, p=1)."""
    for _ in range(2):
        frames = (frames - 1) // 2 + 1
    return frames


# --- model ----------------------------------------------------------------------------

class SyllableCTC(nn.Module):
    def __init__(self, num_units: int, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.2,
                 n_mels: int = N_MELS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        conv_bands = subsampled_length(n_mels)
        self.proj = nn.Linear(32 * conv_bands, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_units + 1)          # +1 = CTC blank (index 0)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats (N, T, n_mels) -> log-probs (T', N, units+1) in nn.CTCLoss
        layout, plus each clip's real output length T'_i."""
        x = self.conv(feats.unsqueeze(1))                             # (N, 32, T', F')
        n, c, t, f = x.shape
        x = self.proj(x.permute(0, 2, 1, 3).reshape(n, t, c * f))
        out_lengths = torch.tensor([subsampled_length(int(L)) for L in lengths], dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(x, out_lengths.clamp(max=t), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=t)
        logits = self.fc(self.drop(x))
        return torch.log_softmax(logits, dim=2).permute(1, 0, 2), out_lengths


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, idx_to_unit: dict) -> list[list[str]]:
    """Standard CTC collapse: best unit per frame, merge repeats, drop blanks."""
    best = log_probs.argmax(2).T                                      # (N, T')
    results = []
    for seq, length in zip(best.tolist(), lengths.tolist()):
        units, prev = [], None
        for idx in seq[:length]:
            if idx != BLANK and idx != prev:
                units.append(idx_to_unit[idx])
            prev = idx
        results.append(units)
    return results


# --- data -------------------------------------------------------------------------------

def _usable_records(records: list[dict], audio_root: Path, max_seconds: float) -> list[dict]:
    """Attach syllable targets and drop clips that cannot be learned from:
    no Chinese text, longer than max_seconds, or too short for their text
    (CTC needs at least one output frame per syllable)."""
    kept = []
    for r in records:
        sylls = to_syllables(r["text"], tones=False)
        if not sylls:
            continue
        with wave.open(str(audio_root / r["audio"]), "rb") as wf:
            seconds = wf.getnframes() / wf.getframerate()
        frames = int(seconds * SAMPLE_RATE) // MEL_HOP + 1
        if seconds > max_seconds or subsampled_length(frames) < len(sylls):
            continue
        kept.append({**r, "syllables": sylls, "seconds": seconds})
    return kept


class SpeechDataset(Dataset):
    def __init__(self, records: list[dict], audio_root: Path, units: dict[str, int], train: bool):
        self.records, self.audio_root, self.units, self.train = records, Path(audio_root), units, train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        feats = waveform_to_logmel(load_waveform(self.audio_root / r["audio"]))
        if self.train:
            feats = _spec_augment(feats)
        label = torch.tensor([self.units[s] for s in r["syllables"] if s in self.units], dtype=torch.long)
        return feats, label, r["syllables"]


def collate(batch):
    feats, labels, sylls = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    return padded, lengths, torch.cat(labels), torch.tensor([len(lb) for lb in labels]), list(sylls)


def _length_batches(records: list[dict], batch_size: int, shuffle: bool) -> list[list[int]]:
    """Batches of similar-length clips (less padding, faster epochs)."""
    order = sorted(range(len(records)), key=lambda i: records[i]["seconds"])
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


def syllable_error_rate(pred: list[str], target: list[str]) -> float:
    return char_error_rate(pred, target)       # edit distance works on any sequence


# --- training ------------------------------------------------------------------------------

def train(train_manifest: Path, val_manifest: Path, audio_root: Path | None, out_dir: Path,
          epochs: int = 60, batch_size: int = 16, lr: float = 1e-3, patience: int = 8,
          max_seconds: float = 12.0, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.2,
          weight_decay: float = 1e-4, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    audio_root = Path(audio_root) if audio_root else Path(train_manifest).resolve().parent
    train_raw = json.loads(Path(train_manifest).read_text(encoding="utf-8"))
    val_raw = json.loads(Path(val_manifest).read_text(encoding="utf-8"))
    train_records = _usable_records(train_raw, audio_root, max_seconds)
    val_records = _usable_records(val_raw, audio_root, max_seconds)
    print(f"usable clips: train {len(train_records)}/{len(train_raw)}, val {len(val_records)}/{len(val_raw)}"
          f" (skipped: no Chinese text, > {max_seconds}s, or too short for their text)", flush=True)
    if not train_records or not val_records:
        raise ValueError("no usable clips -- check the manifest paths / --audio-root / --max-seconds")

    inventory = sorted({s for r in train_records for s in r["syllables"]})
    units = {s: i + 1 for i, s in enumerate(inventory)}               # 0 = blank
    idx_to_unit = {i: s for s, i in units.items()}

    train_set = SpeechDataset(train_records, audio_root, units, train=True)
    val_set = SpeechDataset(val_records, audio_root, units, train=False)
    val_loader = DataLoader(val_set, batch_sampler=_length_batches(val_records, batch_size, False), collate_fn=collate)

    model = SyllableCTC(len(units), hidden_size, num_layers, dropout).to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = plateau_scheduler(optimizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "units.json").write_text(json.dumps(units, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(
        {"kind": "syllable_ctc", "hidden_size": hidden_size, "num_layers": num_layers, "n_mels": N_MELS}, indent=2))

    stopper = EarlyStopper(patience)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loader = DataLoader(train_set, batch_sampler=_length_batches(train_records, batch_size, True), collate_fn=collate)
        loss_sum, n_sum = 0.0, 0
        for feats, lengths, targets, target_lengths, _ in loader:
            feats = feats.to(device)
            optimizer.zero_grad()
            log_probs, out_lengths = model(feats, lengths)
            loss = criterion(log_probs, targets.to(device), out_lengths.to(device), target_lengths.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item() * feats.size(0)
            n_sum += feats.size(0)
        train_loss = loss_sum / n_sum

        model.eval()
        val_sum, val_n, ser_sum = 0.0, 0, 0.0
        with torch.no_grad():
            for feats, lengths, targets, target_lengths, sylls in val_loader:
                log_probs, out_lengths = model(feats.to(device), lengths)
                loss = criterion(log_probs, targets.to(device), out_lengths.to(device), target_lengths.to(device))
                val_sum += loss.item() * feats.size(0)
                val_n += feats.size(0)
                preds = greedy_decode(log_probs.cpu(), out_lengths, idx_to_unit)
                ser_sum += sum(syllable_error_rate(p, t) for p, t in zip(preds, sylls))
        val_loss, val_ser = val_sum / val_n, ser_sum / val_n
        scheduler.step(val_loss)
        warning = loss_gap_warning(train_loss, val_loss, epoch, epochs)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_ser={val_ser:.3f}{warning}",
              flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ser": val_ser})
        if stopper.step(val_loss, model, epoch):
            break

    stopper.restore_best(model)
    save_checkpoint(model, out_dir, history)
    return model, units, history


# --- inference ---------------------------------------------------------------------------------

_loaded_models: dict = {}


def _load(out_dir: Path):
    """(model, idx_to_unit) for a trained syllable checkpoint, else None."""
    key = str(out_dir)
    if key in _loaded_models:
        return _loaded_models[key]
    units_path, config_path, model_path = out_dir / "units.json", out_dir / "config.json", out_dir / "best_model.pt"
    if not (units_path.exists() and config_path.exists() and model_path.exists()):
        return None
    units = json.loads(units_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = SyllableCTC(len(units), config["hidden_size"], config["num_layers"], 0.0, config.get("n_mels", N_MELS))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    loaded = (model, {i: s for s, i in units.items()})
    _loaded_models[key] = loaded
    return loaded


# --- 去口語化：轉錄後的文字過濾（見 CLAUDE.md to-do，使用者要的是「轉錄後
# 文字過濾」而非模型本身學會跳過語助詞——後者需要語音資料集本身的文字標註
# 就先人工濾掉語助詞，例如語音說「嗯我覺得」但標註寫「我覺得」，目前完全沒有
# 真實語音資料集，做不到；文字過濾這條路不需要任何訓練資料就能先動起來）。
#
# 辨識結果（音節 CTC + syllable_lm.py 轉成的漢字）沒有詞邊界、沒有標點，
# 中文又沒有空白分詞，所以
# 這裡採用「這些字元幾乎只會單獨當語助詞/感嘆詞用、極少是正常詞彙的一部分」
# 的務實判斷（嗯/啊/呃/欸/誒/喔/齁/呦/噢/唉），不管它出現在句子的哪個位置都
# 直接濾掉——這是簡單的規則式過濾，不是語言理解，符合 Rule 06 的精神：判斷
# 「濾掉哪些字」的規則是 sinco 自己寫死的，不是叫另一個 AI 模型去理解語意再
# 決定要不要濾。已知的取捨：「太好了啊」這種語尾語氣詞也會被濾成「太好了」
# ——這正是「去口語化」要的效果（口語語氣詞本來就是要濾掉的東西），但如果
# 未來語料庫出現「呃逆」（打嗝的醫學名詞）這種罕見詞，會被誤濾成「逆」，這是
# 這個簡單規則的已知盲點，機率很低所以先不特別處理。
_FILLER_CHARS = "嗯啊呃欸誒喔齁呦噢唉"
_FILLER_CHAR_PATTERN = re.compile(f"[{_FILLER_CHARS}]+")
_FILLER_WORDS_EN = re.compile(r"\b(?:um+|uh+|erm+|hmm+)\b", re.IGNORECASE)


def remove_fillers(text: str) -> str:
    """濾掉「嗯」「啊」這類語助詞/感嘆詞（中英文都處理），回傳去口語化後的
    文字。純規則式字串處理，不呼叫任何模型。
    """
    text = _FILLER_WORDS_EN.sub("", text)
    text = _FILLER_CHAR_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def recognize_syllables(waveform: np.ndarray, out_dir: Path = DEFAULT_OUT_DIR) -> list[str] | None:
    """Toneless syllables heard in `waveform`, or None without a checkpoint."""
    loaded = _load(Path(out_dir))
    if loaded is None:
        return None
    model, idx_to_unit = loaded
    feats = waveform_to_logmel(waveform).unsqueeze(0)
    with torch.no_grad():
        log_probs, out_lengths = model(feats, torch.tensor([feats.shape[1]]))
    return greedy_decode(log_probs, out_lengths, idx_to_unit)[0]


def recognize_waveform(waveform: np.ndarray, out_dir: Path = DEFAULT_OUT_DIR, lm_dir: Path | None = None) -> str:
    """Audio already in memory (e.g. a microphone capture) -> Chinese text:
    acoustic model -> syllables -> syllable_lm -> characters -> remove_fillers().

    Returns a message containing 「尚未訓練」 (rather than crashing) when no
    checkpoint exists yet -- the GUI/CLI show it instead of sending it to sinco.
    """
    from syllable_lm import DEFAULT_LM_DIR, SyllableLM

    sylls = recognize_syllables(waveform, out_dir)
    if sylls is None:
        if (Path(out_dir) / "charset.json").exists():
            return "語音辨識模型尚未訓練（這是舊版字元模型，已停用），請用新版重新執行 `python speech_to_text.py --train-manifest ... --val-manifest ...`。"
        return "語音辨識模型尚未訓練，請先執行 `python speech_to_text.py --train-manifest ... --val-manifest ...` 進行訓練。"
    if not sylls:
        return ""
    lm = SyllableLM.load(lm_dir or DEFAULT_LM_DIR)
    text = lm.decode(sylls) if lm is not None else " ".join(sylls)
    return remove_fillers(text)


def recognize(audio_path: Path, out_dir: Path = DEFAULT_OUT_DIR) -> str:
    """Same as recognize_waveform(), reading a WAV file first."""
    return recognize_waveform(load_waveform(Path(audio_path)), out_dir)


def main():
    parser = argparse.ArgumentParser(description="Train or run the from-scratch syllable CTC speech recognizer")
    parser.add_argument("--train-manifest", type=Path, help='JSON file of [{"audio": ..., "text": ...}, ...]')
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="base dir audio paths are relative to (default: --train-manifest's own folder)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=12.0, help="skip (never cut) clips longer than this")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--recognize", type=Path, default=None,
                        help="skip training; transcribe this WAV file with --out-dir's checkpoint")
    args = parser.parse_args()

    if args.recognize:
        print(recognize(args.recognize, out_dir=args.out_dir))
        return
    if not (args.train_manifest and args.val_manifest):
        parser.error("--train-manifest and --val-manifest are required unless --recognize is given")
    train(args.train_manifest, args.val_manifest, args.audio_root, args.out_dir, args.epochs, args.batch_size,
          args.lr, args.patience, args.max_seconds, args.hidden_size, args.num_layers, args.dropout,
          args.weight_decay, args.device)


if __name__ == "__main__":
    main()
