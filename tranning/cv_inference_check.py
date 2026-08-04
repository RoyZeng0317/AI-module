"""Test inference on Common Voice trained model."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speech_to_text import _load, load_waveform, waveform_to_spectrogram, _fit_time_frames, DEFAULT_OUT_DIR

loaded = _load(DEFAULT_OUT_DIR)
if loaded is None:
    print("No checkpoint found")
    sys.exit(1)

model, config, idx_to_char = loaded
print(f"Model: {len(idx_to_char)} chars, max_frames={config['max_frames']}")

import torch

data_dir = Path(__file__).resolve().parent / "commonvoice_zhTW"
records = json.loads((data_dir / "val.json").read_text(encoding="utf-8"))

for r in records[:10]:
    wav_path = data_dir / r["audio"]
    if not wav_path.exists():
        continue
    waveform = load_waveform(wav_path)
    spec = waveform_to_spectrogram(waveform)
    spec = _fit_time_frames(spec, config["max_frames"]).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        log_probs = model(spec)

    pred_indices = log_probs.argmax(dim=-1).squeeze(0).tolist()
    pred = []
    prev = -1
    for idx in pred_indices:
        if isinstance(idx, list):
            idx = idx[0] if idx else 0
        if idx != prev and idx != 0:
            key = int(idx)
            if key in idx_to_char:
                pred.append(idx_to_char[key])
        prev = idx
    pred_text = "".join(pred)

    target = r["text"]
    match = "YES" if pred_text == target else "NO"
    print(f"Target:  {target[:60]}")
    print(f"Predict: {pred_text[:60]}")
    print(f"Match:   {match}")
    print()
