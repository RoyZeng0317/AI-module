"""Quick check of the trained syllable speech model on speech_data/train_clean.json:
prints target text, heard syllables and the final (language-model) text."""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pinyin_units import to_syllables  # noqa: E402
from speech_to_text import (DEFAULT_OUT_DIR, load_waveform, recognize_syllables,  # noqa: E402
                            recognize_waveform, syllable_error_rate)

data_dir = Path(__file__).resolve().parent / "speech_data"
records = json.loads((data_dir / "train_clean.json").read_text(encoding="utf-8"))

for r in records[:5]:
    wav_path = data_dir / r["audio"]
    if not wav_path.exists():
        continue
    waveform = load_waveform(wav_path)
    heard = recognize_syllables(waveform, DEFAULT_OUT_DIR)
    if heard is None:
        print("No checkpoint found -- train speech_to_text.py first")
        sys.exit(1)
    target = to_syllables(r["text"], tones=False)
    print(f"Target:    {r['text'][:60]}")
    print(f"Syllables: {' '.join(heard)[:80]}  (SER {syllable_error_rate(heard, target):.2f})")
    print(f"Predict:   {recognize_waveform(waveform, DEFAULT_OUT_DIR)[:60]}")
    print()
