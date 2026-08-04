"""Diagnostics for speech model training failure."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speech_to_text import load_waveform, waveform_to_spectrogram, SAMPLE_RATE

data_dir = Path(__file__).resolve().parent / "commonvoice_zhTW"
records = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))

# 1. Character frequency
all_chars = Counter()
for r in records:
    for c in r["text"]:
        all_chars[c] += 1

print("=== Top 20 characters ===")
for char, count in all_chars.most_common(20):
    print(f"  '{char}': {count}")

print(f"\nTotal unique chars: {len(all_chars)}")
print(f"Total characters: {sum(all_chars.values())}")

# 2. Check if '路' is common
lu_count = all_chars.get("路", 0)
print(f"'路' appears {lu_count} times ({lu_count/sum(all_chars.values())*100:.1f}%)")

# 3. Audio duration distribution
durations = []
for r in records[:100]:
    wav_path = data_dir / r["audio"]
    if wav_path.exists():
        wf = load_waveform(wav_path)
        durations.append(len(wf) / SAMPLE_RATE)

if durations:
    print(f"\n=== Audio durations (first 100 clips) ===")
    print(f"  Min: {min(durations):.2f}s")
    print(f"  Max: {max(durations):.2f}s")
    print(f"  Avg: {sum(durations)/len(durations):.2f}s")

# 4. Check text length distribution
text_lens = [len(r["text"]) for r in records]
print(f"\n=== Text lengths ===")
print(f"  Min: {min(text_lens)} chars")
print(f"  Max: {max(text_lens)} chars")
print(f"  Avg: {sum(text_lens)/len(text_lens):.1f} chars")

# 5. Check spectrogram shape for a sample
wav_path = data_dir / records[0]["audio"]
if wav_path.exists():
    wf = load_waveform(wav_path)
    spec = waveform_to_spectrogram(wf)
    print(f"\n=== Spectrogram shape ===")
    print(f"  Shape: {spec.shape} (freq_bins, time_frames)")
    print(f"  Duration: {len(wf)/SAMPLE_RATE:.2f}s")
    print(f"  Text: {records[0]['text']}")
