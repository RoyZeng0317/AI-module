"""Download Common Voice zh-TW from Hugging Face and prepare for training.

Converts MP3 → 16kHz WAV, creates train/val manifests.

Usage:
    python download_commonvoice.py
    python download_commonvoice.py --max-samples 1000
"""

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max number of samples to use (default 5000)")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "commonvoice_zhTW")
    args = parser.parse_args()

    from datasets import load_dataset
    from pydub import AudioSegment
    import numpy as np

    print("Loading Common Voice zh-TW from Hugging Face (streaming, no audio decode)...")
    from datasets import Audio
    ds = load_dataset("OpenFormosa/common_voice_25_zh-TW", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"Dataset loaded in streaming mode")

    # Limit samples
    n = args.max_samples
    print(f"Using up to {n} samples")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.out_dir / "clips"
    wav_dir.mkdir(exist_ok=True)

    records = []
    skipped = 0
    count = 0

    for sample in ds:
        if count >= n:
            break

        sentence = sample.get("sentence", "")
        audio = sample.get("audio")

        if not sentence or not sentence.strip():
            skipped += 1
            count += 1
            continue

        if not audio:
            skipped += 1
            count += 1
            continue

        # Save as WAV
        clip_name = f"cv_{count:06d}.wav"
        clip_path = wav_dir / clip_name

        # audio is a dict with 'path' (local path to cached MP3) or 'bytes'
        try:
            if "bytes" in audio and audio["bytes"]:
                import io as _io
                audio_segment = AudioSegment.from_mp3(_io.BytesIO(audio["bytes"]))
            elif "path" in audio and audio["path"]:
                audio_segment = AudioSegment.from_mp3(audio["path"])
            else:
                skipped += 1
                count += 1
                continue

            audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
            audio_segment.export(str(clip_path), format="wav")
        except Exception as e:
            print(f"  Error processing {count}: {e}")
            skipped += 1
            count += 1
            continue

        records.append({
            "audio": f"clips/{clip_name}",
            "text": sentence.strip(),
        })

        count += 1
        if count % 500 == 0:
            print(f"  Processed {count}/{n}...")

    print(f"Total labeled: {len(records)}, skipped: {skipped}")

    # Shuffle and split 90/10
    np.random.seed(42)
    indices = np.random.permutation(len(records))
    split = int(len(records) * 0.9)
    train_records = [{"audio": records[i]["audio"], "text": records[i]["text"]} for i in indices[:split]]
    val_records = [{"audio": records[i]["audio"], "text": records[i]["text"]} for i in indices[split:]]

    train_path = args.out_dir / "train.json"
    val_path = args.out_dir / "val.json"
    train_path.write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Done ===")
    print(f"Train: {len(train_records)} clips -> {train_path}")
    print(f"Val:   {len(val_records)} clips -> {val_path}")
    print(f"\nNext: python speech_to_text.py --train-manifest commonvoice_zhTW/train.json --val-manifest commonvoice_zhTW/val.json --audio-root commonvoice_zhTW --epochs 50")


if __name__ == "__main__":
    main()
