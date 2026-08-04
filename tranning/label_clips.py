"""Interactive tool to play WAV clips and enter transcriptions.

Reads train.json/val.json, plays each clip that has empty text,
and lets you type the transcription.

Usage:
    python label_clips.py
    python label_clips.py --data-dir speech_data
"""

import argparse
import json
import subprocess
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def play_wav(wav_path: Path):
    """Play a WAV file using ffplay (bundled with ffmpeg)."""
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav_path)],
            timeout=15,
        )
    except FileNotFoundError:
        print("  [ffplay not found - cannot play audio]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "speech_data")
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="Where WAV files are (default: data-dir)")
    args = parser.parse_args()

    audio_root = args.audio_root or args.data_dir

    for split_name in ["train.json", "val.json"]:
        path = args.data_dir / split_name
        if not path.exists():
            continue

        records = json.loads(path.read_text(encoding="utf-8"))
        empty = [r for r in records if not r.get("text")]

        if not empty:
            print(f"\n{split_name}: all clips have text, nothing to label")
            continue

        print(f"\n=== {split_name}: {len(empty)} clips need transcription ===")
        print("For each clip:")
        print("  [Enter] = play audio")
        print("  type text = save transcription and move to next")
        print("  [s] = skip")
        print("  [q] = quit and save\n")

        modified = False
        for i, record in enumerate(empty):
            wav_path = audio_root / record["audio"]
            print(f"[{i+1}/{len(empty)}] {record['audio']}")
            if not wav_path.exists():
                print(f"  WARNING: {wav_path} not found, skipping")
                continue

            while True:
                action = input("  > ").strip()
                if action == "":
                    print("  Playing...")
                    play_wav(wav_path)
                elif action.lower() == "q":
                    print("Saving and quitting...")
                    break
                elif action.lower() == "s":
                    print("  Skipped")
                    break
                else:
                    record["text"] = action
                    modified = True
                    print(f"  Saved: {action}")
                    break

            if action.lower() == "q":
                break

        if modified:
            path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
