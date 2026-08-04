"""Prepare audio training data from MP3 files.

Converts MP3 → 16kHz 16-bit PCM WAV, segments by silence detection,
and creates manifest JSON files for speech_to_text.py training.

Usage:
    python prepare_audio.py
    python prepare_audio.py --mp3-dir Tranning --out-dir speech_data
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

SAMPLE_RATE = 16000
MIN_SEGMENT_MS = 500      # minimum utterance length (ms)
MAX_SEGMENT_MS = 10000    # maximum utterance length (ms)
SILENCE_THRESH_DB = -35   # silence threshold (dBFS)
SILENCE_LEN_MS = 300      # minimum silence gap to split (ms)


def convert_mp3_to_wav(mp3_path: Path, out_dir: Path) -> Path:
    """Convert a single MP3 to 16kHz mono 16-bit PCM WAV."""
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    wav_path = out_dir / mp3_path.with_suffix(".wav").name
    audio.export(str(wav_path), format="wav")
    return wav_path


def segment_audio(wav_path: Path, out_dir: Path, prefix: str) -> list[dict]:
    """Split WAV by silence into individual utterance clips.
    Returns list of {"audio": filename, "duration": seconds}.
    """
    audio = AudioSegment.from_wav(str(wav_path))
    segments_info = detect_nonsilent(
        audio, min_silence_len=SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB, seek_step=10
    )

    if not segments_info:
        # no silence detected — treat entire file as one segment
        segments_info = [[0, len(audio)]]

    clips = []
    for i, (start_ms, end_ms) in enumerate(segments_info):
        dur_ms = end_ms - start_ms
        if dur_ms < MIN_SEGMENT_MS:
            continue
        # split long segments into chunks
        chunks = []
        pos = start_ms
        while pos < end_ms:
            chunk_end = min(pos + MAX_SEGMENT_MS, end_ms)
            chunks.append((pos, chunk_end))
            pos = chunk_end

        for j, (cs, ce) in enumerate(chunks):
            clip = audio[cs:ce]
            clip_name = f"{prefix}_{i:04d}_{j:02d}.wav"
            clip_path = out_dir / clip_name
            clip.export(str(clip_path), format="wav")
            clips.append({
                "audio": clip_name,
                "duration": round((ce - cs) / 1000.0, 2),
            })

    return clips


def sanitize_prefix(name: str) -> str:
    """Make a filesystem-safe prefix from the MP3 filename."""
    name = Path(name).stem
    # remove non-ascii and special chars
    name = re.sub(r"[^\w\u4e00-\u9fff]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    # truncate to reasonable length
    return name[:40] if name else "audio"


def main():
    parser = argparse.ArgumentParser(description="Convert MP3 → WAV and segment for STT training")
    parser.add_argument("--mp3-dir", type=Path, default=Path(__file__).resolve().parent / "Tranning",
                        help="Directory containing MP3 files")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "speech_data",
                        help="Output directory for WAV clips and manifests")
    args = parser.parse_args()

    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    mp3_files = list(args.mp3_dir.glob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {args.mp3_dir}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_clips = []

    for mp3 in mp3_files:
        print(f"\n--- Processing: {mp3.name} ---")
        prefix = sanitize_prefix(mp3.name)

        # Step 1: convert to WAV
        full_wav = convert_mp3_to_wav(mp3, args.out_dir)
        duration_s = len(AudioSegment.from_wav(str(full_wav))) / 1000.0
        print(f"  Converted -> {full_wav.name} ({duration_s:.1f}s)")

        # Step 2: segment
        clips = segment_audio(full_wav, args.out_dir, prefix)
        print(f"  Segmented into {len(clips)} clips")

        for c in clips:
            print(f"    {c['audio']} ({c['duration']}s)")
        all_clips.extend(clips)

    # Step 3: create manifests (80/20 split)
    np.random.seed(42)
    indices = np.random.permutation(len(all_clips))
    split = max(1, int(len(all_clips) * 0.8))
    train_indices = indices[:split]
    val_indices = indices[split:]

    # if val is empty, steal one from train
    if len(val_indices) == 0 and len(train_indices) > 1:
        val_indices = train_indices[-1:]
        train_indices = train_indices[:-1]

    train_records = [{"audio": all_clips[i]["audio"], "text": "", "duration": all_clips[i]["duration"]} for i in train_indices]
    val_records = [{"audio": all_clips[i]["audio"], "text": "", "duration": all_clips[i]["duration"]} for i in val_indices]

    train_path = args.out_dir / "train.json"
    val_path = args.out_dir / "val.json"
    train_path.write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Done ===")
    print(f"Total clips: {len(all_clips)}")
    print(f"Train: {len(train_records)} clips → {train_path}")
    print(f"Val:   {len(val_records)} clips → {val_path}")
    print(f"\nNext step: edit train.json and val.json to fill in the 'text' field for each clip,")
    print(f"then run: python speech_to_text.py --train-manifest speech_data/train.json --val-manifest speech_data/val.json --epochs 50")


if __name__ == "__main__":
    main()
