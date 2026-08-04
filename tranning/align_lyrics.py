"""Auto-align song lyrics to segmented WAV clips using LRC timestamps.

Sorts clips by segment index, reconstructs their position in the original
audio, and matches each clip to the corresponding LRC lyrics line.

Usage:
    python align_lyrics.py
    python align_lyrics.py --data-dir speech_data
"""

import argparse
import json
import re
from pathlib import Path


LRC_LYRICS = """[00:18.12]如果你曾见过我
[00:21.33]眼里的萤火
[00:26.07]我想你就能懂得
[00:29.31]拥抱黑暗的缘由
[00:34.08]人总是想要触碰
[00:37.26]却又害怕收回手
[00:40.83]我偏偏张开了双臂
[00:44.49]享受失重
[00:50.10]若可以我想成为天文学家
[00:58.05]沉迷于你的夜空然后一直一直往下
[01:06.06]也许是你的引力拉住了我没失控
[01:12.84]当你不经意经过我
[01:16.80]我才终于学会坠落
[01:24.03]数不清抹不去来不及所以我
[01:28.05]只能向你坠去
[01:32.07]太绚丽太孤寂没关系
[01:35.07]我相信这种相遇
[01:40.11]降落在银河系的废墟
[01:43.08]交换彼此的彗星
[01:48.12]忽然间我确定
[01:50.10]这满身淤泥
[01:53.28]才是爱的证据
[02:12.12]若可以我想成为天文学家
[02:20.10]沉迷于无限夜空然后一直一直往下
[02:28.08]也许是你的引力拉住了我没失控
[02:34.86]当你不经意经过我
[02:38.79]我们轨道突然重合
[02:46.05]数不清抹不去来不及所以我
[02:50.07]只能向你坠去
[02:54.09]太绚丽太孤寂没关系
[02:57.09]我相信这种相遇
[03:02.07]降落在银河系的废墟
[03:05.07]交换彼此的彗星
[03:10.11]忽然间我确定
[03:12.09]这满身淤泥
[03:15.27]才是爱的证据
[03:38.28]记不起回不去握不紧就让我
[03:42.09]义无反顾飞行
[03:46.11]对不起太任性太着迷
[03:49.14]银河坠落时沉溺
[03:54.12]如果你有一天掉进去
[03:57.09]光年之外的缝隙
[04:04.11]然后你会发现
[04:06.18]那散落灰烬
[04:09.24]是灼热我的心"""


def parse_lrc(lrc_text: str) -> list[tuple[float, str]]:
    entries = []
    for line in lrc_text.strip().splitlines():
        m = re.match(r"\[(\d+):(\d+\.\d+)\](.*)", line.strip())
        if m:
            minutes, seconds, text = int(m.group(1)), float(m.group(2)), m.group(3).strip()
            if text:
                entries.append((minutes * 60 + seconds, text))
    entries.sort(key=lambda x: x[0])
    return entries


def sort_key_from_filename(filename: str) -> tuple:
    """Extract segment and chunk indices from filename for sorting.

    e.g. '银河坠落时_电视剧_炽夏_主题曲_0000_03.wav' -> (0, 3)
         '银河坠落时_电视剧_炽夏_主题曲_0001_00.wav' -> (1, 0)
    """
    m = re.search(r"_(\d+)_(\d+)\.wav$", filename)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (9999, 0)


def assign_lyrics_to_clips(clips: list[dict], lrc_entries: list[tuple[float, str]],
                           vocal_start: float = 18.0) -> list[dict]:
    """Assign lyrics to song clips based on chronological position.

    vocal_start: the approximate time (seconds) where vocals begin in the song.
    """
    # Sort clips by their original position in the audio
    clips_sorted = sorted(clips, key=lambda c: sort_key_from_filename(c["audio"]))

    # Reconstruct timeline: each clip starts where the previous one ended
    current_time = 0.0
    for clip in clips_sorted:
        clip["_start"] = current_time
        clip["_end"] = current_time + clip.get("duration", 0)
        current_time = clip["_end"]

    # For each clip, find the LRC line(s) that overlap with its time window
    for clip in clips_sorted:
        start = clip["_start"]
        end = clip["_end"]

        # Skip clips that are mostly before vocals start
        if end < vocal_start:
            clip["text"] = "(前奏)"
            continue

        # Find overlapping LRC lines
        overlapping = []
        for ts, text in lrc_entries:
            # A line "belongs to" the clip if its timestamp falls within [start, end)
            # or if it's the closest line to this clip
            if ts >= start - 1.0 and ts < end + 1.0:
                overlapping.append(text)

        if overlapping:
            clip["text"] = " ".join(overlapping)
        else:
            # Find the nearest preceding line
            nearest = ""
            for ts, text in lrc_entries:
                if ts <= start + 1.0:
                    nearest = text
                else:
                    break
            clip["text"] = nearest if nearest else "(間奏)"

        # Clean up temp keys
        clip.pop("_start", None)
        clip.pop("_end", None)

    return clips_sorted


def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "speech_data")
    args = parser.parse_args()

    lrc_entries = parse_lrc(LRC_LYRICS)
    print(f"Parsed {len(lrc_entries)} LRC lines")

    # Rebuild ALL clips from both train and val, align, then re-split
    all_clips = []
    for split_name in ["train.json", "val.json"]:
        path = args.data_dir / split_name
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))
            all_clips.extend(records)

    print(f"Total clips loaded: {len(all_clips)}")

    # Separate song clips from English clips
    song_clips = [c for c in all_clips if "银河" in c["audio"]]
    english_clips = [c for c in all_clips if "Fluent" in c["audio"]]
    print(f"Song clips: {len(song_clips)}, English clips: {len(english_clips)}")

    # Align song clips with lyrics
    if song_clips:
        song_clips = assign_lyrics_to_clips(song_clips, lrc_entries)
        filled = sum(1 for c in song_clips if c["text"] and c["text"] not in ("(前奏)", "(間奏)"))
        print(f"Song clips with lyrics: {filled}/{len(song_clips)}")
        for c in song_clips[:5]:
            print(f"  {c['audio']}: {c['text'][:50]}...")
        print("  ...")

    # English clips - leave empty for manual fill
    for c in english_clips:
        c["text"] = ""

    # Re-split into train/val (80/20)
    import numpy as np
    np.random.seed(42)

    all_aligned = song_clips + english_clips
    indices = np.random.permutation(len(all_aligned))
    split = max(1, int(len(all_aligned) * 0.8))
    train_indices = indices[:split]
    val_indices = indices[split:]
    if len(val_indices) == 0 and len(train_indices) > 1:
        val_indices = train_indices[-1:]
        train_indices = train_indices[:-1]

    train_records = []
    for i in train_indices:
        r = {k: v for k, v in all_aligned[i].items() if k in ("audio", "text")}
        train_records.append(r)
    val_records = []
    for i in val_indices:
        r = {k: v for k, v in all_aligned[i].items() if k in ("audio", "text")}
        val_records.append(r)

    train_path = args.data_dir / "train.json"
    val_path = args.data_dir / "val.json"
    train_path.write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8")

    filled_train = sum(1 for r in train_records if r["text"] and r["text"] not in ("(前奏)", "(間奏)"))
    filled_val = sum(1 for r in val_records if r["text"] and r["text"] not in ("(前奏)", "(間奏)"))
    print(f"\n=== Done ===")
    print(f"Train: {len(train_records)} clips ({filled_train} with lyrics)")
    print(f"Val:   {len(val_records)} clips ({filled_val} with lyrics)")
    print(f"\nEnglish clips still need manual transcription.")
    print(f"Edit train.json and val.json to fill in text for English clips.")


if __name__ == "__main__":
    main()
