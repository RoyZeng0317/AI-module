"""Convert an already-downloaded, already-extracted AISHELL-1 corpus
(openslr.org/33, folder `data_aishell`) into this project's speech manifests
for speech_to_text.py. Nothing is downloaded and no WAV is copied -- the
manifests point at the WAVs where they already are.

Expected layout (the official release):

    data_aishell/
      transcript/aishell_transcript_v0.8.txt   # "<utt_id> 词 词 词 ..." per line
      wav/train/S0002/BAC009S0002W0122.wav     # 16 kHz mono 16-bit PCM
      wav/dev/S0724/...
      wav/test/S0764/...

The official download ships wav/S0002.tar.gz ... (one archive per speaker)
that must be extracted first; pass --extract to let this script do it
(safely: absolute paths, '..' members and links are rejected), or extract them
yourself inside data_aishell/wav/.

Output (in --out-dir), split by AISHELL's own speaker-disjoint folders, no
random split:
    train -> train.json, dev -> val.json, test -> test.json
each a JSON list [{"audio": "wav/train/S0002/BAC009S0002W0122.wav", "text": "而对楼市成交抑制作用最大的限购"}]
with audio paths relative to the data_aishell folder (pass it as --audio-root).

Transcripts are Simplified Chinese. That is fine for the recognizer itself
(its targets are pinyin syllables), but syllable_lm.py is a Traditional
Chinese LM: only give `syllable_lm.py --build --extra-text` manifests made
with --to-traditional (OpenCC s2t, pip install opencc-python-reimplemented).

Usage:
    python aishell_import.py --aishell-dir D:/datasets/data_aishell
    python aishell_import.py --aishell-dir D:/datasets/data_aishell --to-traditional --out-dir aishell1_zhTW
    python aishell_import.py --aishell-dir D:/datasets/data_aishell --max-per-split 500   # quick trial
    python aishell_import.py --aishell-dir D:/datasets/data_aishell --extract             # unpack wav/*.tar.gz first
"""

import argparse
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / "aishell1"
TRANSCRIPT_REL = Path("transcript") / "aishell_transcript_v0.8.txt"
SPLITS = {"train": "train.json", "dev": "val.json", "test": "test.json"}


def find_root(path: Path) -> Path:
    """Accept either data_aishell itself or the folder that contains it."""
    path = Path(path)
    for cand in (path, path / "data_aishell"):
        if (cand / "transcript").is_dir() or (cand / "wav").is_dir():
            return cand
    raise SystemExit(f"[aishell_import] {path} does not look like data_aishell "
                     f"(expected transcript/ and wav/ inside it)")


def load_transcripts(path: Path) -> dict[str, str]:
    """utt_id -> text with the word-separating spaces removed."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        text = "".join(parts[1:])
        if text:
            out[parts[0]] = text
    return out


def find_wavs(root: Path) -> dict[str, dict[str, Path]]:
    """split -> {utt_id: wav path} from wav/{train,dev,test}/<speaker>/*.wav."""
    found = {}
    for split in SPLITS:
        split_dir = root / "wav" / split
        found[split] = {p.stem: p for p in sorted(split_dir.rglob("*.wav"))} if split_dir.is_dir() else {}
    return found


def _safe_members(tar: tarfile.TarFile, dest: Path):
    dest = dest.resolve()
    for m in tar.getmembers():
        name = PurePosixPath(m.name.replace("\\", "/"))
        if not name.parts or name.is_absolute() or ".." in name.parts or ":" in name.parts[0]:
            raise ValueError(f"unsafe path in archive: {m.name!r}")
        if not (m.isfile() or m.isdir()):
            raise ValueError(f"refusing link/special member in archive: {m.name!r}")
        target = (dest / m.name).resolve()
        if dest != target and dest not in target.parents:
            raise ValueError(f"archive member escapes destination: {m.name!r}")
        yield m


def extract_archives(wav_dir: Path) -> int:
    """Extract every wav/*.tar.gz into wav/ (members look like train/S0002/x.wav)."""
    archives = sorted(Path(wav_dir).glob("*.tar.gz"))
    for i, arc in enumerate(archives, 1):
        with tarfile.open(arc, "r:gz") as tar:
            members = list(_safe_members(tar, wav_dir))
            if hasattr(tarfile, "data_filter"):   # Python 3.12+ / patched 3.8+: second line of defence
                tar.extractall(wav_dir, members=members, filter="data")
            else:
                tar.extractall(wav_dir, members=members)
        if i % 50 == 0 or i == len(archives):
            print(f"  extracted {i}/{len(archives)} archives")
    return len(archives)


def make_converter(to_traditional: bool):
    if not to_traditional:
        return lambda s: s
    try:
        from opencc import OpenCC
    except ImportError:
        raise SystemExit("[aishell_import] --to-traditional needs OpenCC: "
                         "pip install opencc-python-reimplemented")
    cc = OpenCC("s2t")
    return cc.convert


def build_manifests(root: Path, out_dir: Path, to_traditional: bool = False,
                    max_per_split: int | None = None) -> dict:
    """Write train/val/test manifests; return a summary dict."""
    root = Path(root)
    transcripts = load_transcripts(root / TRANSCRIPT_REL)
    wavs = find_wavs(root)
    convert = make_converter(to_traditional)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_wav_ids = set()
    summary = {"splits": {}, "no_transcript": 0, "no_wav": 0, "paths": {}}
    for split, fname in SPLITS.items():
        records = []
        for utt, path in wavs[split].items():
            all_wav_ids.add(utt)
            if utt not in transcripts:
                summary["no_transcript"] += 1
                continue
            records.append({"audio": path.relative_to(root).as_posix(), "text": convert(transcripts[utt])})
        if max_per_split is not None:
            records = records[:max_per_split]
        out_path = out_dir / fname
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["splits"][split] = len(records)
        summary["paths"][split] = out_path
    summary["no_wav"] = sum(1 for utt in transcripts if utt not in all_wav_ids)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="AISHELL-1 (data_aishell) -> speech_to_text.py manifests")
    ap.add_argument("--aishell-dir", type=Path, required=True,
                    help="the extracted data_aishell folder (or the folder containing it)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--to-traditional", action="store_true",
                    help="convert transcripts Simplified -> Traditional with OpenCC s2t")
    ap.add_argument("--max-per-split", type=int, default=None, help="keep at most N utterances per split")
    ap.add_argument("--extract", action="store_true",
                    help="extract wav/*.tar.gz (per-speaker archives) into wav/ before importing")
    args = ap.parse_args(argv)

    root = find_root(args.aishell_dir)
    transcript = root / TRANSCRIPT_REL
    if not transcript.is_file():
        raise SystemExit(f"[aishell_import] transcript not found: {transcript}")
    make_converter(args.to_traditional)  # fail fast before any work

    wav_dir = root / "wav"
    has_wav = any(any(v) for v in find_wavs(root).values())
    archives = sorted(wav_dir.glob("*.tar.gz")) if wav_dir.is_dir() else []
    if args.extract and archives:
        print(f"Extracting {len(archives)} archives in {wav_dir} ...")
        extract_archives(wav_dir)
        has_wav = any(any(v) for v in find_wavs(root).values())
    if not has_wav:
        if archives:
            raise SystemExit(
                f"[aishell_import] found {len(archives)} .tar.gz archives in {wav_dir} but no .wav files.\n"
                f"  AISHELL-1 ships one archive per speaker; extract them first, either:\n"
                f"    python aishell_import.py --aishell-dir \"{root}\" --extract\n"
                f"  or by hand inside {wav_dir}:\n"
                f"    for f in *.tar.gz; do tar -xzf \"$f\"; done      (Linux/macOS/Git Bash)\n"
                f"  so that wav/train/S0002/BAC009S0002W0122.wav etc. exist.")
        raise SystemExit(f"[aishell_import] no .wav files under {wav_dir}/{{train,dev,test}}/")

    summary = build_manifests(root, args.out_dir, args.to_traditional, args.max_per_split)

    print("=== AISHELL-1 import done ===")
    for split, n in summary["splits"].items():
        print(f"  {split:5s} -> {summary['paths'][split]}  ({n} utterances)")
    print(f"  skipped: {summary['no_transcript']} wav without transcript, "
          f"{summary['no_wav']} transcript lines without wav")
    print(f"  text: {'Traditional (OpenCC s2t)' if args.to_traditional else 'Simplified (as shipped)'}")
    out = args.out_dir
    print("\nNext:")
    print(f"  python speech_to_text.py --train-manifest \"{out / 'train.json'}\" "
          f"--val-manifest \"{out / 'val.json'}\" --audio-root \"{root}\"")
    if not args.to_traditional:
        print("\nWARNING: these transcripts are Simplified Chinese. Training the recognizer on them is fine\n"
              "  (its targets are pinyin syllables), but do NOT pass them to\n"
              "  `syllable_lm.py --build --extra-text` -- Simplified characters would leak into the\n"
              "  Traditional language model. Re-run with --to-traditional (into another --out-dir) for that.")
    return summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
