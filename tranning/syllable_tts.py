"""Speech synthesis (TTS) by syllable concatenation -- sinco reads its replies
aloud in YOUR OWN recorded voice, no cloud TTS, no pretrained voice (Rule 06).

How it works: Mandarin text -> tonal pinyin syllables (pinyin_units.py, e.g.
"你好" -> ni3 hao3) -> one short recording per syllable, glued together with
silence trimmed, loudness matched, a short crossfade between syllables and a
pause at punctuation. It sounds robotic (each syllable is said in isolation,
there is no sentence melody), but it works as soon as the syllables are
recorded, with zero training. voice_clone.py (Tacotron-style, needs a
recorded sentence dataset + training) can replace it later -- see
synthesize(backend="clone").

Recordings live in data/voice_syllables/<syllable>.wav ("ni3.wav", neutral
tone = 5). A missing tone falls back to another tone of the same syllable
("hao4" -> hao3.wav), so ~400 recordings already cover every syllable; the
rest only improve the tones.

Usage:
    python syllable_tts.py --coverage                 # what is still missing for sinco's replies
    python syllable_tts.py --record                   # record missing syllables, most useful first
    python syllable_tts.py --record --limit 50        # ... only 50 this session
    python syllable_tts.py --import-dir my_wavs/      # copy ni3.wav-style files in
    python syllable_tts.py --say "你好，我是 sinco。"  # speak (needs sounddevice)
    python syllable_tts.py --say "你好" --out hi.wav   # write a WAV instead
"""

import argparse
import json
import re
import shutil
import threading
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from pinyin_units import PAUSE, strip_tone, to_syllables
from speech_to_text import SAMPLE_RATE, load_waveform

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DEFAULT_VOICE_DIR = DATA / "voice_syllables"
REPLY_SOURCE = DATA / "pairs.json"

SYLLABLE_RE = re.compile(r"^[a-z]+[1-5]$")
CROSSFADE = int(0.015 * SAMPLE_RATE)      # 15 ms overlap between neighbouring syllables
PAUSE_SAMPLES = int(0.25 * SAMPLE_RATE)   # silence at punctuation
MAX_SYLLABLE = int(0.45 * SAMPLE_RATE)    # isolated syllables are said slowly; cap their length
TARGET_RMS = 0.08
RECORD_SECONDS = 1.2

NOT_RECORDED = "語音合成尚未錄製音節，請先執行 `python tranning/syllable_tts.py --record`。"


# --- what to record ------------------------------------------------------------------

def reply_texts(path: Path = REPLY_SOURCE) -> list[str]:
    """sinco's own replies: what the synthesizer will actually have to say."""
    if not Path(path).exists():
        return []
    return [p.get("reply", "") for p in json.loads(Path(path).read_text(encoding="utf-8"))]


def recording_list(texts: list[str]) -> list[str]:
    """Tonal syllables in the order they are worth recording: first the most
    common tone of every base syllable (most frequent base first -- after
    these every syllable can be spoken, thanks to the tone fallback), then
    the remaining tones by frequency."""
    counts = Counter(s for t in texts for s in to_syllables(t))
    by_base: dict[str, list[str]] = {}
    for syl, _ in counts.most_common():
        by_base.setdefault(strip_tone(syl), []).append(syl)
    base_order = sorted(by_base, key=lambda b: -sum(counts[s] for s in by_base[b]))
    first = [by_base[b][0] for b in base_order]
    rest = sorted((s for b in base_order for s in by_base[b][1:]), key=lambda s: -counts[s])
    return first + rest


def recorded(voice_dir: Path = DEFAULT_VOICE_DIR) -> set[str]:
    voice_dir = Path(voice_dir)
    if not voice_dir.exists():
        return set()
    return {p.stem for p in voice_dir.glob("*.wav") if SYLLABLE_RE.match(p.stem)}


def coverage(texts: list[str], voice_dir: Path = DEFAULT_VOICE_DIR) -> dict:
    """Share of syllable occurrences in `texts` that can be spoken exactly
    (right tone) and at all (any tone of the same syllable)."""
    have = recorded(voice_dir)
    bases = {strip_tone(s) for s in have}
    sylls = [s for t in texts for s in to_syllables(t)]
    total = len(sylls) or 1
    return {
        "recorded": len(have),
        "exact": sum(s in have for s in sylls) / total,
        "any_tone": sum(strip_tone(s) in bases for s in sylls) / total,
        "missing": [s for s in recording_list(texts) if s not in have],
    }


# --- synthesis -------------------------------------------------------------------------------

def _trim(wave_: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """Cut leading/trailing silence (frames quieter than threshold x peak)."""
    if wave_.size == 0:
        return wave_
    frame = 160
    n = len(wave_) // frame
    if n == 0:
        return wave_
    energy = np.abs(wave_[: n * frame]).reshape(n, frame).max(axis=1)
    loud = np.nonzero(energy > threshold * max(energy.max(), 1e-6))[0]
    if loud.size == 0:
        return wave_[:0]
    return wave_[loud[0] * frame:(loud[-1] + 1) * frame]


def _prepare(wave_: np.ndarray) -> np.ndarray:
    wave_ = _trim(wave_)
    if wave_.size == 0:
        return wave_
    if len(wave_) > MAX_SYLLABLE:                      # keep the onset, fade the long tail out
        wave_ = wave_[:MAX_SYLLABLE].copy()
        fade = min(len(wave_), int(0.06 * SAMPLE_RATE))
        wave_[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    rms = float(np.sqrt(np.mean(wave_ ** 2))) or 1e-6
    return np.clip(wave_ * (TARGET_RMS / rms), -1, 1).astype(np.float32)


class SyllableVoice:
    """Loaded, cleaned syllable recordings of one voice directory."""

    def __init__(self, voice_dir: Path = DEFAULT_VOICE_DIR):
        self.voice_dir = Path(voice_dir)
        self.files = recorded(self.voice_dir)
        self.units: dict[str, np.ndarray] = {}
        for name in sorted(self.files):
            clip = _prepare(load_waveform(self.voice_dir / f"{name}.wav"))
            if clip.size:
                self.units[name] = clip
        self.by_base: dict[str, list[str]] = {}
        for name in self.units:
            self.by_base.setdefault(strip_tone(name), []).append(name)

    def lookup(self, syllable: str) -> np.ndarray | None:
        if syllable in self.units:
            return self.units[syllable]
        others = self.by_base.get(strip_tone(syllable))
        return self.units[others[0]] if others else None

    def synthesize(self, text: str) -> tuple[np.ndarray, list[str]]:
        """(waveform, syllables that had no recording at all). Unrecorded
        syllables become a short silence so the rhythm stays intact."""
        pieces: list[np.ndarray] = []
        missing: list[str] = []
        for syl in to_syllables(text, keep_pauses=True):
            if syl == PAUSE:
                pieces.append(np.zeros(PAUSE_SAMPLES, dtype=np.float32))
                continue
            clip = self.lookup(syl)
            if clip is None:
                missing.append(syl)
                clip = np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)
            pieces.append(clip)
        return _concat(pieces), missing


def _concat(pieces: list[np.ndarray]) -> np.ndarray:
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    out = pieces[0].astype(np.float32)
    for nxt in pieces[1:]:
        k = min(CROSSFADE, len(out), len(nxt))
        if k == 0:
            out = np.concatenate([out, nxt])
            continue
        ramp = np.linspace(0, 1, k, dtype=np.float32)
        mixed = out[-k:] * (1 - ramp) + nxt[:k] * ramp
        out = np.concatenate([out[:-k], mixed, nxt[k:]])
    # short silence at both ends so playback does not clip the first sound
    pad = np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([pad, out, pad])


_voices: dict[str, SyllableVoice] = {}


def _voice(voice_dir: Path) -> SyllableVoice:
    key = str(Path(voice_dir).resolve())
    cached = _voices.get(key)
    if cached is None or recorded(voice_dir) != cached.files:
        cached = _voices[key] = SyllableVoice(voice_dir)   # reload after new recordings
    return cached


def synthesize(text: str, voice_dir: Path = DEFAULT_VOICE_DIR, backend: str = "concat"):
    """Text -> 1-D float32 waveform at SAMPLE_RATE, or a str message (like the
    other sinco models) when nothing can be spoken yet.

    backend="concat" (default): the syllable recordings above.
    backend="clone": voice_clone.py's trained Tacotron-style model instead
    (its own placeholder message if that is not trained yet).
    """
    if backend == "clone":
        from voice_clone import clone_voice   # heavy import (chats.py), only when asked for
        return clone_voice(text)
    voice = _voice(voice_dir)
    if not voice.units:
        return NOT_RECORDED
    waveform, _ = voice.synthesize(text)
    return waveform


def save_wav(waveform: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# --- playback ------------------------------------------------------------------------------

_speak_lock = threading.Lock()


def speak(text: str, voice_dir: Path = DEFAULT_VOICE_DIR, block: bool = False) -> str | None:
    """Synthesize and play through the default speaker. Returns None when
    playing, or the reason it could not (not recorded / no sounddevice).
    A new call interrupts whatever is still playing."""
    result = synthesize(text, voice_dir)
    if isinstance(result, str):
        return result
    if result.size == 0:
        return None
    try:
        import sounddevice as sd
    except ImportError:
        return "語音播放需要 sounddevice 套件（pip install sounddevice）。"
    with _speak_lock:
        sd.stop()
        sd.play(result, SAMPLE_RATE)
    if block:
        sd.wait()
    return None


def speak_async(text: str, voice_dir: Path = DEFAULT_VOICE_DIR,
                on_error=None) -> threading.Thread:
    """speak() on a background thread (synthesis takes a moment for long
    replies; a Tk/CLI caller must not freeze). on_error(msg) gets the reason
    if nothing could be played."""
    def run():
        try:
            msg = speak(text, voice_dir)
        except Exception as exc:          # audio device errors must never kill the chat
            msg = f"語音播放失敗：{exc}"
        if msg and on_error:
            on_error(msg)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


# --- recording / import ------------------------------------------------------------------

def example_chars(texts: list[str]) -> dict[str, str]:
    """A character for each tonal syllable, to show what to say while recording."""
    from pinyin_units import CJK_RE
    examples: dict[str, str] = {}
    for t in texts:
        chars = [c for c in t if CJK_RE.match(c)]
        sylls = to_syllables("".join(chars))
        if len(chars) == len(sylls):
            for c, s in zip(chars, sylls):
                examples.setdefault(s, c)
    return examples


def record_session(voice_dir: Path = DEFAULT_VOICE_DIR, limit: int | None = None,
                   texts: list[str] | None = None) -> int:
    """Interactive: prompts each missing syllable, records RECORD_SECONDS,
    plays it back; Enter keeps it, r redoes, s skips, q quits. Returns how
    many were saved."""
    import sounddevice as sd

    texts = reply_texts() if texts is None else texts
    todo = coverage(texts, voice_dir)["missing"]
    if limit:
        todo = todo[:limit]
    examples = example_chars(texts)
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} syllables to record. Say each one clearly, once, right after you press Enter.")
    saved = 0
    for i, syl in enumerate(todo, 1):
        while True:
            hint = examples.get(syl, "")
            cmd = input(f"[{i}/{len(todo)}] {syl}  {hint}  -- Enter=record  s=skip  q=quit > ").strip().lower()
            if cmd == "q":
                return saved
            if cmd == "s":
                break
            audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            audio = audio[:, 0]
            if _trim(audio).size == 0 or np.abs(audio).max() < 0.02:
                print("  (too quiet -- try again, closer to the microphone)")
                continue
            sd.play(_prepare(audio), SAMPLE_RATE)
            sd.wait()
            if input("  Enter=keep  r=redo > ").strip().lower() == "r":
                continue
            save_wav(audio, voice_dir / f"{syl}.wav")
            saved += 1
            break
    return saved


def import_dir(src: Path, voice_dir: Path = DEFAULT_VOICE_DIR) -> list[str]:
    """Copy WAV files named like a tonal syllable (ni3.wav, Hao3.WAV) into the
    voice directory. Returns the imported syllables; other files are ignored."""
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    imported = []
    for p in sorted(Path(src).iterdir()):
        name = p.stem.lower().replace("ü", "v")
        if p.suffix.lower() == ".wav" and SYLLABLE_RE.match(name):
            load_waveform(p)                     # raises on unsupported formats before copying
            shutil.copyfile(p, voice_dir / f"{name}.wav")
            imported.append(name)
    return imported


def main():
    ap = argparse.ArgumentParser(description="Syllable-concatenation speech synthesis in your own voice")
    ap.add_argument("--voice-dir", type=Path, default=DEFAULT_VOICE_DIR)
    ap.add_argument("--coverage", action="store_true", help="how much of sinco's replies can be spoken")
    ap.add_argument("--record", action="store_true", help="record missing syllables (most useful first)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--import-dir", type=Path, default=None, help="folder of ni3.wav-style recordings")
    ap.add_argument("--say", default=None, help="text to speak")
    ap.add_argument("--out", type=Path, default=None, help="with --say: write a WAV instead of playing")
    args = ap.parse_args()

    if args.import_dir:
        print(f"imported {len(import_dir(args.import_dir, args.voice_dir))} syllables")
    if args.record:
        print(f"saved {record_session(args.voice_dir, args.limit)} recordings")
    if args.coverage or args.record or args.import_dir:
        cov = coverage(reply_texts(), args.voice_dir)
        print(f"recorded {cov['recorded']} syllables; sinco's replies: {cov['exact']:.0%} exact tone, "
              f"{cov['any_tone']:.0%} speakable; next to record: {' '.join(cov['missing'][:20])}")
    if args.say:
        if args.out:
            result = synthesize(args.say, args.voice_dir)
            if isinstance(result, str):
                print(result)
            else:
                save_wav(result, args.out)
                print(f"wrote {args.out}")
        else:
            msg = speak(args.say, args.voice_dir, block=True)
            if msg:
                print(msg)


if __name__ == "__main__":
    main()
