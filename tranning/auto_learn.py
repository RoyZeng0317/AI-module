"""auto_learn.py — candidate pool for turning tools.py's live web lookups
into future chats.py training pairs.

Rule 06 boundary: this module never calls another AI model. tools.py's
web_search() already does the only network call (DuckDuckGo's Instant
Answer API, a plain data API) — this module just stores what it found as
{"prompt", "reply", "topic"} candidates, the same shape chats.py's
data/pairs.json already uses. Sinco does NOT learn from these
automatically: they sit in a review queue (list_candidates()) until a human
approves one (approve_candidate()), which appends it to data/pairs.json —
retraining itself still only happens when you run
`python chats.py --data data/pairs.json --epochs N` yourself. No automatic
retraining on approve: the current checkpoint already converged (see
CLAUDE.md 修正日誌 — 1500 epoch, loss 0.0011), and quietly retraining on
every approval risks silently degrading it before anyone notices.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANDIDATES_PATH = DATA_DIR / "auto_learn_candidates.json"
PAIRS_PATH = DATA_DIR / "pairs.json"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def save_candidate(prompt: str, reply: str, topic: str, source: str = "duckduckgo",
                    path: Path | None = None) -> dict:
    """Record a live web-lookup result as a training candidate. Deduped by
    topic (not by exact prompt) — repeatedly asking about the same subject
    in slightly different phrasings shouldn't pile up near-duplicate
    candidates; the first answer found for a topic is kept.
    """
    path = path or CANDIDATES_PATH
    entries = _load(path)
    existing = next((e for e in entries if e["topic"] == topic), None)
    if existing is not None:
        return existing

    entry = {
        "id": uuid.uuid4().hex[:8],
        "prompt": prompt,
        "reply": reply,
        "topic": topic,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entries.append(entry)
    _save(entries, path)
    return entry


def list_candidates(path: Path | None = None) -> list[dict]:
    return _load(path or CANDIDATES_PATH)


def approve_candidate(candidate_id: str, candidates_path: Path | None = None,
                       pairs_path: Path | None = None) -> bool:
    """Move a candidate into the real training set (data/pairs.json) and
    drop it from the review queue. Does NOT retrain — that stays a manual
    `python chats.py --data ...` step you run whenever you're ready.
    """
    candidates_path = candidates_path or CANDIDATES_PATH
    pairs_path = pairs_path or PAIRS_PATH

    entries = _load(candidates_path)
    match = next((e for e in entries if e["id"] == candidate_id), None)
    if match is None:
        return False

    pairs = _load(pairs_path)
    pairs.append({"prompt": match["prompt"], "reply": match["reply"]})
    _save(pairs, pairs_path)

    remaining = [e for e in entries if e["id"] != candidate_id]
    _save(remaining, candidates_path)
    return True


def reject_candidate(candidate_id: str, path: Path | None = None) -> bool:
    path = path or CANDIDATES_PATH
    entries = _load(path)
    remaining = [e for e in entries if e["id"] != candidate_id]
    if len(remaining) == len(entries):
        return False
    _save(remaining, path)
    return True


def format_candidates(entries: list[dict]) -> str:
    if not entries:
        return "（目前沒有待審核的候選學習內容）"
    lines = []
    for e in entries:
        lines.append(f"[{e['id']}] Q: {e['prompt']}\n    A: {e['reply']}")
    return "\n".join(lines)
