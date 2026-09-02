"""build_code_corpus.py — turn the real project files already sitting in
data/program/ into a plain-text corpus for transformer_chat.py's pretrain()
(next-token prediction), so sinco can learn actual code *syntax statistics*
(tag/selector shapes, indentation, how a function/class/query is normally
structured) from real, varied projects instead of being fed hand-picked
{"prompt": "寫一個 X", "reply": code} pairs to memorize.

This is the code-domain counterpart to build_corpus.py (which does the same
job for the Chinese-Wikipedia chat corpus) — same idea, different source:
that script calls a public API, this one just reads local files, because
data/program/ already IS the raw corpus. pretrain() itself needs zero
changes — it already takes any plain UTF-8 .txt file via --corpus, code or
prose makes no difference to next-token prediction.

Why pretrain (unsupervised) and not more {"prompt", "reply"} pairs like
code_snippet_import.py produces for data/code_pairs.json: that pipeline is
retrieve-or-memorize by design (see code_retrieval.py's docstring) — good
for reproducing an exact known snippet, but "不能死記硬背" (must not just
rote-memorize) needs the model to have internalized generalizable code
structure, which only next-token prediction over many *different* real
files teaches.

Covers .html/.css/.js/.ts/.tsx/.py/.sql (widened from the first HTML/CSS-only
pass — 使用者這輪明確要求前端＋Python＋全端專案的語料). Extending EXT_HEADER
below with another extension -> comment-prefix mapping is enough to widen
this further later.

**Data-hygiene filtering (this is the part that matters for 過擬合／死記硬背,
not just "does it run")** — data/program/ now includes full clones of
deployed full-stack projects, which drag in things that would actively hurt
a next-token corpus if left in:

  - third-party dependency trees (node_modules/.venv/site-packages/…):
    thousands of files of someone else's library code, not this user's own
    code, would completely dilute a corpus meant to teach *this project's*
    coding style/patterns. Pruned during directory traversal (never even
    read), not filtered after the fact — a naive glob would still descend
    into a 63,000-file node_modules tree before throwing the results away.
  - build output (dist*/build*/.next/.firebase) and minified files
    (*.min.*): machine-generated, not representative hand-written code, and
    often huge (a single Vite bundle here is 1.3MB — that alone would
    dominate the entire corpus's token distribution if included).
  - a hard per-file size cap (DEFAULT_MAX_FILE_SIZE): even outside the
    excluded directories, a handful of individual files are outliers (a
    488KB vendored xterm.js sitting directly in a project folder, 140-212KB
    raw SQL bulk-INSERT data dumps duplicated under two different
    subprojects) — a single such file would get seen far more often, per
    token, than everything else combined, which is exactly the "memorize
    one giant blob" failure mode 過擬合/死記硬背 warns about, just at the
    level of one file instead of one small dataset. Capping file size is a
    blunt but effective guard against both vendor bundles and bulk data
    dumps without needing a fragile per-vendor filename blocklist.
  - exact-duplicate content across the whole corpus (content hash, not
    just filename): several of these real projects reuse identical files
    across subprojects (e.g. the same exam-question SQL dump copied under
    two different app folders, a shared login.css/register.css pair) —
    keeping every copy would just double-weight that one file's patterns
    in training instead of adding real variety.

None of this fixes overfitting at the *model/training* level — that is
still transformer_chat.py's job (dropout, weight decay, ReduceLROnPlateau,
early stopping restoring the best val checkpoint, printed over/underfitting
warnings, all already in `_run_epochs()`). Corpus-side cleaning here can only
make sure the *data itself* isn't quietly pushing the model toward
memorizing a handful of oversized/duplicated files instead of learning the
patterns that actually repeat across many different real files.

Each file is prefixed with a syntactically-valid same-language comment
naming its path (an HTML comment before HTML content, a `#` comment before
Python, etc.) so the model sees a real, in-language "a new file starts here"
signal in the token stream, instead of an arbitrary marker that never
appears in real code and would just be noise to it. Files are read as whole
units (not split into functions like code_snippet_import.py does) because
next-token pretraining wants realistic *whole-file* context.

Usage:
    python build_code_corpus.py --source ../data/program --out ../data/corpus_code_fullstack.txt
"""

import argparse
import hashlib
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = _PROJECT_ROOT / "data" / "program"
DEFAULT_OUT = _PROJECT_ROOT / "data" / "corpus_code_fullstack.txt"

# individual files above this size are dropped even if their directory
# wasn't already excluded (see module docstring: catches vendored libraries
# and bulk data dumps that don't happen to live under a recognizably-named
# vendor directory). 50KB is generous relative to every hand-written file
# actually seen in this project's own data/program/ folders (a few KB each);
# it is not generous relative to a 1.3MB build bundle or a 200KB SQL dump.
DEFAULT_MAX_FILE_SIZE = 50_000

# extension -> (comment_open, comment_close), each language's own real
# comment syntax so the inserted file-boundary marker is valid code, not
# noise the model has to learn to ignore.
EXT_HEADER = {
    ".html": ("<!-- ", " -->"),
    ".css": ("/* ", " */"),
    ".js": ("// ", ""),
    ".ts": ("// ", ""),
    ".tsx": ("// ", ""),
    ".py": ("# ", ""),
    ".sql": ("-- ", ""),
}

# directory names pruned during traversal -- never even descended into, so
# a huge node_modules/.venv tree costs nothing beyond an os.walk() name
# check, not "scan everything then filter". Substring/prefix matching (not
# exact equality) deliberately catches renamed copies actually found in
# this project's data (e.g. "node_modules copy") and hashed build-output
# folders (e.g. "dist-royhomenas"), not just the exact canonical names.
_VENDOR_DIR_EXACT = {".venv", "venv", ".git", "__pycache__", ".next", ".firebase", "site-packages", "vendor"}
# "dist"/"build" only match as a whole word or a hyphen/underscore-prefixed
# word (dist-royhomenas, build_output) -- a plain startswith("dist") would
# also wrongly catch a normal folder name like "distinct-feature".
_VENDOR_DIR_WORD_PREFIXES = ("dist", "build")
_VENDOR_DIR_SUFFIXES = (".egg-info", ".dist-info")


def is_vendor_dir(name: str) -> bool:
    n = name.lower()
    if "node_modules" in n:
        return True
    if n in _VENDOR_DIR_EXACT:
        return True
    for word in _VENDOR_DIR_WORD_PREFIXES:
        if n == word or n.startswith(word + "-") or n.startswith(word + "_"):
            return True
    if n.endswith(_VENDOR_DIR_SUFFIXES):
        return True
    return False


def collect_code_files(source: Path, exts: dict[str, tuple[str, str]] = EXT_HEADER,
                        max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> tuple[list[Path], dict[str, list[str]]]:
    """回傳 (依相對路徑排序的檔案清單, 略過清單)。略過清單分三類：
    "empty"（0 位元組）、"oversized"（超過 max_file_size，見模組 docstring
    的資料清理說明）、"minified"（檔名含 .min.，如 jquery.min.js）。
    node_modules/.venv 等第三方依賴目錄在 os.walk 階段就被剪掉，不會出現在
    任何清單裡（從頭就沒被掃到，不是掃到後才丟棄）。排序讓輸出可重現。
    """
    kept: list[Path] = []
    skipped: dict[str, list[str]] = {"empty": [], "oversized": [], "minified": []}

    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = [d for d in dirnames if not is_vendor_dir(d)]
        for fn in filenames:
            path = Path(dirpath) / fn
            if path.suffix.lower() not in exts:
                continue
            rel = str(path.relative_to(source).as_posix())
            if ".min." in fn.lower():
                skipped["minified"].append(rel)
                continue
            size = path.stat().st_size
            if size == 0:
                skipped["empty"].append(rel)
                continue
            if size > max_file_size:
                skipped["oversized"].append(rel)
                continue
            kept.append(path)

    kept.sort(key=lambda p: str(p.relative_to(source)).lower())
    return kept, skipped


def build_corpus(source: Path, out_path: Path, exts: dict[str, tuple[str, str]] = EXT_HEADER,
                  max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> dict:
    files, skipped = collect_code_files(source, exts, max_file_size)
    skipped["duplicate"] = []

    chunks = []
    seen_hashes: set[str] = set()
    for path in files:
        rel = path.relative_to(source)
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            skipped["empty"].append(rel.as_posix())
            continue

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            # 內容跟先前已收錄的某個檔案完全一樣（同一個檔案被複製到多個
            # 子專案下）——只留第一次出現（依排序後的相對路徑，可重現），
            # 避免同一段內容被重複計入權重，見模組 docstring 的說明。
            skipped["duplicate"].append(rel.as_posix())
            continue
        seen_hashes.add(digest)

        open_c, close_c = exts[path.suffix.lower()]
        chunks.append(f"{open_c}{rel.as_posix()}{close_c}\n{text}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_text = "\n\n".join(chunks)
    out_path.write_text(corpus_text, encoding="utf-8")

    return {
        "files_written": len(chunks),
        "skipped": skipped,
        "out_path": str(out_path),
        "total_chars": len(corpus_text),
    }


def _print_skip_list(label: str, names: list[str], limit: int = 20) -> None:
    if not names:
        return
    print(f"略過 {len(names)} 個{label}：")
    for name in names[:limit]:
        print(f"  - {name}")
    if len(names) > limit:
        print(f"  ...（其餘 {len(names) - limit} 個省略）")


def main():
    parser = argparse.ArgumentParser(
        description="Build a plain-text code corpus from data/program/ for transformer_chat.py pretrain()"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-file-size", type=int, default=DEFAULT_MAX_FILE_SIZE,
                         help="超過這個位元組數的單一檔案不納入語料（防止 vendor 大檔/資料傾印獨佔訓練權重）")
    args = parser.parse_args()

    summary = build_corpus(args.source, args.out, max_file_size=args.max_file_size)
    print(f"共納入 {summary['files_written']} 個檔案，寫入 {summary['out_path']}"
          f"（共 {summary['total_chars']} 字元）")
    _print_skip_list("空檔案", summary["skipped"]["empty"])
    _print_skip_list("超過大小上限的檔案", summary["skipped"]["oversized"])
    _print_skip_list("已壓縮(.min.)檔案", summary["skipped"]["minified"])
    _print_skip_list("重複內容的檔案", summary["skipped"]["duplicate"])


if __name__ == "__main__":
    main()
