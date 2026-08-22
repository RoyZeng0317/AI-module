"""code_snippet_import.py — 把手上一批 GitHub 專案的原始碼檔案（.py/.c/.h/.cpp/
.cc/.cxx/.hpp/.cs）轉換成 chats.py 程式碼 checkpoint（tranning/code_runs/，資料來源
data/code_pairs.json）能吃的 {"prompt": ..., "reply": ...} 訓練配對。

沿用單一 code_runs checkpoint，不新增每語言獨立 checkpoint：Python/C/C++/C# 混在
同一份 data/code_pairs.json，靠 prompt 裡的語言字樣（例如「用 C++ 寫一個 XXX」）
讓 chats.py 現有的 seq2seq 模型自己學會依語言分辨，不用改訓練程式碼或新增第二個
checkpoint 路徑。

兩階段，中間先落地成可以人工檢視/修改的 markdown（原始需求：先轉 markdown 再
餵養），不是原始碼掃完直接無人把關寫進訓練資料：

  extract   <source>/**/*.{py,c,h,cpp,cc,cxx,hpp,hh,cs}
            -> <out>/**/*.md（目錄結構鏡射原始碼），每個抓到的函式各自一個
            「## 標題」+ 語言標註的 fenced code block。標題優先取函式緊鄰的
            docstring／註解，抓不到就退回函式（或「類別.方法」）名稱本身。

            這一步是「盡量抓」的 heuristic 掃描，不是完整語法剖析器：
            Python 用標準庫 ast 模組精準剖析（含正確的縮排/巢狀範圍），
            C/C++/C# 沒有 stdlib 可解的 AST，用函式簽名 regex + 大括號配對
            （會跳過字串/字元常值與註解裡的大括號，避免算錯配對，但終究是
            heuristic）——樣板/巨集展開/多行泛型簽名等複雜語法抓不到或抓錯
            都可能發生。設計上預期使用者會先看過 <out> 底下的 markdown 再
            進 pairs 這一步，而不是把 extract 的結果直接當成無人把關的
            真實標籤。

  pairs     <source>/**/*.md（extract 產出的、或使用者自己手動整理好的，
            格式只要是「## 標題」後面接一個 fenced code block」都吃）
            -> 追加進 data/code_pairs.json 的 {"prompt": ..., "reply": ...}。
            prompt 用「用 <語言> 寫一個 <標題>」模板組出來（標題本身已經是
            中文語句就不重複套模板）；reply 是 fenced code block 原文。
            合併既有檔案／去重／可選的 train-val 切分直接重用
            dataset_import.py 的 write_manifest()，輸出格式跟 pairs 模式
            完全相容，可以混著用同一個 --out。

Usage:
    python code_snippet_import.py extract --source <GitHub專案資料夾> --out <markdown輸出資料夾>
    # ...人工看一眼／修改 <markdown輸出資料夾> 底下的 .md...
    python code_snippet_import.py pairs --source <markdown輸出資料夾> --out ../data/code_pairs.json
    python code_snippet_import.py pairs --source <markdown輸出資料夾> --out ../data/code_pairs.json --val-ratio 0.2
"""

import argparse
import ast
import re
from pathlib import Path

from dataset_import import write_manifest

LANG_BY_EXT = {
    ".py": "python",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cs": "csharp",
}

LANG_LABEL = {"python": "Python", "c": "C", "cpp": "C++", "csharp": "C#"}

# C/C++/C# 共用的函式簽名 heuristic：一或多個「型別字」+ 函式名稱
# （允許 Class::method / ~destructor）+ 括號參數 + 可選的 const/override/
# noexcept 修飾詞，最後接開大括號。抓不到多行簽名、泛型模板等複雜寫法。
_CFAMILY_SIG_RE = re.compile(
    r"^[ \t]*"
    r"(?P<ret>(?:[A-Za-z_]\w*(?:::\w+)*[*&]?\s+)+)"
    r"(?P<name>~?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)?)"
    r"\s*\((?P<args>[^;{}()]*)\)\s*"
    r"(?:const\s*)?(?:override\s*)?(?:noexcept\s*)?"
    r"\{",
    re.MULTILINE,
)

# 誤判防呆：if/for/while 這類控制結構在「前面沒有其他型別字」時本來就不會
# match（ret 至少要求一個 token），但 "else if (...) {" 這種 else 會被
# regex 誤當成 ret，把 name 抓成 "if"，這裡額外擋掉。
_CONTROL_KEYWORDS = {"if", "for", "while", "switch", "catch", "else", "return", "do"}

_MD_UNIT_RE = re.compile(
    r"^##\s+(?P<heading>.+?)\s*$\n+```(?P<lang>\w*)\n(?P<code>.*?)```",
    re.MULTILINE | re.DOTALL,
)

_CJK_RE = re.compile(r"[一-鿿]")


def _heading_for_python(node) -> str:
    doc = ast.get_docstring(node)
    if doc:
        first_line = doc.strip().splitlines()[0].strip()
        if first_line:
            return first_line
    return node.name


def extract_python_units(text: str) -> list[tuple[str, str]]:
    """回傳頂層函式與類別方法的 [(標題, 原始碼片段), ...]，用 ast 精準剖析。
    語法錯誤的檔案直接回傳空清單（讓呼叫端印出略過訊息），不炸掉整個掃描。
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    units: list[tuple[str, str]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(text, node)
            if seg:
                units.append((_heading_for_python(node), seg))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    seg = ast.get_source_segment(text, sub)
                    if seg:
                        units.append((f"{node.name}.{_heading_for_python(sub)}", seg))
    return units


def _find_matching_brace(text: str, open_index: int) -> int | None:
    """從 text[open_index]（必須是 '{'）往後找配對的 '}'，跳過字串/字元
    常值與註解裡的大括號，避免算錯配對深度。找不到（例如檔案被截斷）回傳
    None。
    """
    depth = 0
    i = open_index
    n = len(text)
    in_str = in_char = in_line_comment = in_block_comment = False
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
        elif in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif in_char:
            if c == "\\":
                i += 1
            elif c == "'":
                in_char = False
        elif c == "/" and nxt == "/":
            in_line_comment = True
            i += 1
        elif c == "/" and nxt == "*":
            in_block_comment = True
            i += 1
        elif c == '"':
            in_str = True
        elif c == "'":
            in_char = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _heading_for_c_family(text: str, sig_start: int, name: str) -> str:
    """簽名前面緊鄰的 // 或 /* */ 註解區塊當標題，往上掃到第一個空行或非
    註解行為止；沒有註解就退回函式名稱本身。
    """
    prefix = text[:sig_start].rstrip("\n")
    lines = prefix.split("\n")
    collected: list[str] = []
    i = len(lines) - 1
    while i >= 0:
        line = lines[i].strip()
        if line == "":
            if collected:
                break
            i -= 1
            continue
        if line.startswith("//"):
            collected.insert(0, line[2:].strip())
        elif line.startswith("*"):
            collected.insert(0, line[1:].strip())
        elif line.startswith("/*") or line.endswith("*/"):
            cleaned = line.strip("/*").strip()
            if cleaned:
                collected.insert(0, cleaned)
        else:
            break
        i -= 1
    joined = " ".join(part for part in collected if part)
    return joined or name


def extract_c_family_units(text: str) -> list[tuple[str, str]]:
    """回傳 [(標題, 原始碼片段), ...]，用簽名 regex + 大括號配對抓函式本體。
    C/C++/C# 共用同一套 heuristic（語法夠像）。抓不到符合的函式時回傳空清單。
    """
    units: list[tuple[str, str]] = []
    for m in _CFAMILY_SIG_RE.finditer(text):
        name = m.group("name")
        if name in _CONTROL_KEYWORDS:
            continue
        open_brace = m.end() - 1
        close_brace = _find_matching_brace(text, open_brace)
        if close_brace is None:
            continue
        snippet = text[m.start():close_brace + 1].strip()
        heading = _heading_for_c_family(text, m.start(), name)
        units.append((heading, snippet))
    return units


def units_for_source(path: Path) -> tuple[str, list[tuple[str, str]]] | None:
    lang = LANG_BY_EXT.get(path.suffix.lower())
    if lang is None:
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    units = extract_python_units(text) if lang == "python" else extract_c_family_units(text)
    return lang, units


def write_markdown_for_units(lang: str, units: list[tuple[str, str]], out_path: Path) -> None:
    parts = [f"## {heading}\n\n```{lang}\n{code.strip()}\n```\n" for heading, code in units]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def extract_source_tree(source: Path, out: Path) -> dict:
    """掃描 source 底下所有支援的原始碼檔案，逐一寫出鏡射目錄結構的 markdown
    到 out。回傳統計摘要（供 CLI 印出訊息、測試斷言使用）。
    """
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in LANG_BY_EXT)
    written: list[Path] = []
    empty: list[str] = []
    for path in files:
        lang, units = units_for_source(path)
        if not units:
            empty.append(str(path.relative_to(source)))
            continue
        out_path = out / path.relative_to(source).with_suffix(".md")
        write_markdown_for_units(lang, units, out_path)
        written.append(out_path)
    return {"scanned": len(files), "written": written, "empty": empty}


def parse_markdown_units(text: str) -> list[tuple[str, str, str]]:
    """回傳 markdown 內所有 [(標題, 語言, 程式碼), ...]（"## 標題" 後緊接一個
    fenced code block 才算一筆）。"""
    return [
        (m.group("heading").strip(), m.group("lang").strip().lower(), m.group("code").strip())
        for m in _MD_UNIT_RE.finditer(text)
    ]


def build_prompt(heading: str, lang: str) -> str:
    """把「標題 + 語言」組成訓練用的中文 prompt。標題本身若已經是中文語句
    （extract 抓到 docstring/註解的情況）就不重複套「用 XX 寫一個」模板；
    只有標題退回成函式識別字（英文名稱）時才套模板。
    """
    label = LANG_LABEL.get(lang, lang)
    if _CJK_RE.search(heading):
        if label and label not in heading:
            return f"用 {label} {heading}"
        return heading
    if label:
        return f"用 {label} 寫一個 {heading}"
    return f"寫一個 {heading}"


def collect_code_pairs(source: Path) -> list[dict]:
    entries = []
    for path in sorted(source.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for heading, lang, code in parse_markdown_units(text):
            if not code:
                continue
            entries.append({"prompt": build_prompt(heading, lang), "reply": code})
    return entries


def cmd_extract(args) -> None:
    summary = extract_source_tree(args.source, args.out)
    print(f"掃描 {summary['scanned']} 個原始碼檔案，產出 {len(summary['written'])} 個 markdown 檔案到 {args.out}")
    if summary["empty"]:
        empty = summary["empty"]
        print(f"{len(empty)} 個檔案沒抓到任何函式（可能是純宣告/巨集檔，或簽名不符合 heuristic），略過：")
        for name in empty[:20]:
            print(f"  - {name}")
        if len(empty) > 20:
            print(f"  ...（其餘 {len(empty) - 20} 個省略）")


def cmd_pairs(args) -> None:
    entries = collect_code_pairs(args.source)
    if not entries:
        print(f"沒有在 {args.source} 底下找到任何可用的 markdown 程式碼區塊。")
    summary = write_manifest(entries, args.out, val_ratio=args.val_ratio, seed=args.seed)
    print(f"共匯入 {summary['total']} 筆")
    if summary["val_path"]:
        print(f"  train: {summary['train_path']}（{summary['train_count']} 筆）")
        print(f"  val:   {summary['val_path']}（{summary['val_count']} 筆）")
    else:
        print(f"  寫入: {summary['train_path']}")
        if args.val_ratio > 0:
            print("  （筆數不足 2 筆，無法切出 held-out val，全部寫進同一個檔案）")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="原始碼資料夾 -> markdown 資料夾")
    p_extract.add_argument("--source", type=Path, required=True)
    p_extract.add_argument("--out", type=Path, required=True)
    p_extract.set_defaults(func=cmd_extract)

    p_pairs = sub.add_parser("pairs", help="markdown 資料夾 -> code_pairs.json 追加")
    p_pairs.add_argument("--source", type=Path, required=True)
    p_pairs.add_argument("--out", type=Path, required=True)
    p_pairs.add_argument("--val-ratio", type=float, default=0.0)
    p_pairs.add_argument("--seed", type=int, default=42)
    p_pairs.set_defaults(func=cmd_pairs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
