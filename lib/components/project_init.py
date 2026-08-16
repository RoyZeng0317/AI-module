"""project_init.py — CLAUDE.md 需求 #01 的 `/init` 指令：掃描使用者指定的
任意專案資料夾，產生/更新一份 CLAUDE.md 摘要。

刻意不靠模型生成內容：sinco 目前的聊天模型（tranning/chats.py）是從零
訓練的小型 seq2seq，沒有真正讀懂任意程式碼庫的能力，硬要模型「生成」
專案摘要只會編造內容。改用 deterministic 的檔案樹分析（偵測技術棧標記檔、
列出頂層檔案/資料夾、統計副檔名分佈）——跟 Claude Code 的 /init 目標一樣
（讓 AI 助理對這個專案有基本認識），但誠實地只做結構列表層級的事，不假裝
理解商業邏輯。

重新執行 /init 只會更新 CLAUDE.md 裡 sinco:init 標記之間的區塊，區塊外
使用者自己寫的規則/需求完全不會被動到；沒有 CLAUDE.md 就整份新建。
"""

import re
from pathlib import Path

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".next", "target", ".pytest_cache", ".mypy_cache", ".idea", ".vscode", "__MACOSX",
}

# 檔名 -> 技術棧標籤；用「這個檔案存在」當作證據，不去解析內容
STACK_MARKERS = {
    "package.json": "Node.js / JavaScript / TypeScript",
    "pyproject.toml": "Python",
    "requirements.txt": "Python",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java/Kotlin (Gradle)",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "CMakeLists.txt": "C/C++ (CMake)",
}

START_MARK = "<!-- sinco:init:start -->"
END_MARK = "<!-- sinco:init:end -->"


def detect_stack(root: Path) -> list[str]:
    found = [label for marker, label in STACK_MARKERS.items() if (root / marker).exists()]
    return found or ["未偵測到已知的語言/框架標記檔案"]


def top_level_tree(root: Path, max_entries: int = 40) -> list[str]:
    entries = sorted(
        (p for p in root.iterdir() if p.name not in IGNORE_DIRS and not p.name.startswith(".")),
        key=lambda p: (p.is_file(), p.name.lower()),
    )
    lines = []
    for entry in entries[:max_entries]:
        lines.append(f"- {entry.name}{'/' if entry.is_dir() else ''}")
    if len(entries) > max_entries:
        lines.append(f"- …（還有 {len(entries) - max_entries} 項未列出）")
    return lines or ["（空資料夾）"]


def extension_histogram(root: Path, max_files: int = 5000) -> dict[str, int]:
    counts: dict[str, int] = {}
    scanned = 0
    for path in root.rglob("*"):
        if scanned >= max_files:
            break
        if any(part in IGNORE_DIRS or part.startswith(".") for part in path.relative_to(root).parts[:-1]):
            continue
        if path.is_file():
            scanned += 1
            ext = path.suffix.lower() or "(無副檔名)"
            counts[ext] = counts.get(ext, 0) + 1
    return counts


def build_summary(root: Path) -> str:
    stack = detect_stack(root)
    tree = top_level_tree(root)
    top_exts = sorted(extension_histogram(root).items(), key=lambda kv: -kv[1])[:8]

    lines = [
        START_MARK,
        f"# {root.name}",
        "",
        "> 這個區塊由 sinco 的 `/init` 指令自動產生（deterministic 檔案樹分析，"
        "不是模型生成的程式碼理解——sinco 的聊天模型是從零訓練的小型 seq2seq，"
        "還沒有真正讀懂任意程式碼庫的能力）。重新執行 `/init` 只會更新這個"
        "區塊，區塊外你自己寫的內容不會被動到。",
        "",
        "## 偵測到的技術棧",
        *[f"- {s}" for s in stack],
        "",
        "## 頂層檔案/資料夾",
        *tree,
        "",
        "## 主要副檔名分佈（依檔案數排序，已忽略常見雜訊資料夾）",
        *([f"- {ext}: {count}" for ext, count in top_exts] or ["（沒有掃描到檔案）"]),
        "",
        END_MARK,
    ]
    return "\n".join(lines)


def write_claude_md(root: Path) -> Path:
    path = root / "CLAUDE.md"
    summary = build_summary(root)

    if not path.exists():
        path.write_text(summary + "\n", encoding="utf-8")
        return path

    existing = path.read_text(encoding="utf-8")
    if START_MARK in existing and END_MARK in existing:
        pattern = re.compile(re.escape(START_MARK) + r".*?" + re.escape(END_MARK), re.DOTALL)
        updated = pattern.sub(summary, existing)
    else:
        updated = existing.rstrip("\n") + "\n\n" + summary + "\n"
    path.write_text(updated, encoding="utf-8")
    return path
