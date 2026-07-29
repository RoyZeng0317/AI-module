"""sinco CLI —— 終端機聊天介面，介面設計仿照 Claude Code CLI（橫幅、斜線指令、
「思考中」狀態列），但後端跟 Tk 桌面版（home_screen.py + conversation.py）共用
同一套邏輯：chats.smart_reply_traced() 負責回覆、app/command/*.md 負責斜線指令
的提示詞內容。這裡不新增任何模型或路由邏輯，純粹是第二個前端介面。

Usage:
    python cli.py
    python cli.py --character 周柯宇   # 啟動時就切換到指定角色人格
"""

import argparse
import json
import sys
from pathlib import Path

APP_COMPONENTS_DIR = Path(__file__).resolve().parent
TRANNING_DIR = APP_COMPONENTS_DIR.parent.parent / "tranning"
COMMAND_DIR = APP_COMPONENTS_DIR.parent / "command"
sys.path.insert(0, str(TRANNING_DIR))

# 台灣 Windows 的傳統主控台編碼是 cp950（Big5），沒有涵蓋 rich 用到的一些符號
# （如 "›"、spinner 用的點字字元）。rich 偵測到「legacy windows console」時會
# 改用 Win32 API 直接照系統代碼頁編碼輸出，遇到這些字元會直接 UnicodeEncodeError
# 讓整支 CLI 崩潰。實測驗證：不設 legacy_windows=False 之前，啟動後第一次
# console.input() 就會噴例外；設定後 + 把 stdout/stderr/stdin 轉成 UTF-8，同樣
# 的操作序列不再出錯。stdin 也要轉，不然透過管線/重導向輸入的中文（例如
# `/character 周柯宇`）會用 cp950 解碼成亂碼，實測確認過這個情境真的會發生。
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from chats import DEFAULT_OUT_DIR, is_code_request, smart_reply_traced
from function import read_as_chat_content

MODULE = "sinco 1.5"
CHARACTERS_DIR = TRANNING_DIR / "characters"
CHARACTER_CHAT_DIR = CHARACTERS_DIR / "character_chat_runs"
DEFAULT_PERSONA = "sinco"

console = Console(legacy_windows=False)


def _markdown_commands() -> list[str]:
    if not COMMAND_DIR.exists():
        return []
    return sorted(p.stem for p in COMMAND_DIR.glob("*.md"))


def _character_names() -> list[str]:
    """人物卡的 name 清單，來自 character_model.py --build 產生的 tranning/characters/*.json。"""
    if not CHARACTERS_DIR.exists():
        return []
    names = []
    for path in CHARACTERS_DIR.glob("*.json"):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if card.get("name"):
            names.append(card["name"])
    return sorted(names)


def print_banner():
    body = Text()
    body.append(MODULE, style="bold cyan")
    body.append("  自建 AI 助理 CLI\n", style="dim")
    body.append("沒有呼叫任何雲端 AI API，回覆全部來自本機訓練的模型。\n\n", style="dim italic")
    body.append("/help", style="bold")
    body.append(" 顯示指令   ", style="dim")
    body.append("/clear", style="bold")
    body.append(" 清除畫面   ", style="dim")
    body.append("exit", style="bold")
    body.append(" 或 Ctrl+C 離開", style="dim")
    console.print(Panel(body, border_style="cyan", expand=False, title="●", title_align="left"))


def print_help(state: dict):
    lines = Text()
    lines.append("內建指令\n", style="bold underline")
    lines.append("  /help", style="bold")
    lines.append("             顯示這個說明\n")
    lines.append("  /clear", style="bold")
    lines.append("            清除畫面\n")
    lines.append("  /open ", style="bold")
    lines.append("<路徑>      讀取檔案內容並送給模型\n")
    lines.append("  /character ", style="bold")
    lines.append("<名稱>   切換角色人格（不帶名稱＝查看目前人格與可選清單）\n")
    lines.append("  exit / quit", style="bold")
    lines.append("       離開\n")

    md_cmds = _markdown_commands()
    if md_cmds:
        lines.append("\n提示詞指令（app/command/*.md）\n", style="bold underline")
        for cmd in md_cmds:
            lines.append(f"  /{cmd}\n")

    lines.append(f"\n目前人格：{state['persona']}", style="dim")
    console.print(Panel(lines, border_style="cyan", expand=False))


def switch_character(name: str, state: dict):
    name = name.strip()
    if not name:
        names = _character_names()
        console.print(f"[dim]目前人格：{escape(state['persona'])}[/dim]")
        if names:
            choices = escape("、".join(names))
            console.print(f"[dim]可用角色：{choices}（用 /character <名稱> 切換，/character sinco 切回預設）[/dim]")
        return

    if name.lower() in {"sinco", "default", "reset"}:
        state["out_dir"] = DEFAULT_OUT_DIR
        state["persona"] = DEFAULT_PERSONA
        console.print(f"[dim]已切換回預設人格：{DEFAULT_PERSONA}[/dim]")
        return

    names = _character_names()
    matched = next((n for n in names if n.lower() == name.lower()), None) \
        or next((n for n in names if name.lower() in n.lower()), None)
    if matched is None:
        # name 是使用者輸入，直接塞進 markup 字串裡：實測驗證過如果裡面剛好含
        # 完整的 "[樣式]...[/樣式]" 片段，rich 真的會把它當成樣式標籤解析並
        # 套用（例如把一段文字變成別的顏色），不是假設性風險——escape() 讓它
        # 只被當成純文字顯示。
        console.print(f"[red]找不到角色「{escape(name)}」。[/red]")
        if names:
            console.print(f"[dim]可用角色：{escape('、'.join(names))}[/dim]")
        else:
            console.print("[dim]tranning/characters/ 底下還沒有任何角色卡（用 character_model.py --build 建立）。[/dim]")
        return

    # 目前只有一個共用的 character_chat_runs（見 CLAUDE.md to-do #14），還沒
    # 做到「一個角色各自獨立聊天 checkpoint」——如果該角色還沒訓練過，
    # smart_reply_traced() 本來就會回傳「尚未訓練」提示，不會在這裡假裝失敗。
    state["out_dir"] = CHARACTER_CHAT_DIR
    state["persona"] = matched
    console.print(f"[dim]已切換人格：{escape(matched)}[/dim]")


def ask_model(message: str, state: dict, check_code: bool = False):
    with console.status("[dim]思考中...[/dim]", spinner="dots"):
        try:
            trace, reply = smart_reply_traced(message, out_dir=state["out_dir"])
        except Exception as exc:  # 模型端任何未預期錯誤都要看得到，不要整支 CLI 崩潰
            console.print(f"[red]發生錯誤：{escape(str(exc))}[/red]")
            return

    trace_line = Text("· ", style="dim")
    trace_line.append(trace, style="dim")
    console.print(trace_line)

    console.print(Text(f"{state['persona']} ›", style="bold magenta"))
    if check_code and is_code_request(message):
        console.print(Panel(Syntax(reply, "python", theme="monokai", word_wrap=True),
                             border_style="grey50", expand=False))
    else:
        console.print(Text(reply))
    console.print()


def process_input(text: str, state: dict) -> bool:
    """處理一次輸入，回傳 False 代表要離開 REPL。"""
    if text.lower() in {"exit", "quit"}:
        console.print("[dim]再見！[/dim]")
        return False

    if not text.startswith("/"):
        ask_model(text, state, check_code=True)
        return True

    cmd, _, arg = text[1:].partition(" ")
    cmd, arg = cmd.strip().lower(), arg.strip()

    if cmd in {"exit", "quit"}:
        console.print("[dim]再見！[/dim]")
        return False
    if cmd == "help":
        print_help(state)
        return True
    if cmd == "clear":
        console.clear()
        print_banner()
        return True
    if cmd == "open":
        if not arg:
            console.print("[red]用法：/open <檔案路徑>[/red]")
            return True
        path = Path(arg).expanduser()
        if not path.is_file():
            console.print(f"[red]找不到檔案：{escape(str(path))}[/red]")
            return True
        console.print(f"[dim]--- {escape(path.name)} ---[/dim]")
        ask_model(read_as_chat_content(path), state)
        return True
    if cmd == "character":
        switch_character(arg, state)
        return True

    md_path = COMMAND_DIR / f"{cmd}.md"
    if not md_path.exists():
        console.print(f"[red]未知指令：/{escape(cmd)}[/red]（輸入 /help 查看可用指令）")
        return True
    ask_model(read_as_chat_content(md_path), state)
    return True


def main():
    parser = argparse.ArgumentParser(description=f"{MODULE} 終端機聊天介面（自建模型，無雲端 API）")
    parser.add_argument("--character", help="啟動時就切換到指定角色人格")
    args = parser.parse_args()

    state = {"out_dir": DEFAULT_OUT_DIR, "persona": DEFAULT_PERSONA}
    if args.character:
        switch_character(args.character, state)

    print_banner()

    while True:
        try:
            raw = console.input("[bold green]You[/bold green] [dim]›[/dim] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            console.print("[dim]再見！[/dim]")
            break

        text = raw.strip()
        if not text:
            continue

        try:
            if not process_input(text, state):
                break
        except KeyboardInterrupt:
            # 「思考中」狀態下按 Ctrl+C 只取消這一輪，不整個離開 REPL
            console.print("\n[dim]已取消[/dim]")


if __name__ == "__main__":
    main()
