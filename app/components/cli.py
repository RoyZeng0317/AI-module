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

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from chats import DEFAULT_OUT_DIR, is_code_request, smart_reply_traced, tokenize
from function import read_as_chat_content
from memory_store import CATEGORIES as MEMORY_CATEGORIES
from memory_store import add_memory, delete_memory, format_memories, list_memories

MODULE = "Nova 1.5"  # 應用程式/品牌名稱；聊天時的助理人格名稱仍是 "sinco"（DEFAULT_PERSONA，訓練資料/回覆內容都沒有改）
CHARACTERS_DIR = TRANNING_DIR / "characters"
CHARACTER_CHAT_DIR = CHARACTERS_DIR / "character_chat_runs"
DEFAULT_PERSONA = "sinco"
HISTORY_PATH = APP_COMPONENTS_DIR / ".cli_history"  # 方向鍵輸入紀錄持久化檔，跨次執行 CLI 仍可叫出之前打過的內容

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


_BUILTIN_SLASH_COMMANDS = {"help", "clear", "open", "character", "memory"}


def _slash_commands() -> list[str]:
    return sorted(_BUILTIN_SLASH_COMMANDS | set(_markdown_commands()))


def _arg_suggestions(name: str) -> list[str] | None:
    """指令名稱後面帶引數時可以建議的候選值——目前只有 /character（列出
    tranning/characters/ 底下已建立的人物卡），跟 GUI 版 command.py 的
    ARG_SUGGESTIONS 是同一個構想，只是 CLI 跟 GUI 的內建指令集不同。
    """
    if name == "character":
        return _character_names()
    if name == "memory":
        return ["list", "add", "del"]
    return None


class SlashCommandCompleter(Completer):
    """斜線指令 Tab 自動完成，跟 GUI（app/components/command.py 的
    CommandPalette）同一套邏輯：輸入「/」後建議指令名稱；輸入「/指令 」
    （帶空格）之後，如果該指令在 _arg_suggestions() 有登記引數選項，改成
    建議引數值。prompt_toolkit 內建 Tab 循環／方向鍵選單，不用像 Tk 版
    另外刻 Listbox。
    """

    def get_completions(self, document, _complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        if " " in text:
            name, _, partial_arg = text[1:].partition(" ")
            options = _arg_suggestions(name.strip().lower())
            if not options:
                return
            for option in options:
                if option.lower().startswith(partial_arg.lower()):
                    yield Completion(option, start_position=-len(partial_arg))
            return

        query = text[1:].lower()
        for cmd in _slash_commands():
            if cmd.lower().startswith(query):
                yield Completion(cmd, start_position=-len(query))


def print_banner():
    body = Text()
    body.append(MODULE, style="bold cyan")
    body.append("  自建 AI 助理 CLI\n", style="dim")
    body.append("助理人格：sinco ｜ 沒有呼叫任何雲端 AI API，回覆全部來自本機訓練的模型。\n\n", style="dim italic")
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
    lines.append("  /memory", style="bold")
    lines.append("            編輯持久記憶（list/add/del，輸入 /memory 查看完整用法）\n")
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


# /memory [list [分類] | add <分類> <內容> | del <id>]：從 GUI（command.py 的
# CommandPalette._run_memory）移植過來的同一套邏輯，共用同一個 memory_store.py
# 儲存檔——這是使用者自己選分類寫入、可查詢/刪除的持久資料，跟 /open、
# app/command/*.md（讀出來整份塞給模型當提示詞）是不同機制，不會被拿去訓練
# 或影響 chats.py 的 checkpoint。
def run_memory(arg: str):
    sub, _, rest = arg.partition(" ")
    sub, rest = sub.strip().lower(), rest.strip()

    if sub in ("", "list"):
        category = rest.lower() or None
        if category and category not in MEMORY_CATEGORIES:
            console.print(f"[red]未知分類：{escape(category)}（可用：{escape(', '.join(MEMORY_CATEGORIES))}）[/red]")
            return
        console.print(format_memories(list_memories(category)))
        return

    if sub == "add":
        category, _, text = rest.partition(" ")
        try:
            entry = add_memory(category, text)
        except ValueError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            return
        console.print(f"[dim]已新增記憶 [{entry['id']}]：{escape(entry['text'])}[/dim]")
        return

    if sub in ("del", "delete"):
        if delete_memory(rest):
            console.print(f"[dim]已刪除記憶 [{escape(rest)}][/dim]")
        else:
            console.print(f"[red]找不到記憶 id：{escape(rest)}[/red]")
        return

    console.print(
        "[dim]用法：\n"
        "  /memory                    列出全部記憶\n"
        "  /memory list <分類>        列出指定分類\n"
        f"  /memory add <分類> <內容>  新增（分類：{escape(', '.join(MEMORY_CATEGORIES))}）\n"
        "  /memory del <id>           刪除[/dim]"
    )


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

    # sinco 的 tokenizer 是字元級（見 chats.py tokenize()），沒有 BPE/子詞單位，
    # 這裡的「token」就是字元數，跟 Claude 那種子詞 token 不是同一種算法，只是
    # 借用同樣的「輸入+輸出用量」呈現方式，讓你知道這一輪送進/吐出模型的量體。
    in_tokens, out_tokens = len(tokenize(message)), len(tokenize(reply))
    token_line = Text(
        f"· 約 {in_tokens + out_tokens} 個 token（輸入 {in_tokens} + 輸出 {out_tokens}，以字元數估算）",
        style="dim",
    )
    console.print(token_line)

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
    if cmd == "memory":
        run_memory(arg)
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

    # 跟 GUI（app/components/command.py 的 CommandPalette）一樣可以 Tab 快速
    # 鍵入完整斜線指令——console.input()（rich）本身沒有行編輯/自動完成能力，
    # 換成 prompt_toolkit 的 PromptSession 才有 Tab/方向鍵選單，rich 的
    # console 繼續負責其餘所有輸出（兩者只是先後寫 stdout，不會互相干擾）。
    # history=FileHistory(...) 讓方向鍵叫出的輸入紀錄寫進 HISTORY_PATH，跨次
    # 執行 CLI 依然能用上鍵叫出之前打過的內容（不用額外寫程式碼，這是
    # prompt_toolkit 內建行為，只是預設的 InMemoryHistory 只在單次執行期間有效）。
    # prompt_toolkit 在 Windows 上需要真正的主控台螢幕緩衝區（實測：透過管線
    # 重導向輸出，或某些自動化/非互動執行環境會直接缺少這個緩衝區）才能運作，
    # 缺少時建立 PromptSession 會直接丟 NoConsoleScreenBufferError——這裡退回
    # 原本 console.input() 的寫法，沒有 Tab 自動完成但至少不會讓整支 CLI 在
    # 這類環境下直接開不起來（例如你之後想用 `python cli.py < 腳本.txt` 這種
    # 方式做非互動測試）。
    try:
        session: PromptSession | None = PromptSession(
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
            history=FileHistory(str(HISTORY_PATH)),
        )
    except Exception:
        session = None

    while True:
        try:
            if session is not None:
                raw = session.prompt(HTML("<ansigreen><b>You</b></ansigreen> <ansibrightblack>›</ansibrightblack> "))
            else:
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
