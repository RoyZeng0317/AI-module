"""terminal_exec.py — 白名單制終端機指令執行，給 tools.route_reply() 用。

跟 calculus_generator.py/calculus_solver.py 一樣是獨立子模組：sinco 自己的
邏輯（detect_command()）決定「使用者是不是在要求執行某個指令」以及「是哪一
個」，模型本身完全不參與生成要執行的指令字串——這是 Rule 06「全部自建、不
呼叫外部 AI」的同一個邊界（這裡連本機模型都沒用到，純規則比對 + 固定指令表）。

WHITELISTED_COMMANDS 裡每一項的 argv 都是寫死的常數，使用者輸入只能用來
『挑選』表裡的哪個 key，絕不會被拼進實際執行的指令——這是採「白名單制、不
逐次要求人工確認」還能算安全的原因。表裡全部是唯讀／非破壞性指令（不含
git add|commit|push、pip install、del、mkdir 等會改動狀態的指令）；要加會
修改狀態的指令是另一個要重新討論安全機制（例如確認流程）的決定，不能直接
塞進這個白名單。
"""

import re
import subprocess
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller onefile：見 lib/components/cli.py 對 __file__ 在 frozen 模式
    # 下不可靠的說明。
    _PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent

WHITELISTED_COMMANDS: dict[str, tuple[list[str], int]] = {
    "git status": (["git", "status"], 10),
    "git log": (["git", "log", "-10", "--oneline"], 10),
    "git branch": (["git", "branch"], 10),
    "git diff": (["git", "diff", "--stat"], 10),
    "git remote": (["git", "remote", "-v"], 10),
    "python --version": (["python", "--version"], 10),
    "pip list": (["pip", "list"], 15),
    "dir": (["cmd", "/c", "dir"], 10),
    "whoami": (["whoami"], 10),
    "pytest --collect-only": (["pytest", "--collect-only", "tranning/"], 45),
}

_COMMAND_ALIASES: dict[str, str] = {
    "git status": "git status", "git 狀態": "git status", "git 現況": "git status",
    "git log": "git log", "git 紀錄": "git log", "git 日誌": "git log", "git 歷史": "git log",
    "git branch": "git branch", "git 分支": "git branch",
    "git diff": "git diff", "git 差異": "git diff", "git 修改內容": "git diff",
    "git remote": "git remote", "git 遠端": "git remote",
    "python --version": "python --version", "python 版本": "python --version",
    "pip list": "pip list", "已安裝套件": "pip list", "套件清單": "pip list",
    "dir": "dir", "列出檔案": "dir", "目前目錄檔案": "dir",
    "whoami": "whoami", "目前使用者": "whoami",
    "pytest --collect-only": "pytest --collect-only",
    "列出測試": "pytest --collect-only", "測試清單": "pytest --collect-only",
}

_EXEC_REQUEST_HINTS = ("執行", "跑一下", "跑", "下指令", "指令", "run", "execute")


def detect_command(message: str) -> str | None:
    """判斷 message 是否在要求執行某個白名單指令，回傳
    WHITELISTED_COMMANDS 的 key；不是就回傳 None。

    完全等於某個別名（例如訊息就是「git status」）直接算命中；否則要同時
    符合「含執行意圖詞（執行／跑／指令／run…）」且「含某個別名的完整詞」
    （用 \\b 詞界比對，不是裸字串 in，避免像 "dir" 命中英文單字內部）才算數。
    """
    text = message.strip().strip("。！？!?")
    lower = text.lower()

    if lower in _COMMAND_ALIASES:
        return _COMMAND_ALIASES[lower]

    if not any(hint in lower for hint in _EXEC_REQUEST_HINTS):
        return None

    for alias in sorted(_COMMAND_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias.lower())}\b", lower):
            return _COMMAND_ALIASES[alias]
    return None


def looks_like_execution_request(message: str) -> bool:
    """True 表示訊息讀起來像是某種執行指令的請求，不管有沒有命中白名單——
    讓 route_reply() 能區分「根本不是指令請求」（回 None，正常退回聊天模型）
    跟「有這個意圖但被拒絕」（不能安靜地退回，否則聊天模型會亂answer一個假
    結果）。
    """
    return any(hint in message.strip().lower() for hint in _EXEC_REQUEST_HINTS)


def _truncate(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...(已截斷)"


def run_command(key: str) -> tuple[int | None, str, str]:
    """執行 WHITELISTED_COMMANDS[key]，回傳 (returncode, stdout, stderr)。

    key 必須是呼叫端先用 detect_command() 驗證過的合法 key（不合法的 key
    是呼叫端的邏輯錯誤，讓它 KeyError，不吞掉）。returncode 只有在 Python
    端本身失敗（逾時／OSError）時才會是 None，說明會放在 stderr——這個函式
    本身不丟例外給呼叫端。
    """
    argv, timeout = WHITELISTED_COMMANDS[key]
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
            cwd=_PROJECT_ROOT, encoding="utf-8", errors="replace",
        )
        return result.returncode, _truncate(result.stdout), _truncate(result.stderr)
    except subprocess.TimeoutExpired:
        return None, "", f"指令執行超過 {timeout} 秒逾時限制。"
    except OSError as exc:
        return None, "", f"指令執行失敗：{exc}"
