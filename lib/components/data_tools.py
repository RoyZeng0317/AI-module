"""`/data analyze|split|kfold <路徑>`：資料分析與拆分（to-do #53）。

GUI（command.py）、CLI（cli.py）、網頁（web/backend/app.py）共用這一個入口，實際計算在
tranning/data_analysis.py（確定性統計，不交給模型生成數字）。讀取可指向專案外路徑，
輸出一律寫到專案內 output/data_analysis/（規則 01）。
"""
import sys
from pathlib import Path

_TRANNING = Path(__file__).resolve().parents[2] / "tranning"
DATA_ACTIONS = ["analyze", "split", "kfold"]


def run_data(arg: str) -> str:
    if str(_TRANNING) not in sys.path:
        sys.path.append(str(_TRANNING))
    try:
        from data_analysis import run_command
    except Exception as exc:  # 缺套件等，回報而不是讓 UI 崩潰
        return f"資料分析模組載入失敗：{type(exc).__name__}: {exc}"
    return run_command(arg)
