"""/data 指令接線測試（to-do #53）：三個前端共用 data_tools.run_data；計算本身見 tranning/test_data_analysis.py。"""
import ast
from pathlib import Path

from lib.components.data_tools import DATA_ACTIONS, run_data


def test_usage_and_missing_path():
    assert run_data("").startswith("用法")
    assert "找不到" in run_data("analyze definitely_missing_file.csv")


def test_analyze_real_project_csv_returns_markdown(tmp_path, monkeypatch):
    run_data("")  # 先載入 data_analysis 模組
    import data_analysis
    monkeypatch.setattr(data_analysis, "DEFAULT_OUT", tmp_path)  # 報告寫到暫存，不動 repo
    text = run_data("analyze tranning/data.csv")
    assert (tmp_path / "tranning_data_analyze_report.md").exists()
    assert text.startswith("# 資料分析：data.csv") and "height" in text


def test_cli_suggests_data_actions():
    from lib.components import cli
    assert cli._arg_suggestions("data") == DATA_ACTIONS
    assert "data" in cli._slash_commands()


def test_gui_registers_data_command_and_arg_options():
    """GUI 需要 tkinter，這裡直接讀原始碼確認有登記，避免在無 Tk 環境無法驗證。"""
    tree = ast.parse(Path(__file__).with_name("command.py").read_text(encoding="utf-8"))
    values = {t.id: node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
              for t in node.targets if isinstance(t, ast.Name)}
    assert "data" in ast.literal_eval(values["BUILTIN_COMMANDS"])
    assert ast.literal_eval(values["ARG_SUGGESTIONS"].values[-1]) == DATA_ACTIONS
