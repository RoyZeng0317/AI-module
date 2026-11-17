"""data_analysis.py 測試（to-do #53）：用小型合成資料驗證統計、拆分、k-fold、洩漏與指令解析。"""
import csv
import json
from pathlib import Path

import pytest

import data_analysis as da


def _csv(path: Path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture
def project(tmp_path, monkeypatch):
    """把 ROOT/輸出位置指到 tmp，確認輸出不會寫到真專案。"""
    monkeypatch.setattr(da, "ROOT", tmp_path)
    monkeypatch.setattr(da, "DEFAULT_OUT", tmp_path / "output" / "data_analysis")
    return tmp_path


def test_load_formats(tmp_path):
    rows, cols = da.load_table(_csv(tmp_path / "a.csv", ["x", "y"], [[1, "a"], [2, "b"]]))
    assert cols == ["x", "y"] and len(rows) == 2
    (tmp_path / "b.tsv").write_text("x\ty\n1\ta\n", encoding="utf-8")
    assert da.load_table(tmp_path / "b.tsv")[1] == ["x", "y"]
    (tmp_path / "c.jsonl").write_text('{"p":"hi","r":"yo"}\n{"p":"a","r":"b","extra":1}\n', encoding="utf-8")
    assert da.load_table(tmp_path / "c.jsonl")[1] == ["p", "r", "extra"]
    (tmp_path / "d.json").write_text(json.dumps({"data": [{"a": 1}]}), encoding="utf-8")
    assert da.load_table(tmp_path / "d.json")[0] == [{"a": 1}]
    (tmp_path / "e.txt").write_text("一\n\n二\n", encoding="utf-8")
    assert da.load_table(tmp_path / "e.txt")[0] == [{"text": "一"}, {"text": "二"}]
    (tmp_path / "f.csv").write_bytes("名稱,分數\n蘋果,3\n".encode("cp950"))
    assert da.load_table(tmp_path / "f.csv")[0] == [{"名稱": "蘋果", "分數": "3"}]
    with pytest.raises(ValueError, match="不支援"):
        da.load_table(tmp_path / "g.parquet")


def test_types_and_numeric_stats():
    assert da.infer_type(["1", "2.5", "", "NaN"]) == "numeric"
    assert da.infer_type(["a", "b", "a"] * 10) == "categorical"
    assert da.infer_type([[1], [2]]) == "nested"
    assert da.infer_type(["", None]) == "empty"
    s = da.numeric_stats([1, 2, 3, 4, 100])
    assert s["median"] == 3 and s["min"] == 1 and s["max"] == 100 and s["iqr_outliers"] == 1
    assert da.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert da.pearson([1, 1, 1], [1, 2, 3]) is None


def test_analyze_detects_problems():
    rows = [{"id": str(i), "feat": "x" if i % 2 else "y", "label": "a" if i < 27 else "b", "const": "1",
             "sparse": "" if i % 3 else "v"} for i in range(30)]
    rows.append(dict(rows[0]))                       # 重複列
    rows.append(dict(rows[1], label="b"))            # 特徵相同、標籤不同
    result = da.analyze_table(rows, list(rows[0]))
    assert result["target"]["column"] == "label" and result["target"]["auto_detected"]
    assert result["duplicate_rows"] == 1
    assert result["target"]["label_conflicts"] >= 1
    text = "\n".join(result["warnings"])
    for word in ("重複", "不平衡", "只有一種值", "缺值", "標註衝突"):
        assert word in text, word
    assert "以上數字由程式直接計算" in da.format_analysis(dict(result, source="x.csv"))


def test_analyze_unknown_target_raises():
    with pytest.raises(ValueError, match="找不到目標欄位"):
        da.analyze_table([{"a": 1}], ["a"], target="zzz")


def test_split_is_stratified_reproducible_and_complete():
    rows = [{"text": f"t{i}", "label": "a" if i < 80 else "b"} for i in range(100)]
    s1 = da.split_rows(rows, (0.7, 0.15, 0.15), 1, target="label")
    assert s1 == da.split_rows(rows, (0.7, 0.15, 0.15), 1, target="label")
    assert sum(len(v) for v in s1.values()) == 100
    for name in ("val", "test"):
        labels = [r["label"] for r in s1[name]]
        assert labels.count("b") >= 2 and labels.count("a") >= 10
    assert da.check_table_leakage(s1, "label")["leak_count"] == 0


def test_split_keeps_duplicates_and_groups_together():
    rows = [{"text": f"t{i % 10}", "label": "a"} for i in range(40)]  # 每種 text 重複 4 次
    splits = da.split_rows(rows, (0.6, 0.2, 0.2), 0, target="label")
    assert da.check_table_leakage(splits, "label")["leak_count"] == 0
    grouped = [{"user": f"u{i // 5}", "v": i, "label": "a"} for i in range(50)]
    splits = da.split_rows(grouped, (0.6, 0.2, 0.2), 0, target="label", group="user")
    owners = {}
    for name, part in splits.items():
        for r in part:
            assert owners.setdefault(r["user"], name) == name
    with pytest.raises(ValueError, match="ratios"):
        da.split_rows(grouped, (0.5, 0.6))


def test_split_and_kfold_write_outputs_inside_project(project):
    src = _csv(project / "d.csv", ["text", "label"], [[f"t{i}", "a" if i % 3 else "b"] for i in range(60)])
    result = da.split_path(src, target="label")
    out = project / result["out_dir"]
    assert out.is_relative_to(project) and {p.name for p in out.iterdir()} == {"train.csv", "val.csv", "test.csv", "split_manifest.json"}
    assert sum(result["counts"].values()) == 60 and result["leakage"]["leak_count"] == 0
    kf = da.kfold_path(src, k=3, target="label")
    assert sum(f["n"] for f in kf["folds"]) == 60
    with pytest.raises(ValueError, match="專案資料夾內"):
        da.split_path(src, out_dir=project.parent / "outside")
    with pytest.raises(ValueError, match="非空"):
        da.split_path(src, out_dir=out)


def test_image_folder_analyze_and_split(project):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    for cls in ("cat", "dog"):
        (project / "imgs" / cls).mkdir(parents=True)
        for i in range(10):
            arr = np.random.default_rng(hash((cls, i)) % 2**32).integers(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(project / "imgs" / cls / f"{i}.png")
    info = da.analyze_path(project / "imgs")
    assert info["layout"] == "class_folders" and info["total_images"] == 20
    result = da.split_path(project / "imgs")
    assert sum(result["counts"].values()) == 20 and result["leakage"]["leak_count"] == 0
    assert da.analyze_path(project / result["out_dir"])["layout"] == "split"


def test_run_command_and_errors(project):
    _csv(project / "m.csv", ["text", "score"], [[f"t{i}", i % 3] for i in range(30)])
    text = da.run_command("analyze m.csv --target score")
    assert text.startswith("# 資料分析") and "報告已存" in text
    assert da.run_command("") == da.COMMAND_USAGE
    assert "找不到" in da.run_command("analyze nope.csv", save_report=False)
    assert "指令格式錯誤" in da.run_command("analyze m.csv --bogus", save_report=False)
    assert da._tokens(r'split "C:\Users\me\my data.csv" --k 3') == ["split", r"C:\Users\me\my data.csv", "--k", "3"]


def test_route_only_triggers_on_intent_and_real_path(project):
    _csv(project / "m.csv", ["text", "score"], [[f"t{i}", i % 3] for i in range(30)])
    assert da.route_data_request("我今天心情不好") is None
    assert da.route_data_request("幫我分析人生") is None
    assert da.route_data_request("分析 missing.csv") is None
    trace, reply = da.route_data_request("幫我分析 m.csv")
    assert "data_analysis.py" in trace and reply.startswith("# 資料分析")
    trace, reply = da.route_data_request("把 m.csv 依 score 分層拆分")
    assert reply.startswith("# 資料拆分") and "分層：score" in reply
    _, reply = da.route_data_request("m.csv 做 3-fold 交叉驗證")
    assert "k = 3" in reply
