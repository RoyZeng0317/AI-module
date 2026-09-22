import cv2, yolo_custom as yc


def test_make_synthetic_labels_valid(tmp_path):
    yaml = yc.make_synthetic(tmp_path, n=8)
    assert yaml.exists() and len(list((tmp_path / "images" / "train").glob("*.jpg"))) == 8
    for f in (tmp_path / "labels" / "train").glob("*.txt"):
        for line in f.read_text().splitlines():
            c, *box = line.split()
            assert int(c) < len(yc.CLASSES) and all(0 <= float(v) <= 1 for v in box)


def test_apply_rules_ok_missing_and_unsure():
    d = lambda l, c: {"label": l, "confidence": c, "box": [0, 0, 1, 1]}
    assert yc.apply_rules([d("circle", .9), d("circle", .8)], {"circle": 2})["ok"]
    r = yc.apply_rules([d("circle", .9), d("square", .3)], {"circle": 2})
    assert not r["ok"] and "缺少 1" in r["issues"][0] and len(r["unsure"]) == 1  # 低信心不計數


def test_train_and_inspect_smoke(tmp_path, monkeypatch):
    """1 epoch 冒煙測試：只驗管線跑得完，不驗準確度。"""
    monkeypatch.setattr(yc, "RUNS", tmp_path / "runs"); monkeypatch.setattr(yc, "BEST", tmp_path / "runs" / "custom" / "weights" / "best.pt")
    best = yc.train(yc.make_synthetic(tmp_path / "d", n=16), epochs=1)
    assert best.exists()
    img = tmp_path / "d" / "images" / "val" / "0.jpg"
    assert "判定：" in yc.inspect(img, {"circle": 1}, weights=str(best))


def test_parse_rules_json_and_kv():
    assert yc.parse_rules('{"circle": 2}') == yc.parse_rules("circle=2") == {"circle": 2}
    assert yc.parse_rules("circle=2, square=1") == {"circle": 2, "square": 1}


def test_status_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(yc, "RUNS", tmp_path)
    assert "尚無訓練紀錄" in yc.status()
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "results.csv").write_text("   epoch, metrics/mAP50(B), metrics/mAP50-95(B)\n   3, 0.5, 0.4\n")
    (tmp_path / "custom" / "args.yaml").write_text("epochs: 30\n")
    assert yc.status() == "epoch 3/30  mAP50=0.500  mAP50-95=0.400"
