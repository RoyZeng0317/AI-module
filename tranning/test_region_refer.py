"""region_refer.py 管線煙霧測試 — 合成資料不代表真實語意理解能力，只驗證：
(1) 合成資料產生器格式正確、(2) 對比學習訓練迴圈跑得完、(3) refer() 的
「候選框 -> 裁圖 -> 區域 embedding -> 跟文字查詢比對」接線正確（YOLO 偵測部分
用 monkeypatch 假框，不需要真的訓練 YOLO，跟 materials_recongizion/
test_detect_and_read.py 同一套做法）。
"""
import json

import numpy as np
import pytest

import embedding_model
import region_refer as rr  # noqa: F401  (匯入時會把 web/backend 插進 sys.path，detector 才能被找到)
import detector


def _train_tiny_text_encoder(out_dir):
    sentences = ["左邊的紅色圓形", "右邊的綠色方形", "中間的藍色三角形", "下面的黃色圓形",
                 "上面左邊的紅色方形", "上面右邊的綠色三角形", "左邊的黃色三角形", "右邊的藍色圓形"]
    embedding_model.train(sentences=sentences, out_dir=out_dir, epochs=1, batch_size=2,
                           max_len=16, n_layer=1, n_embd=32, n_head=2, vocab_size=300)


def test_make_synthetic_pairs_valid(tmp_path):
    pairs_path = rr.make_synthetic(tmp_path / "data", n=6)
    lines = pairs_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 12  # 每張圖 2~3 個物件 * 6 張
    for line in lines:
        pair = json.loads(line)
        assert (tmp_path / "data" / pair["crop"]).exists()
        assert pair["caption"]


def test_train_runs_end_to_end(tmp_path):
    embed_dir = tmp_path / "embed"
    _train_tiny_text_encoder(embed_dir)
    data_dir = rr.make_synthetic(tmp_path / "data", n=12).parent

    model, history = rr.train(data_dir, out_dir=tmp_path / "region_runs", epochs=1,
                               batch_size=4, patience=1, text_encoder_dir=embed_dir, device="cpu")

    assert (tmp_path / "region_runs" / "region_model.pt").exists()
    assert (tmp_path / "region_runs" / "config.json").exists()
    assert len(history) == 1


def test_train_requires_text_encoder(tmp_path):
    data_dir = rr.make_synthetic(tmp_path / "data", n=4).parent
    with pytest.raises(RuntimeError):
        rr.train(data_dir, out_dir=tmp_path / "region_runs", text_encoder_dir=tmp_path / "no_such_embed")


def test_refer_picks_a_box_without_crashing(tmp_path, monkeypatch):
    embed_dir = tmp_path / "embed"
    _train_tiny_text_encoder(embed_dir)
    data_dir = rr.make_synthetic(tmp_path / "data", n=12).parent
    rr.train(data_dir, out_dir=tmp_path / "region_runs", epochs=1, batch_size=4,
              patience=1, text_encoder_dir=embed_dir, device="cpu")

    image = np.full((64, 64, 3), 30, np.uint8)
    boxes = [[0, 0, 32, 32], [32, 32, 64, 64]]
    monkeypatch.setattr(detector, "detect", lambda image, conf, weights: [
        {"label": "shape", "confidence": 0.9, "box": b} for b in boxes
    ])

    result = rr.refer(image, "紅色", "dummy.pt", region_runs=tmp_path / "region_runs", text_encoder_dir=embed_dir)
    assert result is not None
    assert result.box in boxes
    assert 0.0 <= result.match_score <= 1.0


def test_refer_returns_none_when_no_detections(monkeypatch):
    image = np.zeros((64, 64, 3), np.uint8)
    monkeypatch.setattr(detector, "detect", lambda image, conf, weights: [])
    assert rr.refer(image, "紅色", "dummy.pt") is None


def test_refer_returns_none_when_not_trained(tmp_path, monkeypatch):
    image = np.zeros((64, 64, 3), np.uint8)
    monkeypatch.setattr(detector, "detect", lambda image, conf, weights: [
        {"label": "shape", "confidence": 0.9, "box": [0, 0, 32, 32]},
    ])
    assert rr.refer(image, "紅色", "dummy.pt", region_runs=tmp_path / "no_such_runs") is None
