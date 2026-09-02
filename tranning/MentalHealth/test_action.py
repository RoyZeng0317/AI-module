"""Pipeline smoke test for action.py(情緒傾向分數模型，0~5 有界迴歸)。

不宣稱真實準確度——用合成的正向／負向短句只證明訓練迴圈(vocab 建立、
SmoothL1 迴歸 loss、early stopping、checkpoint 輸出、推論測試集報告)與
predict_emotion() 推論路徑能跑完整個流程不崩潰。
"""

import json

from web.admin.frontend.src.components.action import SCORE_MAX, SCORE_MIN, predict_emotion, train


def _make_synthetic_rows():
    samples = {
        1: ["壓力好大喘不過氣", "覺得很挫折想哭", "最近一直失眠", "什麼事都做不好"],
        2: ["有點焦慮不知道怎麼辦", "心情悶悶的", "腦袋停不下來胡思亂想", "覺得有點累"],
        4: ["今天心情很好", "跟朋友出去玩很開心", "睡了一個好覺", "順利完成了小任務"],
        5: ["順利完成了專案，超有成就感", "今天過得非常美好", "吃到期待已久的美食好幸福", "解決了困擾很久的問題"],
    }
    texts: list[str] = []
    labels: list[int] = []
    for score, rows in samples.items():
        texts += rows * 5
        labels += [score] * (len(rows) * 5)
    return texts, labels


def test_training_and_predict_run_end_to_end(tmp_path, monkeypatch):
    texts, labels = _make_synthetic_rows()
    csv_path = tmp_path / "mental.csv"
    csv_path.write_text(
        "text,score\n" + "\n".join(f"{t},{s}" for t, s in zip(texts, labels)),
        encoding="utf-8",
    )

    out_dir = tmp_path / "runs"
    model, vocab, history = train(
        csv_path=csv_path, out_dir=out_dir, epochs=3, batch_size=4,
        embed_size=16, hidden_size=32, lr=1e-2, max_len=20, dropout=0.1,
        weight_decay=1e-4, val_split=0.2, test_split=0.2, patience=5,
    )

    assert len(history) == 3
    assert (out_dir / "best_model.pt").exists()
    assert json.loads((out_dir / "vocab.json").read_text(encoding="utf-8")) == vocab
    assert all(h["val_loss"] >= 0 for h in history)

    # 推論測試集報告：訓練/驗證都沒看過的資料，檢查模型是不是在推測而非死記硬背
    report = json.loads((out_dir / "generalization_report.json").read_text(encoding="utf-8"))
    assert len(report) > 0
    for row in report:
        assert SCORE_MIN <= row["predicted_score"] <= SCORE_MAX
        assert SCORE_MIN <= row["predicted_label"] <= SCORE_MAX

    result = predict_emotion("今天心情很糟糕", out_dir=out_dir)
    assert result["status"] is None
    assert SCORE_MIN <= result["label"] <= SCORE_MAX
    assert SCORE_MIN <= result["score"] <= SCORE_MAX


def test_predict_without_checkpoint_returns_placeholder(tmp_path):
    result = predict_emotion("隨便一句話", out_dir=tmp_path / "no_checkpoint_here")
    assert result["status"] is not None
    assert "尚未訓練" in result["status"]
    assert result["label"] is None
    assert result["score"] is None
