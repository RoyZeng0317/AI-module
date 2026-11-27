from yolo_train_utils import summarize_overfitting


def test_missing_file_returns_none(tmp_path):
    assert summarize_overfitting(tmp_path / "no_results.csv") is None


def test_overfitting_warning(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,train/box_loss,val/box_loss,metrics/mAP50-95(B)\n"
        "1,1.0,2.0,0.5\n"
    )
    summary = summarize_overfitting(csv_path)
    assert "train_loss=1.0000" in summary and "val_loss=2.0000" in summary
    assert "overfitting" in summary


def test_underfitting_warning(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,train/box_loss,val/box_loss,metrics/mAP50-95(B)\n"
        "1,1.0,1.0,0.02\n"
    )
    assert "underfitting" in summarize_overfitting(csv_path)


def test_no_warning_when_healthy(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,train/box_loss,val/box_loss,metrics/mAP50-95(B)\n"
        "1,1.0,1.1,0.5\n"
    )
    summary = summarize_overfitting(csv_path)
    assert "warning" not in summary
