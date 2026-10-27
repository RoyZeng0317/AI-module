import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tranning")]
from object_labeling import LabelDataset, fit_geometry, project_path, read_labels, rectangle, yolo_line
from object_detection_train import train, validate_data
from lib.yolo.object_labeler import CameraWorker, Labeler


def dataset(tmp_path):
    data = LabelDataset(tmp_path / "資料")
    keys = []
    for color, split in (("red", "train"), ("blue", "val")):
        key = data.add_image(Image.new("RGB", (200, 100), color))
        data.save(key, [[0, 10, 20, 80, 90]], split)
        keys.append(key)
    return data, keys


def test_roundtrip_reverse_clip_and_geometry():
    box = [0, *rectangle([210, 100, -20, 10], 200, 100)]
    assert read_labels(yolo_line(box, 200, 100, 1), 200, 100, 1)[0] == pytest.approx(box)
    scale, ox, oy = fit_geometry(200, 100, 300, 300)
    assert (scale, ox, oy) == (1.5, 0, 75)
    assert ((150-ox)/scale, (150-oy)/scale) == (100, 50)


def test_delete_unmarked_preserves_labels_negatives_and_exports(tmp_path):
    data, keys = dataset(tmp_path)
    exported = data.export()
    pending = data.add_image(Image.new("RGB", (40, 30), "green"))
    source = data.image_path(pending)
    negative = data.add_image(Image.new("RGB", (40, 30), "white"))
    data.save(negative, [], negative=True)
    assert data.delete_unmarked() == 1
    loaded = LabelDataset(data.folder)
    assert pending not in loaded.data["images"] and not source.exists()
    assert set(loaded.data["images"]) == {*keys, negative}
    assert list((data.folder / "trash").glob(f"*/{pending}.png"))
    assert exported.exists() and validate_data(exported) == exported
    assert data.delete_unmarked() == 0


def test_delete_class_remaps_and_requires_review_without_changing_snapshot(tmp_path):
    data, keys = dataset(tmp_path)
    data.add_class("錯字"); data.add_class("直尺")
    data.save(keys[0], [[0, 1, 1, 10, 10], [1, 20, 20, 40, 40], [2, 50, 20, 80, 80]])
    data.save(keys[1], [[2, 10, 20, 80, 90]], "val")
    exported = data.export()
    yaml_before = exported.read_bytes()
    assert data.delete_class("錯字") == 1
    loaded = LabelDataset(data.folder)
    assert loaded.classes == ["橡皮擦", "直尺"]
    assert [b[0] for b in loaded.data["images"][keys[0]]["boxes"]] == [0, 1]
    assert not loaded.data["images"][keys[0]]["reviewed"]
    assert loaded.data["images"][keys[1]]["reviewed"]
    assert loaded.data["images"][keys[1]]["boxes"][0][0] == 1
    assert exported.read_bytes() == yaml_before
    assert not data.unmarked_keys()  # 有保留框的照片不因重新確認而被當成空白刪掉


def test_rename_class_preserves_boxes_and_rejects_invalid_name(tmp_path):
    data, keys = dataset(tmp_path)
    data.add_class("直尺")
    before = [list(b) for b in data.data["images"][keys[0]]["boxes"]]
    data.rename_class("橡皮擦", "橡皮")
    assert LabelDataset(data.folder).classes == ["橡皮", "直尺"]
    assert data.data["images"][keys[0]]["boxes"] == before
    assert data.data["images"][keys[0]]["reviewed"]
    for name in (" ", "直尺"):
        with pytest.raises(ValueError):
            data.rename_class("橡皮", name)
    data.delete_class("直尺")
    with pytest.raises(ValueError):
        data.delete_class("橡皮")


def test_delete_rollback_when_save_fails(tmp_path, monkeypatch):
    data = LabelDataset(tmp_path / "rollback")
    key = data.add_image(Image.new("RGB", (20, 20), "red"))
    path = data.image_path(key)
    def fail():
        raise OSError("模擬寫入失敗")
    monkeypatch.setattr(data, "persist", fail)
    with pytest.raises(OSError):
        data.delete_unmarked()
    assert path.exists() and key in data.data["images"]
    assert key in LabelDataset(data.folder).data["images"]


@pytest.mark.parametrize("points", [[0, 0, 0, 2], [float("nan"), 0, 2, 2], [-10, 0, -2, 3]])
def test_invalid_rectangle(points):
    with pytest.raises(ValueError):
        rectangle(points, 20, 20)


@pytest.mark.parametrize("line", ["0 0.5 0.5 0 0.3", "0 nan 0.5 0.1 0.1", "1 0.5 0.5 0.2 0.2", "0 0.1 0.5 0.5 0.3", "0 1 2"])
def test_invalid_labels(line):
    with pytest.raises(ValueError):
        read_labels(line, 100, 100, 1)


def test_classes_reload_duplicate_and_negative(tmp_path):
    data, keys = dataset(tmp_path)
    assert data.add_class("鉛筆") == 1
    assert data.add_class("橡皮擦") == 0
    duplicate = data.add_image(Image.new("RGB", (200, 100), "red"))
    assert duplicate == keys[0] and len(data.data["images"]) == 2
    pending = data.add_image(Image.new("RGB", (50, 100), "white"))
    with pytest.raises(ValueError):
        data.save(pending, [])
    data.save(pending, [], negative=True)
    loaded = LabelDataset(data.folder)
    assert loaded.classes == ["橡皮擦", "鉛筆"]
    assert loaded.data["images"][pending]["negative"]
    assert loaded.data["images"][keys[0]]["boxes"][0] == pytest.approx([0, 10, 20, 80, 90])
    with pytest.raises(ValueError):
        data.add_class(" ")


def test_same_filename_different_images(tmp_path):
    data = LabelDataset(tmp_path / "資料")
    keys = []
    for name, color in (("a", "red"), ("b", "blue")):
        path = tmp_path / name / "照片.png"
        path.parent.mkdir()
        Image.new("RGB", (20, 20), color).save(path)
        keys.append(data.import_image(path))
    assert keys[0] != keys[1]
    assert all(data.image_path(key).is_file() for key in keys)


def test_export_ignores_pending_and_isolates_snapshots(tmp_path):
    data, _ = dataset(tmp_path)
    data.add_image(Image.new("RGB", (20, 20), "green"))
    path, second = data.export(), data.export()
    assert path != second
    assert validate_data(path) == path
    assert len(list(path.parent.glob("images/*/*.png"))) == 2


def test_missing_split_and_outside_output(tmp_path):
    data = LabelDataset(tmp_path / "資料")
    key = data.add_image(Image.new("RGB", (20, 20)))
    data.save(key, [[0, 0, 0, 10, 10]])
    with pytest.raises(ValueError, match="val"):
        data.export()
    with pytest.raises(ValueError):
        project_path(ROOT.parent / "outside")


def test_training_rejects_missing_label_and_duplicate(tmp_path):
    data, _ = dataset(tmp_path)
    path = data.export()
    train_file = next((path.parent / "images/train").glob("*.png"))
    val_file = next((path.parent / "images/val").glob("*.png"))
    val_file.write_bytes(train_file.read_bytes())
    with pytest.raises(ValueError, match="重複"):
        validate_data(path)
    path = data.export()
    next((path.parent / "labels/val").glob("*.txt")).unlink()
    with pytest.raises(ValueError, match="缺少標註"):
        validate_data(path)


def test_training_warns_missing_classes_without_changing_ids(tmp_path, capsys):
    data, keys = dataset(tmp_path)
    data.add_class("鉛筆"); data.add_class("自動比")
    data.save(keys[1], [[1, 10, 20, 80, 90]], "val")
    path = data.export()
    assert validate_data(path) == path
    output = capsys.readouterr().out
    assert "train 標註框數：橡皮擦=1、鉛筆=0、自動比=0" in output
    assert "val 標註框數：橡皮擦=0、鉛筆=1、自動比=0" in output
    assert "沒有訓練框：鉛筆、自動比" in output
    assert "沒有驗證框：橡皮擦、自動比" in output
    assert data.classes == ["橡皮擦", "鉛筆", "自動比"]
    assert data.data["images"][keys[1]]["boxes"][0][0] == 1


@pytest.mark.parametrize("batch", [None, 1, 8])
def test_training_contract(tmp_path, monkeypatch, batch):
    import ultralytics
    from ultralytics.data import utils as data_utils
    original_font_check = data_utils.check_font
    data, _ = dataset(tmp_path)
    path = data.export()
    weights = tmp_path / "base.pt"; weights.write_bytes(b"fake")
    output, calls = tmp_path / "run", {}
    class Model:
        def __init__(self, file):
            calls["weights"] = file
        def train(self, **kwargs):
            assert data_utils.check_font("Arial.Unicode.ttf") is None
            calls.update(kwargs)
            folder = Path(kwargs["project"]) / kwargs["name"]
            (folder / "weights").mkdir(parents=True)
            (folder / "weights/best.pt").write_bytes(b"fake")
            self.trainer = SimpleNamespace(save_dir=folder)
    monkeypatch.setattr(ultralytics, "YOLO", Model)
    import detector
    monkeypatch.setattr(detector, "_device", lambda: "0")
    options = {} if batch is None else {"batch": batch}
    assert train(path, weights=weights, out_dir=output, **options) == output / "weights/best.pt"
    assert calls["weights"] == str(weights)
    expected_batch = 4 if batch is None else batch
    assert (calls["batch"], calls["nbs"], calls["workers"], calls["device"], calls["exist_ok"]) == (expected_batch, expected_batch, 0, "0", False)
    assert {key: calls[key] for key in ("optimizer", "lr0", "warmup_epochs", "cos_lr", "mosaic", "close_mosaic", "scale", "freeze", "degrees")} == {
        "optimizer": "AdamW", "lr0": 0.001, "warmup_epochs": 0, "cos_lr": True,
        "mosaic": 0.2, "close_mosaic": 10, "scale": 0.3, "freeze": 10, "degrees": 15,
    }
    assert calls["epochs"] == 150
    assert data_utils.check_font is original_font_check
    with pytest.raises(ValueError, match="已存在"):
        train(path, weights=weights, out_dir=output)


def test_training_font_check_restored_after_error(tmp_path, monkeypatch):
    import ultralytics
    from ultralytics.data import utils as data_utils
    original = data_utils.check_font
    data, _ = dataset(tmp_path)
    path = data.export()
    weights = tmp_path / "base.pt"; weights.write_bytes(b"fake")
    class BrokenModel:
        def __init__(self, _):
            pass
        def train(self, **kwargs):
            assert data_utils.check_font("Arial.Unicode.ttf") is None
            raise RuntimeError("fake training failure")
    monkeypatch.setattr(ultralytics, "YOLO", BrokenModel)
    with pytest.raises(RuntimeError, match="fake training"):
        train(path, weights=weights, out_dir=tmp_path / "error_run")
    assert data_utils.check_font is original


class FakeCapture:
    def __init__(self, worker, opened=True):
        self.worker, self.opened, self.released = worker, opened, False
    def isOpened(self):
        return self.opened
    def read(self):
        self.worker.stop_event.set()
        return True, np.zeros((20, 30, 3), np.uint8)
    def release(self):
        self.released = True


def test_camera_custom_weights_and_release(tmp_path):
    weights = tmp_path / "best.pt"; weights.write_bytes(b"fake")
    calls = {}
    def detect(frame, **kwargs):
        calls.update(kwargs)
        return [{"label": "橡皮擦", "confidence": 0.9, "box": [1, 1, 10, 10]}]
    worker = CameraWorker(weights=weights, classes=["橡皮擦"], detector=detect, loader=lambda _: SimpleNamespace(names={0: "橡皮擦"}))
    cap = FakeCapture(worker)
    worker.capture = lambda _: cap
    worker.run()
    message = worker.messages.get_nowait()
    assert message[0] == "frame" and message[2][0]["label"] == "橡皮擦"
    assert calls["weights"] == str(weights) and cap.released


def test_camera_missing_wrong_model_and_failure(tmp_path):
    worker = CameraWorker(weights=tmp_path / "missing.pt", classes=["橡皮擦"], capture=lambda _: pytest.fail("不應開鏡頭"))
    worker.run()
    assert worker.messages.get_nowait()[0] == "error"
    weights = tmp_path / "best.pt"; weights.write_bytes(b"fake")
    worker = CameraWorker(weights=weights, classes=["橡皮擦"], loader=lambda _: SimpleNamespace(names={0: "person"}), capture=lambda _: pytest.fail("不應開鏡頭"))
    worker.run()
    assert "類別" in worker.messages.get_nowait()[1]
    worker = CameraWorker()
    cap = FakeCapture(worker, opened=False); worker.capture = lambda _: cap
    worker.run()
    assert cap.released and worker.messages.get_nowait()[0] == "error"


def test_camera_inference_error_still_releases(tmp_path):
    weights = tmp_path / "best.pt"; weights.write_bytes(b"fake")
    def broken(*args, **kwargs):
        raise RuntimeError("推論測試失敗")
    worker = CameraWorker(weights=weights, classes=["橡皮擦"], loader=lambda _: SimpleNamespace(names={0: "橡皮擦"}), detector=broken)
    cap = FakeCapture(worker); worker.capture = lambda _: cap
    worker.run()
    assert cap.released and "推論測試失敗" in worker.messages.get_nowait()[1]


def test_empty_data_gui_preparation_does_not_launch_training(tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk(); root.withdraw()
    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda title, text, **kwargs: errors.append(text))
    app = Labeler(root, tmp_path / "空白資料")
    try:
        app.start_train()
        task = app.tasks.get(timeout=5)
        app.tasks.put(task)
        app.poll()
        assert errors and app.process is None and not app.preparing
        assert "train" in errors[0]
        app.close()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_close_waits_for_camera_release(tmp_path):
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    app = Labeler(root, tmp_path / "關閉測試")
    alive = [True]
    app.worker = SimpleNamespace(stop=lambda: None, thread=SimpleNamespace(is_alive=lambda: alive[0]))
    app.close()
    assert root.winfo_exists() and app.closing
    alive[0] = False
    app.finish_close()


def test_gui_photo_annotation_reload_smoke(tmp_path):
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    try:
        app = Labeler(root, tmp_path / "GUI資料")
        root.update()
        app.key = app.dataset.add_image(Image.new("RGB", (200, 100), "red"))
        app.load(app.key)
        app.geometry = (2, 0, 10)
        app.drag_start(SimpleNamespace(x=20, y=30))
        app.drag_end(SimpleNamespace(x=160, y=190))
        app.save()
        app.load(app.key)
        assert app.boxes[0] == pytest.approx([0, 10, 10, 80, 90])
        assert app.canvas.find_all()
        app.close()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
