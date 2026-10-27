from pathlib import Path

import pytest

from character_weights import publish_weights


def test_publish_preserves_existing_and_rejects_empty(tmp_path):
    source, target = tmp_path / "best.pt", tmp_path / "characters.pt"
    source.write_bytes(b"checkpoint"); target.write_bytes(b"old")
    assert publish_weights(source, target) == target
    assert target.read_bytes() == b"checkpoint" and source.read_bytes() == b"checkpoint"
    assert next((tmp_path / "characters_backup").glob("*/characters.pt")).read_bytes() == b"old"
    source.write_bytes(b"")
    with pytest.raises(ValueError):
        publish_weights(source, target)
    assert target.read_bytes() == b"checkpoint"


def test_person_gui_uses_separate_dataset_and_output(tmp_path, monkeypatch):
    import tkinter as tk
    from lib.yolo import character_labeler as module
    monkeypatch.setattr(module, "DATASET", tmp_path / "people")
    monkeypatch.setattr(module, "RUNS", tmp_path / "person_runs")
    monkeypatch.setattr(module, "WEIGHTS", tmp_path / "characters.pt")
    monkeypatch.setattr(module, "publish_weights", lambda source: publish_weights(source, module.WEIGHTS))
    root = tk.Tk(); root.withdraw()
    try:
        app = module.CharacterLabeler(root)
        import detector
        cleared = []
        monkeypatch.setattr(detector._model, "cache_clear", lambda: cleared.append(True))
        assert app.dataset.classes == ["人物1"]
        assert app.dataset.folder == tmp_path / "people" and app.runs_root == tmp_path / "person_runs"
        assert not app.best.get()
        source = tmp_path / "best.pt"; source.write_bytes(b"checkpoint")
        app.checkpoint_choice.set("best")
        app.training_finished(source)
        assert Path(app.best.get()) == tmp_path / "characters.pt"
        assert (tmp_path / "characters.pt").read_bytes() == b"checkpoint"
        last = source.with_name("last.pt"); last.write_bytes(b"last-checkpoint")
        app.checkpoint_choice.set("last")
        app.training_finished(source)
        assert (tmp_path / "characters.pt").read_bytes() == b"last-checkpoint"
        assert len(cleared) == 2
        app.close()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_camera_threshold_is_configurable(tmp_path, monkeypatch):
    import tkinter as tk
    from types import SimpleNamespace
    from lib.yolo.object_labeler import CameraWorker, Labeler
    root = tk.Tk(); root.withdraw()
    monkeypatch.setattr(CameraWorker, "start", lambda self: setattr(self, "thread", SimpleNamespace(is_alive=lambda: False)))
    app = Labeler(root, tmp_path / "threshold")
    try:
        app.confidence.set("0.15")
        app.preview()
        assert app.worker.conf == .15
        app.stop_camera(); app.pending_camera = False
        app.confidence.set("0")
        with pytest.raises(ValueError, match="門檻"):
            app.preview()
        app.close()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
