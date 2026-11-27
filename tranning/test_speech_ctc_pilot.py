"""speech_ctc_pilot.py 的工程測試：用合成音和極小的假 checkpoint，不讀真資料、不讀正式權重。
只驗證長度、對齊、遮罩、停止與防護規則，不代表任何辨識品質。
執行：python -m pytest tranning/test_speech_ctc_pilot.py
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_ctc_pilot as sp  # noqa: E402
import speech_to_text as stt  # noqa: E402
from OCR import CRNN  # noqa: E402

CHARS = ["1", "2", "A", "B"]


def tone(seconds, freq=440.0):
    t = np.arange(int(stt.SAMPLE_RATE * seconds)) / stt.SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(stt.SAMPLE_RATE)
        w.writeframes((samples * 32767).astype(np.int16).tobytes())


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    monkeypatch.setattr(sp, "ROOT", root)
    base = root / "tranning" / "speech_runs"
    base.mkdir(parents=True)
    charset = {c: i + 1 for i, c in enumerate(CHARS)}
    (base / "charset.json").write_text(json.dumps(charset), encoding="utf-8")
    (base / "config.json").write_text(json.dumps({"max_frames": 200, "hidden_size": 16}), encoding="utf-8")
    torch.manual_seed(0)
    torch.save(CRNN(len(charset) + 1, stt.FREQ_BINS, 16).state_dict(), base / "best_model.pt")
    data = root / "data"

    def manifest(name, rows):
        for audio, _, seconds in rows:
            write_wav(data / audio, tone(seconds))
        p = data / name
        p.write_text(json.dumps([{"audio": a, "text": t} for a, t, _ in rows], ensure_ascii=False), encoding="utf-8")
        return p

    train = manifest("train.json", [(f"t{i}.wav", "AB" if i % 2 else "12", 0.4 + 0.05 * i) for i in range(6)])
    val = manifest("val.json", [(f"v{i}.wav", "BA", 0.5) for i in range(4)])
    return {"root": root, "base": base, "train": train, "val": val, "data": data, "manifest": manifest}


def run(project, *extra, run_name="r1"):
    args = ["--train-manifest", str(project["train"]), "--val-manifest", str(project["val"]),
            "--base-dir", str(project["base"]), "--run-dir", str(project["root"] / "runs" / run_name),
            "--train-count", "6", "--dev-count", "4", "--compare-count", "2", "--batch-size", "2",
            "--max-updates", "3", "--device", "cpu", *map(str, extra)]
    code = sp.main(args)
    status_path = project["root"] / "runs" / run_name / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
    return code, status, project["root"] / "runs" / run_name


# ---------------- 長度與對齊規則 ----------------
def test_cnn_out_len_matches_real_cnn():
    model = CRNN(5, stt.FREQ_BINS, 16).eval()
    for t in range(4, 70):
        assert model.cnn(torch.zeros(1, 1, stt.FREQ_BINS, t)).shape[-1] == sp.cnn_out_len(t)


def test_ctc_min_steps_counts_adjacent_repeats():
    assert sp.ctc_min_steps([1, 1]) == 3
    assert sp.ctc_min_steps([1, 2, 1]) == 3
    assert sp.ctc_min_steps([1, 1, 2, 2]) == 6
    assert sp.ctc_min_steps([]) == 0


def test_global_cer_is_total_edits_over_total_chars():
    assert sp.global_cer([("ab", "ab"), ("abcd", "abxd")]) == pytest.approx(1 / 6)
    assert sp.global_cer([("abc", "")]) == 1.0


def test_packed_batch_equals_each_sample_alone():
    torch.manual_seed(0)
    model = CRNN(5, stt.FREQ_BINS, 16).eval()
    specs = [stt.waveform_to_spectrogram(tone(s, f)) for s, f in ((0.3, 300), (0.8, 500), (0.5, 700))]
    frames = [s.size(1) for s in specs]
    with torch.no_grad():
        batch, lens = sp.forward_packed(model, specs, frames)
        for k, spec in enumerate(specs):
            alone = model(spec.unsqueeze(0).unsqueeze(0))[:, 0]  # 原本的 CRNN.forward，單筆無補零
            assert lens[k] == alone.size(0)
            assert torch.allclose(batch[: lens[k], k], alone, atol=1e-5)  # 補零不影響有效段


# ---------------- 完整流程 ----------------
def test_tiny_run_writes_everything_and_keeps_base(project, monkeypatch):
    def no_fillers(text):
        raise AssertionError("remove_fillers must not be used")

    monkeypatch.setattr(stt, "remove_fillers", no_fillers)
    before = {p.name: p.read_bytes() for p in project["base"].iterdir()}
    code, status, run_dir = run(project)
    assert (code, status["status"]) == (0, "completed"), status
    assert status["stop_reason"] == "max_updates" and status["updates"] == 3
    hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert hist[0]["epoch"] == 0 and hist[0]["updates"] == 0
    for h in hist:
        assert {"train_loss", "val_loss", "val_cer", "updates", "seconds"} <= set(h)
    rows = json.loads((run_dir / "raw_compare.json").read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 2 and {"baseline_output", "final_output", "reference", "seconds"} <= set(rows[0])
    assert (run_dir / "pilot_model.pt").is_file()
    assert {p.name: p.read_bytes() for p in project["base"].iterdir()} == before  # 正式權重不變
    assert run(project)[0] == 2  # 同一個 run 資料夾不能再用


def test_unknown_train_char_stops_without_training(project):
    project["manifest"]("train.json", [(f"t{i}.wav", "AZ", 0.5) for i in range(6)])
    code, status, run_dir = run(project)
    assert (code, status["status"]) == (2, "rejected")
    assert "missing from the base charset" in status["reason"]
    assert not (run_dir / "pilot_model.pt").exists()


def test_incompatible_checkpoint_is_rejected(project):
    charset = {c: i + 1 for i, c in enumerate(CHARS + ["C"])}  # 字表多一個字，權重形狀對不上
    (project["base"] / "charset.json").write_text(json.dumps(charset), encoding="utf-8")
    code, status, _ = run(project)
    assert (code, status["status"]) == (2, "rejected")
    assert "does not match" in status["reason"]


def test_dev_unknown_chars_are_scored_not_dropped(project):
    project["manifest"]("val.json", [(f"v{i}.wav", "BZ" if i == 0 else "BA", 0.5) for i in range(4)])
    code, status, run_dir = run(project)
    assert code == 0
    data = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    assert data["dev_unknown_chars"] == {"Z": 1}
    assert len(data["dev"]) == 4  # 仍在 dev 裡，計入 CER
    pop = data["metric_population"]  # loss 與 CER 的母體不同，要分開寫清楚
    assert (pop["val_loss"]["items"], pop["val_loss"]["reference_chars"]) == (3, 6)
    assert (pop["val_cer"]["items"], pop["val_cer"]["reference_chars"]) == (4, 8)
    assert status["metric_population"] == pop


def test_long_and_unalignable_clips_are_excluded_and_recorded(project):
    rows = [(f"t{i}.wav", "AB", 0.5) for i in range(5)]
    rows.append(("long.wav", "AB", 3.0))
    rows.append(("short.wav", "AABB12AB", 0.1))  # 太短，CTC 放不下
    project["manifest"]("train.json", rows)
    code, status, run_dir = run(project, "--max-audio-seconds", "2", "--train-count", "7")
    assert code == 0
    ex = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))["excluded"]["train"]
    reasons = {e["audio"]: e["reason"] for e in ex}
    assert reasons == {"long.wav": "longer than 2.0 s", "short.wav": "too short for CTC alignment"}


def test_regression_clips_never_enter_train_or_dev(project):
    reg = project["data"] / "reg.json"
    reg.write_text(json.dumps([{"audio": "v0.wav", "text": "BA"}]), encoding="utf-8")
    code, status, run_dir = run(project, "--regression-manifest", reg, "--dev-count", "3")
    assert code == 0
    data = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    assert "v0.wav" not in data["dev"] + data["train"] and data["regression"] == ["v0.wav"]
    rows = json.loads((run_dir / "raw_compare.json").read_text(encoding="utf-8"))["rows"]
    assert rows[-1]["set"] == "regression"


def test_nan_stops_without_retry(project, monkeypatch):
    real = sp.forward_packed

    def nan_forward(model, specs, frames):
        log_probs, lens = real(model, specs, frames)
        return log_probs * float("nan"), lens

    monkeypatch.setattr(sp, "forward_packed", nan_forward)
    code, status, run_dir = run(project)
    assert (code, status["status"]) == (3, "stopped")
    assert "non-finite" in status["reason"]
    assert not (run_dir / "pilot_model.pt").exists()
    # NaN 出現在訓練前的第 0 輪評估：還沒有完成任何一輪，所以沒有 history
    assert not (run_dir / "history.json").exists()


def test_history_is_written_after_every_epoch(project, monkeypatch):
    writes = []
    real = sp.write_json

    def spy(path, obj):
        if Path(path).name == "history.json":
            writes.append([h["epoch"] for h in obj])
        real(path, obj)

    monkeypatch.setattr(sp, "write_json", spy)
    code, _, _ = run(project, "--max-updates", "6", "--epochs", "2")  # 6 筆 / batch 2 = 每 epoch 3 次更新
    assert code == 0
    assert writes == [[0], [0, 1], [0, 1, 2]]


def test_non_finite_parameter_after_update_stops(project, monkeypatch):
    real_step = torch.optim.AdamW.step

    def bad_step(self, *a, **k):
        out = real_step(self, *a, **k)
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(float("nan"))  # 梯度正常，但更新後權重壞掉
        return out

    monkeypatch.setattr(torch.optim.AdamW, "step", bad_step)
    code, status, run_dir = run(project)
    assert (code, status["status"]) == (3, "stopped")
    assert "non-finite parameter after update 1" in status["reason"]
    assert not (run_dir / "pilot_model.pt").exists()  # 壞權重不存檔
    # 訓練中途停止：已完成的第 0 輪紀錄仍在
    hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert [h["epoch"] for h in hist] == [0]


def test_path_guards(project, tmp_path):
    outside = tmp_path / "outside_run"
    for run_dir in (outside, project["root"] / ".env" / "r", project["base"] / "inside"):
        args = ["--train-manifest", str(project["train"]), "--val-manifest", str(project["val"]),
                "--base-dir", str(project["base"]), "--run-dir", str(run_dir), "--device", "cpu"]
        assert sp.main(args) == 2, run_dir
        assert not (run_dir / "status.json").exists()
