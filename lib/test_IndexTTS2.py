"""Unit tests for lib/IndexTTS2.py.

Same contract as test_NVIDIA.py: never touch a real GPU/checkpoint. The
"not installed" path is tested for real (this sandbox genuinely doesn't have
the `indextts` package), and the "installed" path is tested by injecting a
fake `indextts.infer_v2` module into sys.modules before calling in.
"""

import sys
import types

import pytest

import lib.IndexTTS2 as indextts2


@pytest.fixture(autouse=True)
def _reset_cached_model():
    indextts2._model = None
    yield
    indextts2._model = None


def test_get_model_raises_friendly_error_when_package_not_installed():
    with pytest.raises(RuntimeError, match="未安裝"):
        indextts2._get_model()


class _FakeIndexTTS2Impl:
    def __init__(self, cfg_path: str, model_dir: str):
        self.cfg_path = cfg_path
        self.model_dir = model_dir
        self.infer_calls: list[dict] = []

    def infer(self, spk_audio_prompt, text, output_path):
        self.infer_calls.append(
            {"spk_audio_prompt": spk_audio_prompt, "text": text, "output_path": output_path}
        )


def _install_fake_indextts_package(monkeypatch, fake_impl_cls=_FakeIndexTTS2Impl):
    fake_module = types.ModuleType("indextts.infer_v2")
    fake_module.IndexTTS2 = fake_impl_cls
    monkeypatch.setitem(sys.modules, "indextts", types.ModuleType("indextts"))
    monkeypatch.setitem(sys.modules, "indextts.infer_v2", fake_module)


def test_get_model_raises_friendly_error_when_checkpoint_missing(monkeypatch, tmp_path):
    _install_fake_indextts_package(monkeypatch)
    with pytest.raises(RuntimeError, match="checkpoint"):
        indextts2._get_model(checkpoint_dir=tmp_path / "no_such_checkpoint")


def test_get_model_loads_once_and_caches(monkeypatch, tmp_path):
    _install_fake_indextts_package(monkeypatch)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    model1 = indextts2._get_model(checkpoint_dir=tmp_path)
    model2 = indextts2._get_model(checkpoint_dir=tmp_path)

    assert model1 is model2
    assert isinstance(model1, _FakeIndexTTS2Impl)
    assert model1.model_dir == str(tmp_path)


def test_indextts2_speak_calls_infer_with_expected_args(monkeypatch, tmp_path):
    _install_fake_indextts_package(monkeypatch)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    out_wav = tmp_path / "out.wav"

    result = indextts2.indextts2_speak(
        "你好，這是測試", spk_audio_prompt=tmp_path / "ref.wav",
        output_path=out_wav, checkpoint_dir=tmp_path,
    )

    assert result == str(out_wav)
    model = indextts2._model
    assert model.infer_calls == [
        {"spk_audio_prompt": str(tmp_path / "ref.wav"), "text": "你好，這是測試",
         "output_path": str(out_wav)}
    ]
