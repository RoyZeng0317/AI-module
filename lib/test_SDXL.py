"""Unit tests for lib/SDXL.py.

Same contract as test_IndexTTS2.py: never touch a real GPU/checkpoint. The
"not installed" path is tested for real (this sandbox genuinely doesn't have
the `diffusers` package), and the "installed" path is tested by injecting a
fake `diffusers` module into sys.modules before calling in.
"""

import sys
import types

import pytest

import lib.SDXL as sdxl


@pytest.fixture(autouse=True)
def _reset_cached_pipeline():
    sdxl._pipeline = None
    yield
    sdxl._pipeline = None


def test_get_pipeline_raises_friendly_error_when_package_not_installed():
    with pytest.raises(RuntimeError, match="未安裝"):
        sdxl._get_pipeline()


class _FakeImage:
    def __init__(self):
        self.saved_to: str | None = None

    def save(self, path):
        self.saved_to = path


class _FakeSDXLResult:
    def __init__(self, images):
        self.images = images


class _FakePipeline:
    def __init__(self, **kwargs):
        self.from_pretrained_kwargs = kwargs
        self.offloaded = False
        self.calls: list[dict] = []
        self.last_image: _FakeImage | None = None

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def __call__(self, prompt, negative_prompt=None, num_inference_steps=30):
        self.calls.append({"prompt": prompt, "negative_prompt": negative_prompt,
                            "num_inference_steps": num_inference_steps})
        self.last_image = _FakeImage()
        return _FakeSDXLResult(images=[self.last_image])


class _FakeDiffusionPipeline:
    captured_kwargs: dict = {}

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.captured_kwargs = {"model_id": model_id, **kwargs}
        return _FakePipeline(**kwargs)


def _install_fake_diffusers_package(monkeypatch, torch_module=None):
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.DiffusionPipeline = _FakeDiffusionPipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    if torch_module is None:
        torch_module = types.ModuleType("torch")
        torch_module.float16 = "float16"
    monkeypatch.setitem(sys.modules, "torch", torch_module)


def test_get_pipeline_loads_once_and_caches_with_offload_enabled(monkeypatch, tmp_path):
    _install_fake_diffusers_package(monkeypatch)

    pipeline1 = sdxl._get_pipeline(cache_dir=tmp_path)
    pipeline2 = sdxl._get_pipeline(cache_dir=tmp_path)

    assert pipeline1 is pipeline2
    assert isinstance(pipeline1, _FakePipeline)
    assert pipeline1.offloaded is True
    assert _FakeDiffusionPipeline.captured_kwargs["model_id"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert _FakeDiffusionPipeline.captured_kwargs["cache_dir"] == str(tmp_path)


def test_sdxl_generate_calls_pipeline_and_saves_image(monkeypatch, tmp_path):
    _install_fake_diffusers_package(monkeypatch)
    out_path = tmp_path / "out.png"

    result = sdxl.sdxl_generate("一隻貓在花園裡", output_path=out_path,
                                 negative_prompt="模糊", cache_dir=tmp_path)

    assert result == str(out_path)
    pipeline = sdxl._pipeline
    assert pipeline.calls == [
        {"prompt": "一隻貓在花園裡", "negative_prompt": "模糊", "num_inference_steps": 30}
    ]
    assert pipeline.last_image.saved_to == str(out_path)
