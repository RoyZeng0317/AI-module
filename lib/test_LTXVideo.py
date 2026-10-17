"""Unit tests for lib/LTXVideo.py.

Same contract as test_IndexTTS2.py / test_SDXL.py: never touch a real
GPU/checkpoint. The "not installed" path is tested for real (this sandbox
genuinely doesn't have the `diffusers` package), and the "installed" path is
tested by injecting a fake `diffusers` module into sys.modules.
"""

import sys
import types

import pytest

import lib.LTXVideo as ltxvideo


@pytest.fixture(autouse=True)
def _reset_cached_pipeline():
    ltxvideo._pipeline = None
    yield
    ltxvideo._pipeline = None


def test_get_pipeline_raises_friendly_error_when_package_not_installed():
    with pytest.raises(RuntimeError, match="未安裝"):
        ltxvideo._get_pipeline()


class _FakeFrames:
    def __init__(self, frames):
        self.frames = [frames]


class _FakePipeline:
    def __init__(self, **kwargs):
        self.from_pretrained_kwargs = kwargs
        self.offloaded = False
        self.calls: list[dict] = []

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def __call__(self, prompt, negative_prompt, width, height, num_frames, num_inference_steps):
        self.calls.append({
            "prompt": prompt, "negative_prompt": negative_prompt, "width": width,
            "height": height, "num_frames": num_frames, "num_inference_steps": num_inference_steps,
        })
        return _FakeFrames(frames=["frame0", "frame1"])


class _FakeLTXPipeline:
    captured_kwargs: dict = {}

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.captured_kwargs = {"model_id": model_id, **kwargs}
        return _FakePipeline(**kwargs)


def _install_fake_diffusers_package(monkeypatch):
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.LTXPipeline = _FakeLTXPipeline

    exported_videos: list[dict] = []
    fake_utils = types.ModuleType("diffusers.utils")

    def _fake_export_to_video(frames, output_path, fps):
        exported_videos.append({"frames": frames, "output_path": output_path, "fps": fps})

    fake_utils.export_to_video = _fake_export_to_video
    fake_diffusers.utils = fake_utils

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", fake_utils)

    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    return exported_videos


def test_get_pipeline_loads_once_and_caches_with_offload_enabled(monkeypatch, tmp_path):
    _install_fake_diffusers_package(monkeypatch)

    pipeline1 = ltxvideo._get_pipeline(cache_dir=tmp_path)
    pipeline2 = ltxvideo._get_pipeline(cache_dir=tmp_path)

    assert pipeline1 is pipeline2
    assert isinstance(pipeline1, _FakePipeline)
    assert pipeline1.offloaded is True
    assert _FakeLTXPipeline.captured_kwargs["model_id"] == "Lightricks/LTX-Video"
    assert _FakeLTXPipeline.captured_kwargs["cache_dir"] == str(tmp_path)


def test_ltxvideo_generate_calls_pipeline_and_exports_video(monkeypatch, tmp_path):
    exported_videos = _install_fake_diffusers_package(monkeypatch)
    out_path = tmp_path / "out.mp4"

    result = ltxvideo.ltxvideo_generate("一隻貓在花園裡奔跑", output_path=out_path,
                                         cache_dir=tmp_path)

    assert result == str(out_path)
    pipeline = ltxvideo._pipeline
    assert pipeline.calls == [{
        "prompt": "一隻貓在花園裡奔跑",
        "negative_prompt": ltxvideo.DEFAULT_NEGATIVE_PROMPT,
        "width": 704, "height": 480, "num_frames": 161, "num_inference_steps": 50,
    }]
    assert exported_videos == [
        {"frames": ["frame0", "frame1"], "output_path": str(out_path), "fps": 24}
    ]
