"""本輪附件白名單與預覽子程序，不依賴實際桌面或網路。"""
import json
from PIL import Image
import lib.components.image_search_viewer as viewer


def test_preview_only_uses_this_reply_and_cache_pngs(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    allowed, outside = cache / "a.png", tmp_path / "outside.png"
    Image.new("RGB", (4, 4)).save(allowed)
    Image.new("RGB", (4, 4)).save(outside)
    monkeypatch.setattr(viewer, "CACHE_DIR", cache)
    reply = f"![候選](<{allowed.as_posix()}>)\n![錯誤](<{outside.as_posix()}>)\n![網路](<https://example.com/a.png>)"
    assert viewer.attachment_paths(reply) == [allowed]
    assert viewer.attachment_paths("下一輪沒有附件") == []


def test_open_previews_uses_argv_no_shell_and_no_redownload(monkeypatch, tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(path)
    monkeypatch.setattr(viewer, "CACHE_DIR", tmp_path)
    seen = []
    monkeypatch.setattr(viewer.subprocess, "Popen", lambda args, **kw: seen.append((args, kw)))
    assert viewer.open_previews(f"![預覽](<{path.as_posix()}>)") == 1
    args, kw = seen[0]
    assert args[0] == viewer.sys.executable and args[-1] == str(path)
    assert "shell" not in kw and kw["cwd"] == viewer.ROOT
    assert viewer.open_previews("沒有附件") == 0 and len(seen) == 1


def test_preview_cap_and_deduplication(monkeypatch, tmp_path):
    monkeypatch.setattr(viewer, "CACHE_DIR", tmp_path)
    paths = [tmp_path / f"{i}.png" for i in range(5)]
    for path in paths:
        path.touch()
    reply = "\n".join(f"![candidate](<{path.as_posix()}>)" for path in [paths[0], *paths])
    assert viewer.attachment_paths(reply) == paths[:3]
