"""Unit tests for package_checkpoint.py — bundling a trained checkpoint +
its loader code into a portable, self-contained .zip.
"""

import zipfile

import pytest

from package_checkpoint import package_checkpoint


def _write_fake_checkpoint(checkpoint_dir):
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(b"fake-weights")
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "bpe_vocab.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "bpe_merges.json").write_text("[]", encoding="utf-8")
    (checkpoint_dir / "history.json").write_text("[]", encoding="utf-8")


def test_package_checkpoint_includes_checkpoint_and_source_files(tmp_path):
    checkpoint_dir = tmp_path / "gpt_code_pretrain_runs"
    _write_fake_checkpoint(checkpoint_dir)
    out_path = tmp_path / "dist" / "gpt_code_pretrain_runs.zip"

    summary = package_checkpoint(checkpoint_dir, out_path)

    assert out_path.exists()
    with zipfile.ZipFile(out_path) as zf:
        names = set(zf.namelist())
    assert {"model.pt", "config.json", "bpe_vocab.json", "bpe_merges.json",
            "transformer_chat.py", "bpe_tokenizer.py", "README.txt"} <= names
    assert set(summary["included_files"]) <= names


def test_package_checkpoint_readme_mentions_correct_command_for_pretrain_only(tmp_path):
    checkpoint_dir = tmp_path / "gpt_code_pretrain_runs"
    _write_fake_checkpoint(checkpoint_dir)
    out_path = tmp_path / "out.zip"

    package_checkpoint(checkpoint_dir, out_path)

    with zipfile.ZipFile(out_path) as zf:
        readme = zf.read("README.txt").decode("utf-8")
    assert "complete" in readme
    assert "XPU" in readme or "Arc" in readme


def test_package_checkpoint_rejects_incomplete_checkpoint_dir(tmp_path):
    checkpoint_dir = tmp_path / "half_written_run"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    # model.pt/bpe_vocab.json/bpe_merges.json missing -- simulates a
    # checkpoint dir caught mid-training, must not be packaged as if complete.

    with pytest.raises(FileNotFoundError):
        package_checkpoint(checkpoint_dir, tmp_path / "out.zip")


def test_package_checkpoint_omits_missing_optional_history_file(tmp_path):
    checkpoint_dir = tmp_path / "run_without_history"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_bytes(b"fake")
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "bpe_vocab.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "bpe_merges.json").write_text("[]", encoding="utf-8")
    out_path = tmp_path / "out.zip"

    summary = package_checkpoint(checkpoint_dir, out_path)

    assert "history.json" not in summary["included_files"]
