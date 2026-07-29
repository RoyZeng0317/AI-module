"""Unit tests for train_gui.py's pure logic (field validation + CLI-arg
building). Not a full GUI test — that's a manual smoke check (launch it,
see the tabs render) since there's no meaningful way to assert on rendered
widget layout. tk.StringVar/BooleanVar need a live Tcl interpreter to exist,
so these tests spin up one hidden (withdrawn) root window rather than
mocking tkinter away.
"""

import tkinter as tk

import pytest

from train_gui import Field, build_argv, collect_values


@pytest.fixture(scope="module")
def root():
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


def _with_var(root: tk.Tk, f: Field) -> Field:
    if f.kind == "bool":
        f.var = tk.BooleanVar(root, value=bool(f.default))
    else:
        f.var = tk.StringVar(root, value=str(f.default))
    return f


def test_build_argv_skips_empty_string_and_false_bool():
    fields = [
        Field("a", "A", "str", "--a"),
        Field("b", "B", "int", "--b"),
        Field("c", "C", "bool", "--flag"),
    ]
    values = {"a": "", "b": 5, "c": False}
    assert build_argv(fields, values) == ["--b", "5"]


def test_build_argv_includes_bool_flag_when_true():
    fields = [Field("c", "C", "bool", "--flag")]
    assert build_argv(fields, {"c": True}) == ["--flag"]


def test_build_argv_ignores_fields_without_a_flag():
    # circuit_diagram_train.py's mode-specific fields (data.yaml vs
    # dataset-dir+classes) use flag=None and get folded into argv by hand —
    # build_argv must leave them out rather than emitting "--None value".
    fields = [Field("data", "Data", "file", None)]
    assert build_argv(fields, {"data": "some/path.yaml"}) == []


def test_collect_values_rejects_missing_required(root, monkeypatch):
    shown = []
    monkeypatch.setattr("train_gui.messagebox.showerror", lambda *a: shown.append(a))
    f = _with_var(root, Field("data_dir", "資料夾", "dir", "--data-dir", default="", required=True))
    assert collect_values([f]) is None
    assert shown  # a dialog would have been shown


def test_collect_values_rejects_non_numeric_input(root, monkeypatch):
    monkeypatch.setattr("train_gui.messagebox.showerror", lambda *a: None)
    f = _with_var(root, Field("epochs", "Epochs", "int", "--epochs", default="not-a-number"))
    assert collect_values([f]) is None


def test_collect_values_happy_path(root):
    fields = [
        _with_var(root, Field("epochs", "Epochs", "int", "--epochs", default=10)),
        _with_var(root, Field("lr", "LR", "float", "--lr", default=0.001)),
        _with_var(root, Field("flag", "Flag", "bool", "--flag", default=True)),
        _with_var(root, Field("out_dir", "Out", "str", "--out-dir", default="runs")),
    ]
    assert collect_values(fields) == {"epochs": 10, "lr": 0.001, "flag": True, "out_dir": "runs"}


def test_collect_values_optional_blank_field_is_empty_string(root):
    f = _with_var(root, Field("images_root", "Images root", "dir", "--images-root", default=""))
    assert collect_values([f]) == {"images_root": ""}


def test_end_to_end_field_to_argv_matches_script_flags(root):
    # Round-trips a couple of real fields the way _start() does: collect
    # from the (fake) widget values, then build the argv the subprocess
    # actually receives.
    fields = [
        _with_var(root, Field("data_dir", "Data dir", "dir", "--data-dir", default="/tmp/split", required=True)),
        _with_var(root, Field("epochs", "Epochs", "int", "--epochs", default=30)),
        _with_var(root, Field("unfreeze_backbone", "Unfreeze", "bool", "--unfreeze-backbone", default=False)),
    ]
    values = collect_values(fields)
    assert build_argv(fields, values) == ["--data-dir", "/tmp/split", "--epochs", "30"]
