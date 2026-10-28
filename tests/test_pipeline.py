from __future__ import annotations

import sys
from pathlib import Path

import pytest

from run_pipeline import resolve_data_dir, resolve_python


def test_resolve_python_falls_back_to_running_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("run_pipeline.RUNTIME", tmp_path)
    assert resolve_python(None) == Path(sys.executable)


def test_resolve_python_rejects_missing_explicit_interpreter(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_python(str(tmp_path / "missing-python"))


def test_resolve_data_dir_accepts_existing_directory(tmp_path: Path) -> None:
    assert resolve_data_dir(str(tmp_path)) == tmp_path.resolve()


def test_resolve_data_dir_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(str(tmp_path / "missing-data"))
