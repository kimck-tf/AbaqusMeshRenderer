from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import build


def _capture_pyinstaller_args(monkeypatch, tmp_path: Path) -> list[str]:
    captured: list[str] = []
    package = ModuleType("PyInstaller")
    package.__path__ = []  # type: ignore[attr-defined]
    main = ModuleType("PyInstaller.__main__")

    def fake_run(args: list[str]) -> None:
        captured.extend(args)
        output = tmp_path / "dist" / "meshview.exe"
        output.parent.mkdir()
        output.touch()

    main.run = fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PyInstaller", package)
    monkeypatch.setitem(sys.modules, "PyInstaller.__main__", main)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "SHADERS", tmp_path / "meshview" / "render" / "shaders")
    return captured


def test_release_build_hides_console(monkeypatch, tmp_path: Path) -> None:
    args = _capture_pyinstaller_args(monkeypatch, tmp_path)

    assert build.build(debug=False) == 0
    assert "--windowed" in args


def test_debug_build_keeps_console(monkeypatch, tmp_path: Path) -> None:
    args = _capture_pyinstaller_args(monkeypatch, tmp_path)

    assert build.build(debug=True) == 0
    assert "--windowed" not in args
