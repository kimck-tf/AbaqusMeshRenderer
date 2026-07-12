from __future__ import annotations

import sys

import moderngl_window as mglw

from meshview.render import viewport


def test_run_viewer_shields_mesh_path_from_moderngl_cli(monkeypatch) -> None:
    parsed_args = None

    def fake_run_window_config(config_cls, args=None) -> None:
        nonlocal parsed_args
        parser = mglw.create_parser()
        parsed_args = mglw.parse_args(args=args, parser=parser)

    monkeypatch.setattr(sys, "argv", ["meshview.exe", "model.inp"])
    monkeypatch.setattr(mglw, "run_window_config", fake_run_window_config)

    viewport.run_viewer("model.inp")

    assert parsed_args is not None
