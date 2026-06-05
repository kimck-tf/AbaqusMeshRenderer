"""``meshview`` command-line entry point.

Usage::

    meshview model.inp        # open the viewer on a file
    meshview                  # open a File-dialog, then the viewer
    meshview --info model.blk # print mesh statistics, no window

File paths are never hard-coded: they come from the CLI argument, a File-Open
dialog, or drag-and-drop onto the window.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .. import __version__
from ..core.surface import extract_surface
from ..io.loader import load_mesh, supported_extensions


def _print_info(path: str) -> int:
    """Load ``path`` and print mesh statistics without opening a window."""

    mesh = load_mesh(path)
    surface = extract_surface(mesh)
    lo, hi = mesh.bounding_box()

    print(f"File              : {path}")
    print(f"Nodes             : {mesh.n_nodes}")
    print(f"Elements          : {mesh.n_elements}")
    print("Element blocks    :")
    for block in mesh.element_blocks:
        print(f"  - {block.elem_type:<8} {block.count}")
    print(f"Surface triangles : {surface.n_triangles}")
    print(f"Surface vertices  : {surface.n_vertices}")
    print(f"Bounding box min  : ({lo[0]:.4g}, {lo[1]:.4g}, {lo[2]:.4g})")
    print(f"Bounding box max  : ({hi[0]:.4g}, {hi[1]:.4g}, {hi[2]:.4g})")
    if mesh.node_sets:
        print(f"Node sets         : {', '.join(mesh.node_sets)}")
    if mesh.elem_sets:
        print(f"Element sets      : {', '.join(mesh.elem_sets)}")
    if mesh.ignored_elements:
        summary = ", ".join(f"{t} (x{n})" for t, n in mesh.ignored_elements.items())
        print(f"Ignored (unsupported): {summary}")
    return 0


def _ask_open_dialog() -> str | None:
    """Show a native File-Open dialog; return the chosen path or ``None``."""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:  # noqa: BLE001 - tk may be unavailable
        return None
    root = tk.Tk()
    root.withdraw()
    exts = " ".join(f"*{e}" for e in supported_extensions())
    path = filedialog.askopenfilename(
        title="Open mesh", filetypes=[("Abaqus mesh", exts), ("All files", "*.*")]
    )
    root.destroy()
    return path or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meshview", description=__doc__)
    p.add_argument("path", nargs="?", help="Abaqus .inp/.blk file to open")
    p.add_argument("--info", action="store_true", help="print mesh stats and exit")
    p.add_argument("--version", action="version", version=f"meshview {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.info:
        if not args.path:
            print("error: --info requires a file path", file=sys.stderr)
            return 2
        return _print_info(args.path)

    path = args.path or _ask_open_dialog()
    # Import the GL layer lazily so --info works in headless environments.
    from ..render.viewport import run_viewer

    run_viewer(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
