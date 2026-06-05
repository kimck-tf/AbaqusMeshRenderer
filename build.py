"""One-command onefile build for meshview.

Run ``python build.py`` (on Windows for a ``.exe``, but it works on any OS) to
produce a single self-contained executable in ``dist/``. The same invocation is
used in CI so packaging — the project's biggest risk — is verified continuously.

Heavy GUI frameworks (Qt/VTK) are intentionally absent, so the binary stays
small. moderngl loads its GL backend through ``glcontext`` at runtime and
moderngl-window discovers windowing backends dynamically, so both are collected
explicitly below; the GLSL shaders are bundled as data.

Optional flags:
  --upx        enable UPX compression (smaller, but may trip antivirus)
  --debug      keep the build console / verbose PyInstaller output
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SHADERS = ROOT / "meshview" / "render" / "shaders"


def build(upx: bool = False, debug: bool = False) -> int:
    try:
        import PyInstaller.__main__ as pyi
    except ImportError:
        print("PyInstaller is required: pip install pyinstaller", file=sys.stderr)
        return 1

    sep = os.pathsep  # ':' on POSIX, ';' on Windows
    args = [
        str(ROOT / "meshview" / "__main__.py"),
        "--name", "meshview",
        "--onefile",
        "--clean",
        "--noconfirm",
        # Bundle the GLSL shaders next to their package.
        "--add-data", f"{SHADERS}{sep}meshview/render/shaders",
        # moderngl GL backend + windowing backends are imported dynamically.
        "--collect-all", "glcontext",
        "--collect-all", "moderngl_window",
        "--collect-submodules", "pyglet",
        # Trim obvious bloat we never import.
        "--exclude-module", "tkinter.test",
        "--exclude-module", "test",
    ]
    if not upx:
        args.append("--noupx")  # otherwise PyInstaller uses UPX if found on PATH
    if not debug:
        args.append("--log-level=WARN")

    print("Running PyInstaller:\n  " + " ".join(args))
    pyi.run(args)

    exe = ROOT / "dist" / ("meshview.exe" if os.name == "nt" else "meshview")
    if exe.exists():
        size_mb = exe.stat().st_size / (1024 * 1024)
        print(f"\nBuilt {exe} ({size_mb:.1f} MB)")
        return 0
    print("Build finished but executable not found", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # A stale spec from a previous run can shadow our arguments.
    spec = ROOT / "meshview.spec"
    if spec.exists():
        spec.unlink()
    if (ROOT / "build").exists():
        shutil.rmtree(ROOT / "build")
    return build(upx="--upx" in argv, debug="--debug" in argv)


if __name__ == "__main__":
    raise SystemExit(main())
