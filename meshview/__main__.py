"""Allow ``python -m meshview`` and serve as the PyInstaller entry script.

Uses an absolute import so the same file works both as ``python -m meshview``
(package context present) and as PyInstaller's top-level frozen script (where
there is no parent package).
"""

from meshview.app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
