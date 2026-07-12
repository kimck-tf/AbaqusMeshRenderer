# Windows Onefile Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 콘솔 창이 보이지 않는 Windows 단일 실행 파일 `dist/meshview.exe`를 생성하고 독립 실행을 검증한다.

**Architecture:** 기존 `build.py`의 PyInstaller `--onefile` 구성을 유지한다. 릴리스 빌드 인자에만 `--windowed`를 추가하고 `--debug` 빌드는 콘솔을 유지하며, PyInstaller 호출을 가짜 모듈로 대체한 단위 테스트로 두 정책을 검증한다.

**Tech Stack:** Python 3.10+, PyInstaller 6+, pytest 9, Ruff

## Global Constraints

- 산출물 이름은 `dist/meshview.exe`이다.
- 기본 빌드에서는 콘솔 창을 표시하지 않는다.
- `--debug` 빌드에서는 진단용 콘솔을 유지한다.
- 설치 프로그램, 바로가기, 아이콘, 자동 업데이트는 추가하지 않는다.
- 파일 인코딩은 UTF-8을 사용한다.

---

### Task 1: 릴리스와 디버그 빌드의 콘솔 정책

**Files:**
- Create: `tests/test_build.py`
- Modify: `build.py:38-65`

**Interfaces:**
- Consumes: `build.build(upx: bool = False, debug: bool = False) -> int`
- Produces: 릴리스 PyInstaller 인자에는 `--windowed`가 포함되고 디버그 인자에는 포함되지 않는 동작

- [ ] **Step 1: 실패하는 테스트 작성**

```python
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
```

- [ ] **Step 2: 테스트가 예상대로 실패하는지 확인**

Run: `python -m pytest tests/test_build.py -v`

Expected: `test_release_build_hides_console`가 `assert "--windowed" in args`에서 FAIL하고 `test_debug_build_keeps_console`는 PASS한다.

- [ ] **Step 3: 최소 구현 작성**

`build.py`의 `args` 목록 생성 뒤 디버그 분기에 다음 정책을 적용한다.

```python
if not debug:
    args.append("--windowed")
    args.append("--log-level=WARN")
```

기존의 별도 `if not debug: args.append("--log-level=WARN")` 분기는 위 코드로 대체한다.

- [ ] **Step 4: 대상 테스트 통과 확인**

Run: `python -m pytest tests/test_build.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: 전체 회귀 검사와 정적 검사**

Run: `python -m pytest -v`

Expected: 모든 비-GL 테스트가 PASS하고 환경에서 GL 컨텍스트를 만들 수 없으면 GL 테스트만 SKIP한다.

Run: `python -m ruff check meshview tests build.py`

Expected: `All checks passed!`

- [ ] **Step 6: 구현 커밋**

```powershell
git add -- build.py tests/test_build.py
git commit -m "build: hide console in release executable"
```

### Task 2: Windows 단일 실행 파일 생성과 검증

**Files:**
- Generated: `dist/meshview.exe`

**Interfaces:**
- Consumes: `python build.py`, `tests/fixtures/single_hex.inp`
- Produces: Python 설치 없이 실행 가능한 `dist/meshview.exe`

- [ ] **Step 1: 릴리스 실행 파일 빌드**

Run: `python build.py`

Expected: 종료 코드 0, 마지막 출력에 `Built ...\dist\meshview.exe`와 파일 크기가 표시된다.

- [ ] **Step 2: 산출물과 크기 확인**

```powershell
$exe = Get-Item 'dist/meshview.exe'
[pscustomobject]@{ Path = $exe.FullName; SizeMB = [math]::Round($exe.Length / 1MB, 1) }
```

Expected: 파일이 존재하고 `SizeMB`가 100 이하이다.

- [ ] **Step 3: 독립 실행 정보 모드 확인**

Run: `& 'dist/meshview.exe' --info 'tests/fixtures/single_hex.inp'; $LASTEXITCODE`

Expected: Nodes 8, Elements 1, Surface triangles 12가 출력되고 종료 코드는 0이다.

- [ ] **Step 4: GUI 기동 스모크 검사**

```powershell
$process = Start-Process -FilePath (Resolve-Path 'dist/meshview.exe') -ArgumentList (Resolve-Path 'tests/fixtures/single_hex.inp') -PassThru
Start-Sleep -Seconds 5
$started = -not $process.HasExited
if ($started) { Stop-Process -Id $process.Id }
$started
```

Expected: `True`. 실행 후 5초 동안 프로세스가 유지되어 렌더링 창의 정상 기동을 입증한다.

- [ ] **Step 5: 최종 작업 트리 확인**

Run: `git status --short`

Expected: 사용자의 기존 미추적 `AGENTS.md`와 빌드 산출물 외에 의도하지 않은 소스 변경이 없다.
