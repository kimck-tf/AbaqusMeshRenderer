# meshview — 경량 3D 메쉬 뷰어 (Abaqus `.inp` / `.blk`)

Abaqus 입력 파일(`*.inp`)과 모델 정의 파일(`*.blk`)을 읽어 **외피 표면**을
셰이딩 + 와이어프레임으로 렌더링하는 경량 Python 3D 유한요소 메쉬 뷰어입니다.
무거운 의존성(VTK/PyVista/Qt)을 배제해 단일 실행파일 크기를 최소화했습니다.

- 렌더링: `moderngl` + `moderngl-window`(pyglet 백엔드)
- 수치/지오메트리: `numpy` (카메라 행렬·쿼터니언 직접 구현)
- 파서: 자체 Abaqus 키워드 파서 (`.inp` / `.blk` 공용)

## 설치

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

```bash
# 파일을 인자로 열기
python -m meshview model.inp

# 인자 없이 실행 → File-Open 다이얼로그
python -m meshview

# 창으로 .inp / .blk 파일을 드래그앤드롭해도 열립니다.

# 렌더링 없이 메쉬 통계만 출력
python -m meshview --info model.blk
```

`--info` 출력 예시:

```
File              : model.inp
Nodes             : 12
Elements          : 2
Element blocks    :
  - C3D8     2
Surface triangles : 20
Surface vertices  : 12
Bounding box min  : (0, 0, 0)
Bounding box max  : (2, 1, 1)
```

## 조작 (기본 프리셋: HyperMesh)

뷰 조작은 모두 `Ctrl + 마우스 버튼`으로 통일되어 있어, 수식어 없는 좌클릭은
선택/픽 용도로 비워 둡니다. 바인딩은 `meshview/camera/controls.py`의 프리셋
(`HYPERMESH`, `GENERAL`)으로 재매핑할 수 있습니다.

| 동작 | 바인딩 |
|---|---|
| 회전 (Rotate) | `Ctrl` + 좌버튼 드래그 |
| 이동 (Pan) | `Ctrl` + 우버튼 드래그 |
| 확대/축소 (Zoom) | `Ctrl` + 중간버튼 드래그 / 스크롤 휠 |
| 화면 맞춤 (Fit) | `Ctrl` + 중간버튼 클릭 / `F` 키 |
| 리셋 (Reset) | `R` 키 |
| 선택/픽 | 좌클릭 (수식어 없음) |
| 셰이딩 토글 | `S` 키 |
| 와이어프레임 토글 | `W` 키 |
| 표준 뷰 | Numpad `1`~`7` (앞/뒤/우/좌/위/아래/등각) |

## 지원 요소 타입

내부 요소는 GPU로 보내지 않고, 솔리드 요소는 **외피 표면**만 추출해 렌더링합니다.
**2차 요소는 코너 노드만** 사용하고 mid-side 노드는 면 정의에서 제외합니다.

| 분류 | 타입 |
|---|---|
| 사면체 (Tetra) | `C3D4`, `C3D10` |
| 육면체 (Hexa) | `C3D8`(/R/H/I), `C3D20`(/R) |
| 쐐기 (Wedge) | `C3D6`, `C3D15` |
| 피라미드 (Pyramid) | `C3D5`, `C3D13` |
| 셸/멤브레인 (Shell) | `S3`/`S3R`/`STRI3`/`S6`, `S4`/`S4R`/`S8`/`S8R`, `M3D*`, `R3D3`/`R3D4` |

**미지원 타입**은 조용히 버리지 않고, 무엇을 무시했는지 경고 로그로 요약합니다
(`--info` 출력의 `Ignored (unsupported)` 항목, 또는 `-v` 로그).

## 빌드 (단일 실행파일)

```bash
pip install pyinstaller
python build.py            # dist/meshview(.exe) 생성
python build.py --upx      # UPX 압축(더 작지만 백신 오탐 가능)
python build.py --debug    # 빌드 콘솔 유지 + 상세 로그
```

PyInstaller `--onefile`로 단일 실행파일을 만듭니다. Windows에서 실행하면
`dist/meshview.exe`가 생성되며, Python이 설치되지 않은 클린 환경에서 동작합니다.
검증된 onefile 크기는 약 45MB(목표 ≤ 약 100MB)입니다.

## 개발 / 테스트

```bash
pip install -r requirements.txt
pytest                     # GL 컨텍스트가 없으면 render 스모크 테스트는 skip
ruff check meshview tests
```

`io` / `core` / `camera` 패키지는 GPU·윈도우 의존성이 전혀 없어 윈도우 없이
단위 테스트됩니다. GL 스모크 테스트(`tests/test_render_smoke.py`)는 `gl` 마커로
게이팅되어, OpenGL 컨텍스트가 없는 CI에서는 자동으로 skip됩니다.

## 구조

```
meshview/
  io/      확장자 레지스트리 + Abaqus 키워드 파서 (.inp/.blk 공용)
  core/    MeshData, 요소 면 테이블, 외피 표면 추출 (순수 numpy)
  camera/  view/projection 행렬, 아크볼, fit, (state,event)->state 리듀서
  render/  moderngl 렌더러(VBO/IBO), 셰이더, 윈도우/입력 배선
  app/     CLI 진입점 (열기 / --info / 파일 다이얼로그)
```
