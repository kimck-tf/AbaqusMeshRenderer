# Windows 단일 실행 파일 빌드 설계

## 목적

기존 `meshview` Python 애플리케이션을 Python 설치 없이 실행할 수 있는 Windows 단일 실행 파일 `dist/meshview.exe`로 패키징한다. 일반 사용자가 실행할 때 콘솔 창은 표시하지 않는다.

## 범위

- 기존 `build.py`의 PyInstaller `--onefile` 구성을 유지한다.
- 기본 빌드에 콘솔을 숨기는 PyInstaller 옵션을 적용한다.
- `--debug` 빌드는 진단 로그를 볼 수 있도록 콘솔을 유지한다.
- 실행 파일 이름은 기존과 동일하게 `meshview.exe`로 유지한다.
- 설치 프로그램, 바탕화면 바로가기, 아이콘 제작, 자동 업데이트는 포함하지 않는다.

## 구현 방식

`build.py`가 PyInstaller 인자를 조립할 때 일반 빌드에는 `--windowed`를 추가한다. `--debug`가 지정된 경우에는 `--windowed`를 추가하지 않아 기존처럼 콘솔과 상세 로그를 사용할 수 있게 한다.

별도 `.spec` 파일이나 후처리 도구를 도입하지 않는다. 현재 빌드 스크립트의 단일 진입점과 재현 가능한 CI 구성을 그대로 활용하는 것이 변경 범위와 유지보수 비용이 가장 작다.

## 빌드 및 산출물

기본 빌드 명령은 다음과 같다.

```powershell
python build.py
```

성공 시 다음 파일이 생성된다.

```text
dist/meshview.exe
```

진단용 빌드는 다음과 같이 실행한다.

```powershell
python build.py --debug
```

## 검증 기준

1. 전체 자동화 테스트가 통과한다.
2. Ruff 정적 검사가 통과한다.
3. `python build.py`가 오류 없이 완료되고 `dist/meshview.exe`가 생성된다.
4. 실행 파일 크기가 프로젝트 목표인 약 100MB 이하인지 확인한다.
5. 실행 파일에 `--info tests/fixtures/single_hex.inp`를 전달했을 때 메쉬 통계가 정상 출력되고 종료 코드가 0이다.
6. 테스트 모델을 GUI로 열었을 때 애플리케이션 프로세스가 정상 기동한다.

## 오류 처리

빌드 의존성이 없거나 PyInstaller가 실패하면 기존 `build.py`의 오류 반환 방식을 유지한다. GUI 릴리스에서 콘솔이 숨겨지므로 문제 진단이 필요한 경우 `python build.py --debug`로 만든 실행 파일을 사용한다.

## 변경 제한

이번 작업에서는 패키징에 직접 필요한 `build.py`와 관련 테스트·문서만 수정한다. 렌더러, 파서, 카메라 및 UI 동작은 변경하지 않는다.
