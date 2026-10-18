# Badminton AI Coach — 동영상 플레이어

Windows 10 / Python 3.12 / PySide6 기반 첫 번째 구현입니다.
AI 관절 인식과 코칭은 아직 구현하지 않았습니다.

## 실행 (PowerShell, 프로젝트 폴더)

```powershell
.\.venv\Scripts\python.exe -m app.main
```

직접 실행도 가능합니다: `.\.venv\Scripts\python.exe app\main.py`.
현재 프로젝트의 `.runtime`에는 Python 3.12.15가 있으며 `.venv`가 이를 사용합니다.
두 폴더를 이동하거나 삭제하면 가상환경을 다시 생성해야 합니다.
다른 PC에서 설치된 Python 3.12를 사용하는 경우:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 조작

- MP4 동영상 불러오기: 버튼 또는 Ctrl+O
- 재생 / 일시정지: 버튼 또는 Space
- 이전 / 다음 프레임: 버튼 또는 좌우 방향키 (이동하면 일시정지)
- 재생 위치: 슬라이더 클릭 또는 드래그 후 놓기 (이동하면 일시정지)
- 영상 끝에서는 일시정지하며 재생을 누르면 처음부터 시작합니다.
- 영상은 비율을 유지해 표시합니다. 현재 시간과 전체 시간은 밀리초까지 표시합니다.

디코딩은 별도 작업 스레드에서 수행합니다. 이번 버전은 영상만 재생하며
오디오는 지원하지 않습니다. OpenCV가 지원하는 코덱의 MP4를 사용하세요.
프레임 시간은 파일 FPS 기준이며 가변 프레임률 영상은 실제 타임스탬프와 차이가 날 수 있습니다.

## 로그 및 설정

- `logs/app.log`: 실행, 열기, 재생, 탐색, 종료 및 오류 추적을 누적 기록합니다.
- `output/settings.ini`: 창 크기/위치/상태와 마지막 파일 선택 폴더를 저장합니다.
- 기본 창 크기: 1100×760, 최소 크기: 640×420.
- 마지막 영상을 자동으로 열거나 재생하지 않습니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 임시 MP4와 오프스크린 Qt를 사용하여 실제 디코딩, 재생/일시정지,
프레임 탐색/경계, 파일 오류 후 복구, 설정 복원을 확인합니다.
실제 디스플레이의 외관 및 사용자 보유 영상별 코덱 호환성은 별도 확인이 필요합니다.

기존 `app/core/frame_processor.py`, `app/core/pose_detector.py`는 후속 구현을 위해 그대로 유지합니다.
