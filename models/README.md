# Pose Landmarker 모델

- 배포자: Google MediaPipe
- 모델: Pose Landmarker **Full**, float16, 모델 배포 버전 **1**
- 파일: `pose_landmarker_full.task` (9,398,198 bytes)
- SHA-256: `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`
- 다운로드/검증일: 2026-10-19 (한국 시간)
- 공식 다운로드: <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task>
- 공식 설명/모델 카드 링크: <https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models>
- 공식 Python API: <https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python>
- 검증한 런타임: Python 3.12.15, MediaPipe 1.1.1, PySide6 6.10.2,
  OpenCV contrib 4.13.0.92, NumPy 2.5.4 (Windows)

재다운로드 (프로젝트 폴더 PowerShell):

```powershell
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task' -OutFile models/pose_landmarker_full.task
Get-FileHash models/pose_landmarker_full.task -Algorithm SHA256
```

애플리케이션 실행 시 다운로드하지 않습니다. 모델이 없거나 손상되면 AI 오류를
안내하고 기존 플레이어는 유지합니다. 모델 파일은 원본을 변경하지 않고 사용합니다.
재배포 시 공식 모델 카드의 이용 조건을 확인하세요.
