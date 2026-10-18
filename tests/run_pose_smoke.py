# 실제 사용자 지정 MP4와 공식 모델로 전체 분석 및 Qt 화면 통합 검증을 실행한다.
# UI 타이머 응답, 프레임/시간 대응, 다인 검출, 마우스 선수 선택, 캐시 추적,
# 오버레이 OFF 원본 동일성과 크기 변경을 확인하고 output에 JSON/PNG를 남긴다.
# 영상별 인식 정확도 및 영구 신원 보장은 이 자동 검증만으로 판단하지 않는다.
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=Path)
parser.add_argument("--offscreen", action="store_true")
args = parser.parse_args()
if args.offscreen:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from app.main import configure_logging
from app.ui.main_window import MainWindow
from app.ui.pose_overlay import image_rect
from app.core.player_tracker import crowded, pick_person


def main():
    configure_logging()
    app = QApplication([])
    video = args.video or next((ROOT / "data" / "videos").glob("*.mp4"))
    window = MainWindow(ROOT / "output" / "pose_smoke_settings.ini")
    window.overlay_check.setChecked(True)
    window.show()
    report = {"video": str(video), "errors": [], "ui_ticks_during_analysis": 0}
    stage, anchor_index, anchor_pose, started = "open", None, None, time.monotonic()
    failed = False

    def error(message):
        report["errors"].append(message)

    window.on_ai_error = error
    window.player.error.disconnect(window.on_error)
    window.player.error.connect(error)
    window.player.request("open", str(video))

    def poll():
        nonlocal stage, anchor_index, anchor_pose, failed
        try:
            if time.monotonic() - started > 240:
                raise TimeoutError("실제 영상 통합 검증 240초 초과")
            if report["errors"]:
                raise RuntimeError(report["errors"][-1])
            if stage == "open" and window.original is not None:
                report["fps"], report["video_frames"] = window.fps, window.count
                window.start_analysis()
                stage = "analysis"
            elif stage == "analysis":
                if window.analysis_worker is not None:
                    report["ui_ticks_during_analysis"] += 1
                    return
                assert window.cache.state == "complete", window.cache.state
                frames = window.cache.frames
                assert len(frames) == window.count
                stamps = [frames[i].timestamp_ms for i in range(window.count)]
                assert all(a < b for a, b in zip(stamps, stamps[1:]))
                assert all(abs(frames[i].timestamp_ms - i * 1000 / window.fps) <= 0.5 for i in frames)
                report["analyzed_frames"] = len(frames)
                report["detected_frames"] = sum(f.detection_success for f in frames.values())
                report["multiple_people_frames"] = sum(len(f.poses) >= 2 for f in frames.values())
                report["multiple_trackable_people_frames"] = sum(sum(p.trackable for p in f.poses) >= 2 for f in frames.values())
                report["max_people"] = max(len(f.poses) for f in frames.values())
                report["synchronized_timestamps"] = True
                for i, frame in frames.items():
                    if sum(p.trackable for p in frame.poses) < 2:
                        continue
                    for pi, pose in enumerate(frame.poses):
                        if not pose.trackable or pose.bbox is None:
                            continue
                        x1, y1, x2, y2 = pose.bbox
                        x, y = (x1 + x2) / 2, (y1 + y2) / 2
                        if pick_person(frame.poses, x, y) == pi and not crowded(pose, frame.poses, pi):
                            anchor_index, anchor_pose = i, pi
                            report["click_normalized"] = [x, y]
                            break
                    if anchor_index is not None:
                        break
                if anchor_index is None:
                    raise AssertionError("실제 영상에 클릭 가능한 다인 프레임이 없습니다.")
                window.player.request("seek", anchor_index)
                stage = "seek"
            elif stage == "seek" and window.index == anchor_index:
                window.begin_selection()
                assert window.video.selecting
                x, y = report["click_normalized"]
                rect = image_rect(window.video.width(), window.video.height(), *window.video.source_size)
                QTest.mouseClick(window.video, Qt.MouseButton.LeftButton,
                                 pos=QPoint(round(rect.x() + x * rect.width()), round(rect.y() + y * rect.height())))
                assert window.cache.anchor == (anchor_index, anchor_pose)
                stage = "track"
            elif stage == "track" and not window.tracking_pending:
                assert window.cache.selections[anchor_index].selected_pose_index == anchor_pose
                report["selected_frame"], report["selected_pose"] = anchor_index, anchor_pose
                counts = {}
                for selected in window.cache.selections.values():
                    counts[selected.state] = counts.get(selected.state, 0) + 1
                report["tracking_states"] = counts
                assert all(s.selected_pose_index is None for s in window.cache.selections.values()
                           if s.state in ("lost", "ambiguous", "reselect_required"))
                window.resize(1000, 700)
                stage = "render"
            elif stage == "render":
                window.render_frame()
                window.grab().save(str(ROOT / "output" / "pose_smoke_overlay.png"))
                annotated = window.video.pixmap().toImage()
                window.overlay_check.setChecked(False)
                expected = window.original.scaled(window.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation).toImage()
                assert window.video.pixmap().toImage() == expected
                assert annotated != expected, "골격 ON 화면이 원본과 동일합니다."
                report["overlay_off_matches_original"] = True
                report["mouse_selection_on_real_multi_person_frame"] = True
                report["source_resolution"] = list(window.video.source_size)
                window.grab().save(str(ROOT / "output" / "pose_smoke_original.png"))
                timer.stop()
                window.close()
        except Exception as exc:
            failed = True
            logging.exception("실제 영상 통합 검증 실패")
            report["errors"].append(str(exc))
            timer.stop()
            window.close()

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(50)
    app.exec()
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["passed"] = not failed and stage == "render" and not report["errors"]
    output = ROOT / "output" / "pose_smoke_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
