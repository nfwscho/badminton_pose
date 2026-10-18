# 실제 Windows GUI/공식 모델/사용자 문제 영상으로 5.1 정상 분석 도달 경로를 검증한다.
# 원본 저장 파일은 읽기만 하고 실행별 격리 저장소에서 레거시 연결, 명시적 테스트 승인,
# 전체 스윙 검출→검토→활성 버튼 클릭→코칭 결과/저장/캐시 재사용을 확인한다.
# 1280×720/1920×1080 화면을 저장한다. 자동 승인은 테스트용이며 실제 선수/기술 정답 검증이 아니다.
import argparse
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=Path, default=ROOT / "data/videos/20260722_171402.mp4")
parser.add_argument("--offscreen", action="store_true")
args = parser.parse_args()
if args.offscreen:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication
from app.main import configure_logging
from app.ui.main_window import MainWindow
from app.analysis.motion_models import make_context
from app.analysis.motion_store import MotionStore
from app.analysis.coach_store import CoachStore
from app.core.player_tracker import pick_person


def main():
    configure_logging()
    app = QApplication([])
    folder = ROOT / "output/readiness_smoke" / str(time.time_ns())
    folder.mkdir(parents=True)
    settings = QSettings(str(folder / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("motion/hand", "right")
    settings.sync()
    w = MainWindow(folder / "settings.ini")
    w.motion.store = MotionStore(folder / "motion_events")
    w.coach.store = CoachStore(folder / "coaching")
    w.resize(1280, 720)
    w.show()
    report = {"video": str(args.video), "errors": [], "ui_ticks": 0, "simulated_user_confirmations": True,
              "real_coaching_accuracy_validated": False, "native_pointer_injection": False, "output": str(folder)}
    failures = report["errors"]
    w.on_ai_error = failures.append
    w.motion.show_error = failures.append
    w.coach.show_error = failures.append
    w.player.error.disconnect(w.on_error)
    w.player.error.connect(failures.append)
    phase, seed, anchor, pose_index, first_report = "open", None, None, None, None
    started = time.monotonic()
    run_id = None
    source_hashes = {}
    def approve_link(text):
        report["link_confirmation_requested"] = True
        return True  # 격리된 테스트 문서에만 적용한다. 실제 사용자 승인으로 기록하지 않는다.
    w.workflow.confirm_link = approve_link
    w.player.request("open", str(args.video))

    def poll():
        nonlocal phase, seed, anchor, pose_index, first_report, run_id
        try:
            if time.monotonic() - started > 180:
                raise TimeoutError("5.1 실제 GUI 통합 검증 180초 초과")
            if failures:
                raise RuntimeError(failures[-1])
            report["ui_ticks"] += 1
            if phase == "open" and w.timeline is not None and w.original is not None:
                report["video_frames"] = w.count
                seed = MotionStore(ROOT / "output/motion_events").load(w.timeline)
                for source_file in (MotionStore(ROOT / "output/motion_events").path_for(w.timeline.identity),
                                    CoachStore(ROOT / "output/coaching").path_for(w.timeline.identity)):
                    if source_file.exists():
                        source_hashes[source_file] = hashlib.sha256(source_file.read_bytes()).hexdigest()
                if seed:
                    w.motion.store.save(seed, w.timeline)
                    report["legacy_seed_events"] = len(seed.events)
                    report["legacy_source_modified"] = seed.modified
                w.workflow.perform("pose")
                phase = "pose"
            elif phase == "pose" and w.analysis_worker is None and w.cache.state == "complete":
                report["pose_frames"] = len(w.cache.frames)
                run_id = w.cache.run_id
                anchor = seed.automatic.source_anchor["frame"] if seed else 0
                frame = w.cache.frames[anchor]
                for i, pose in enumerate(frame.poses):
                    if not pose.trackable:
                        continue
                    context = make_context(w.timeline.identity, anchor, pose, "right", 0)
                    if seed is None or context["athlete_key"] == seed.automatic.athlete_key:
                        x = sum(pose.joints[j].x for j in (11, 12, 23, 24)) / 4
                        y = sum(pose.joints[j].y for j in (11, 12, 23, 24)) / 4
                        if pick_person(frame.poses, x, y) == i:
                            pose_index = i
                            report["selection_xy"] = [x, y]
                            break
                if pose_index is None:
                    raise AssertionError("저장된 기준 선수와 현재 검출을 안전하게 연결할 수 없음 — 자동 정상 처리하지 않음")
                w.player.request("seek", anchor)
                phase = "select"
            elif phase == "select" and w.index == anchor:
                w.workflow.perform("player")
                w.select_person(*report["selection_xy"])
                assert w.cache.anchor == (anchor, pose_index)
                phase = "measure"
            elif phase == "measure" and w.cache.measurements and not w.measurement_pending and not w.tracking_pending:
                report["measurement_valid_frames"] = sum(m.analyzable for m in w.cache.measurements.values())
                if seed:
                    assert w.motion.document is not None and w.motion.historical
                    assert not w.coach.analyze_button.isEnabled()
                    report["historical_block_reproduced"] = True
                    w.workflow.start_check(True)
                    phase = "link"
                else:
                    w.motion.run_analysis()
                    phase = "detect"
            elif phase == "link" and not w.workflow.workers:
                assert not w.workflow.last_error, w.workflow.last_error
                assert not w.motion.historical
                assert w.coach.available()
                # 저장 파일을 불러왔다고 선택/검토를 자동 승인하지 않는다. 다음 안내 작업을 사용한다.
                if not w.coach.selected_event():
                    assert w.workflow.next_action()[0] == "swing"
                    w.workflow.perform("swing")
                assert w.coach.analyze_button.isEnabled()
                report["legacy_link_verified"] = True
                assert [(e.event_id, e.start_frame, e.end_frame, e.phases, e.user_label) for e in w.motion.document.events] == [
                    (e.event_id, e.start_frame, e.end_frame, e.phases, e.user_label) for e in seed.events]
                report["legacy_boundaries_preserved"] = True
                if seed.modified:
                    # 수동 원본은 재검출로 교체하지 않고 해당 범위의 코칭으로 이어간다.
                    report["manual_redetection_skipped"] = True
                    phase = "detect"
                else:
                    w.motion.run_analysis()
                    phase = "detect"
            elif phase == "detect" and not w.motion.workers:
                assert w.motion.document and w.motion.document.events
                assert w.coach.analyze_button.isEnabled(), "스윙 검출 직후 버튼이 자동 활성화되지 않음"
                report["swing_candidates"] = len(w.motion.document.events)
                report["auto_selected_event"] = w.motion.panel.selected_id
                if not w.coach.selected_event():
                    w.workflow.perform("swing")
                w.workflow.perform("review")
                w.workflow.review_button.click()
                phase = "review"
            elif phase == "review" and not w.workflow.workers:
                w.workflow.refresh(force=True)
                assert w.workflow.readiness.can_run
                assert w.workflow.state()["reviewed"]
                assert w.coach.analyze_button.isEnabled()
                report["ready_before_coaching"] = asdict(w.workflow.readiness)
                w.coach.analyze_button.click()  # 활성 UI 버튼의 정상 clicked 경로를 사용한다.
                phase = "coaching"
            elif phase == "coaching" and not w.coach.workers:
                first_report = w.coach.panel.report
                assert first_report is not None
                assert w.coach.panel.list.count() > 0
                assert not w.coach.is_stale(first_report)
                report["coach_state"] = first_report.quality.state
                report["coach_messages"] = len(first_report.messages)
                report["quality"] = asdict(first_report.quality)
                w.coach.export(folder / "results.json")
                w.coach.export(folder / "results.csv")
                w.coach.panel_check.setChecked(True)
                w.motion.panel_check.setChecked(False)
                w.panel_check.setChecked(False)
                w.resize(1280, 720)
                phase = "small"
            elif phase == "small":
                assert w.width() <= 1280 and w.height() <= 720, (w.width(), w.height())
                assert w.workflow.conditions_button.isVisible()
                report["small_window"] = [w.width(), w.height()]
                report["small_video"] = [w.video.width(), w.video.height()]
                w.grab().save(str(folder / "1280x720.png"))
                w.resize(1920, 1080)
                phase = "large"
            elif phase == "large":
                assert w.width() <= 1920 and w.height() <= 1080
                report["large_window"] = [w.width(), w.height()]
                w.panel_check.setChecked(True)
                w.motion.panel_check.setChecked(True)
                message = w.coach.panel.selected_message()
                assert message is not None
                report["evidence_target"] = message.start_frame
                bounds = [w.range_a, w.range_b]
                w.coach.show_evidence(message)
                assert [w.range_a, w.range_b] == bounds
                phase = "evidence"
            elif phase == "evidence" and w.index == report["evidence_target"]:
                import numpy as np
                np.testing.assert_allclose(w.measurement_panel.plot.viewRange()[0], w.motion.panel.timeline_plot.viewRange()[0], atol=1e-6)
                assert abs(w.measurement_panel.cursor.value() - w.timeline.seconds[w.index]) < 1e-9
                assert abs(w.motion.panel.cursor.value() - w.timeline.seconds[w.index]) < 1e-9
                report["evidence_pts_and_graph_sync"] = True
                w.grab().save(str(folder / "1920x1080.png"))
                w.motion.use_event_range()
                assert w.coach.analyze_button.isEnabled()
                assert not w.coach.is_stale(first_report)
                w.coach.analyze_button.click()
                phase = "reuse"
            elif phase == "reuse" and not w.coach.workers:
                assert w.cache.run_id == run_id
                assert w.coach.panel.report.input_key == first_report.input_key
                assert w.coach.panel.report.generated_at == first_report.generated_at
                report["pose_not_reexecuted"] = True
                report["coaching_cache_reused_after_ab"] = True
                w.workflow.inspect()
                phase = "diagnostic"
            elif phase == "diagnostic" and not w.workflow.workers:
                assert w.workflow.previews
                w.workflow.dialog.grab().save(str(folder / "diagnostic.png"))
                report["diagnostic_rule_checks"] = len(w.workflow.previews[first_report.event_id].messages)
                report["final_steps"] = w.workflow.readiness.steps
                assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in source_hashes.items())
                report["user_source_files_unchanged"] = True
                report["passed"] = True
                timer.stop()
                w.close()
        except Exception as exc:
            import logging
            logging.exception("5.1 실제 GUI 통합 검증 실패")
            failures.append(str(exc))
            report["failed_phase"] = phase
            report["passed"] = False
            timer.stop()
            w.close()
    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(50)
    app.exec()
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    path = ROOT / "output/readiness_smoke_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report.get("passed") and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
