# 실제 관절 측정 캐시 모양의 정답 포함 합성 시계열로 로컬 코칭 계약을 검사한다.
# 품질/신원/단계 보류, 누락값 비생성, 좌우 손, 근거 PTS, 입력·규칙 캐시 무효화,
# 반복 비교 제한, JSON/CSV 및 사용자 검토 보존을 검증한다. 실제 기술 평가 정확도와 구분한다.
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_motion import synthetic_case, analyze
from app.analysis.coach_engine import analyze_coaching
from app.analysis.coach_models import CoachReview, ENGINE_VERSION, event_key, review_key
from app.analysis.coach_rules import DEFAULT_RULES, load_rules
from app.analysis.coach_store import CoachStore, export_reports, report_from_dict
from app.analysis.joint_angles import MetricValue
from app.analysis.motion_models import MotionDocument, phase_span, normalize_phases


class CoachTests(unittest.TestCase):
    def setUp(self):
        self.case = synthetic_case((1.0, 3.4))
        result, _ = analyze(self.case)
        self.doc = MotionDocument(result, result.events)
        self.rules = load_rules()

    def run_engine(self, doc=None, confirmations=None, **kwargs):
        frames, selections, measurements, tl, _, _ = self.case
        return analyze_coaching(doc or self.doc, frames, selections, measurements, tl, self.rules,
                                confirmations=confirmations, **kwargs)

    def confirmed(self, doc=None):
        return {e.event_id: event_key(e) for e in (doc or self.doc).events}

    def test_actual_values_and_provenance(self):
        report = self.run_engine(confirmations=self.confirmed())[0]
        self.assertEqual(report.quality.state, "MEASUREMENT_ONLY")
        self.assertEqual(report.engine_version, ENGINE_VERSION)
        self.assertEqual(len(report.messages), 7)
        o = report.features["FORWARD_SWING.elbow_change"]
        raw = self.case[2]
        self.assertAlmostEqual(o.value, raw[o.end_frame].values["right_elbow"].raw - raw[o.start_frame].values["right_elbow"].raw)
        self.assertTrue(o.valid)
        self.assertEqual(o.unit, "도")
        for message in report.messages:
            self.assertEqual(message.start_pts, self.case[3].raw_pts_seconds[message.start_frame])
            self.assertEqual(message.end_pts, self.case[3].raw_pts_seconds[message.end_frame])
            self.assertEqual(message.player_id, self.doc.events[0].player_id)
        self.assertFalse(report.features["RECOVERY.next_ready_time"].valid)

    def test_user_confirmation_required_no_defect_claim(self):
        report = self.run_engine()[0]
        self.assertEqual(report.quality.state, "COACHING_DEFERRED")
        self.assertTrue(all(m.status == "COACHING_DEFERRED" for m in report.messages))
        self.assertTrue(any(o.valid for o in report.features.values()))
        self.assertTrue(all(m.level != "INFERRED" for m in report.messages))

    def test_unknown_stage_not_created(self):
        event = self.doc.events[0]
        event = replace(event, kind="UNKNOWN", phases=normalize_phases([], event.start_frame, event.end_frame, self.case[3]))
        doc = replace(self.doc, events=(event,))
        report = self.run_engine(doc, self.confirmed(doc))[0]
        self.assertEqual(report.quality.state, "COACHING_DEFERRED")
        self.assertFalse(any(key.startswith("FORWARD_SWING") for key in report.features))
        self.assertFalse(next(m for m in report.messages if m.rule_id == "forward_extension").measurements)

    def test_missing_tracking_deferred_no_gap_change(self):
        i = 72
        self.case[1][i] = replace(self.case[1][i], state="lost", selected_pose_index=None)
        report = self.run_engine(confirmations=self.confirmed())[0]
        self.assertEqual(report.quality.state, "COACHING_DEFERRED")
        self.assertFalse(report.features["FORWARD_SWING.elbow_change"].valid)
        self.assertIsNone(report.features["FORWARD_SWING.elbow_change"].value)

    def test_foreign_player_never_generates_angle_observations(self):
        i = 72
        self.case[2][i] = replace(self.case[2][i], player_id=88)
        report = self.run_engine(confirmations=self.confirmed())[0]
        self.assertFalse(report.quality.integrity_ok)
        self.assertTrue(all(not o.valid for name, o in report.features.items() if ".elbow_" in name))
        self.assertTrue(all(m.status == "COACHING_DEFERRED" for m in report.messages))

    def test_foreign_event_document_rejected(self):
        doc = replace(self.doc, events=(replace(self.doc.events[0], athlete_key="foreign"),))
        with self.assertRaises(ValueError):
            self.run_engine(doc)

    def test_missing_metric_does_not_become_zero(self):
        for i, m in list(self.case[2].items()):
            self.case[2][i] = replace(m, values={**m.values, "trunk_tilt": MetricValue(None, False, "occluded")})
        report = self.run_engine(confirmations=self.confirmed())[0]
        self.assertFalse(report.features["SWING.trunk_change"].valid)
        self.assertIsNone(report.features["SWING.trunk_change"].value)
        self.assertEqual(next(m for m in report.messages if m.rule_id == "trunk_change").status, "COACHING_DEFERRED")

    def test_nan_raw_deferred_not_invented(self):
        i = 72
        m = self.case[2][i]
        self.case[2][i] = replace(m, values={**m.values, "right_elbow": MetricValue(float("nan"), True)})
        report = self.run_engine()[0]
        self.assertEqual(report.quality.state, "COACHING_DEFERRED")
        self.assertTrue(all(o.value is None or __import__("math").isfinite(o.value) for o in report.features.values()))

    def test_left_hand_uses_left_cache(self):
        self.case = synthetic_case(hand="left")
        result, _ = analyze(self.case, "left")
        self.doc = MotionDocument(result, result.events)
        report = self.run_engine(confirmations=self.confirmed())[0]
        self.assertTrue(report.features["FORWARD_SWING.elbow_change"].value > 0)
        self.assertTrue(all("left_elbow" in m.graph_metrics for m in report.messages))

    def test_pts_and_source_mismatch_deferred(self):
        i = 72
        self.case[2][i] = replace(self.case[2][i], timestamp_seconds=10)
        report = self.run_engine()[0]
        self.assertFalse(report.quality.integrity_ok)
        self.assertTrue(any("시간" in r for r in report.quality.reasons))

    def test_cache_content_and_rule_version_invalidation(self):
        first = self.run_engine()
        with patch("app.analysis.coach_engine.extract_features", side_effect=AssertionError("캐시 재사용 시 특징 재계산 금지")):
            reused = self.run_engine(cache={r.event_id: r for r in first})
        self.assertIs(reused[0], first[0])
        i = 72
        m = self.case[2][i]
        self.case[2][i] = replace(m, values={**m.values, "right_elbow": MetricValue(10, True)})
        changed = self.run_engine(cache={r.event_id: r for r in first})
        self.assertNotEqual(first[0].input_key, changed[0].input_key)
        with tempfile.TemporaryDirectory() as directory:
            data = json.loads(DEFAULT_RULES.read_text(encoding="utf-8"))
            data["version"] = "test-2"
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.rules = load_rules(path)
        revised = self.run_engine(cache={r.event_id: r for r in changed})
        self.assertNotEqual(changed[0].input_key, revised[0].input_key)

    def test_event_phase_edit_invalidates_confirmation_and_key(self):
        first = self.run_engine(confirmations=self.confirmed())[0]
        event = self.doc.events[0]
        edited = replace(event, user_label="수동 확인 전 새 라벨", manually_modified=True, source="manual")
        updated = self.run_engine(replace(self.doc, events=(edited,)), self.confirmed())[0]
        self.assertNotEqual(first.input_key, updated.input_key)
        self.assertFalse(updated.quality.user_confirmed)

    def test_repeat_comparison_requires_conditions_and_differences_only(self):
        reports = self.run_engine(confirmations=self.confirmed())
        self.assertTrue(reports[0].comparison["reasons"])
        reports = self.run_engine(confirmations=self.confirmed(), comparison_confirmed=True, comparison_group="테스트 동일 기술")
        self.assertFalse(reports[0].comparison["reasons"])
        difference = reports[0].comparison["differences"]["SWING.duration"]
        self.assertTrue(difference["valid"])
        self.assertAlmostEqual(difference["difference"], self.doc.events[0].duration - self.doc.events[1].duration, places=6)
        self.assertFalse(any("score" in key or "similarity" in key for key in reports[0].comparison))

    def test_different_hand_or_kind_comparison_refused(self):
        event = self.doc.events[1]
        doc = replace(self.doc, events=(self.doc.events[0], replace(event, kind="OTHER_SWING")))
        reports = self.run_engine(doc, self.confirmed(doc), comparison_confirmed=True, comparison_group="기술")
        self.assertTrue(all(r.comparison["reasons"] for r in reports))

    def test_store_export_and_review_roundtrip(self):
        reports = self.run_engine(confirmations=self.confirmed())
        review = CoachReview("error", "=메모", "가림 오인", True, reports[0].input_key, "2026-10-19")
        reviews = {review_key(reports[0].event_id, reports[0].messages[0].rule_id): review}
        with tempfile.TemporaryDirectory() as directory:
            store = CoachStore(directory)
            mapped = {r.event_id: r for r in reports}
            store.save(self.case[3], mapped, reviews, self.confirmed(), [])
            loaded, restored, confirmations, _ = store.load(self.case[3])
            self.assertEqual(mapped, loaded)
            self.assertEqual(reviews, restored)
            self.assertEqual(confirmations, self.confirmed())
            export_reports(Path(directory) / "out.json", reports, reviews, {reports[0].event_id})
            export_reports(Path(directory) / "out.csv", reports, reviews)
            exported = json.loads((Path(directory) / "out.json").read_text(encoding="utf-8"))
            self.assertTrue(exported["reports"][0]["stale"])
            with (Path(directory) / "out.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), sum(len(r.messages) for r in reports))
            self.assertIn("가림 오인", rows[0]["review_json"])

    def test_corrupt_evidence_and_other_video_refused(self):
        report = self.run_engine()[0]
        data = asdict(report)
        data["messages"][0]["start_frame"] = 299
        with self.assertRaises(ValueError):
            report_from_dict(data, self.case[3])
        with self.assertRaises(ValueError):
            report_from_dict(asdict(report), replace(self.case[3], identity=("other.mp4", 1, 2)))

    def test_rule_config_validation_and_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            data = json.loads(DEFAULT_RULES.read_text(encoding="utf-8"))
            path = Path(directory) / "rules.json"
            data["rules"][0]["enabled"] = False
            path.write_text(json.dumps(data), encoding="utf-8")
            self.rules = load_rules(path)
            self.assertFalse(any(m.rule_id == "backswing_elbow" for m in self.run_engine()[0].messages))
            data["rules"][0]["validated_standard"] = True
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rules(path)

    def test_cancel_no_partial_report(self):
        self.assertIsNone(self.run_engine(cancelled=lambda: True))

    def test_phase_boundary_edit_requires_new_confirmation(self):
        first = self.run_engine(confirmations=self.confirmed())[0]
        event = self.doc.events[0]
        ready, back, *remaining = event.phases
        tl = self.case[3]
        phases = (phase_span(ready.phase, ready.start_frame, ready.end_frame - 1, tl, "수동 경계 검토"),
                  phase_span(back.phase, back.start_frame - 1, back.end_frame, tl, "수동 경계 검토"), *remaining)
        edited = replace(event, phases=phases, source="manual", manually_modified=True)
        doc = replace(self.doc, events=(edited, self.doc.events[1]))
        result = self.run_engine(doc, self.confirmed())[0]
        self.assertFalse(result.quality.user_confirmed)
        self.assertNotEqual(first.input_key, result.input_key)

    def test_foreign_track_segment_and_pose_time_deferred(self):
        i = 72
        self.case[2][i] = replace(self.case[2][i], track_segment="7:foreign")
        report = self.run_engine()[0]
        self.assertFalse(report.quality.integrity_ok)
        self.case = synthetic_case((1.0, 3.4))
        self.case[0][i] = replace(self.case[0][i], timestamp_seconds=20)
        report = self.run_engine()[0]
        self.assertFalse(report.quality.integrity_ok)


if __name__ == "__main__":
    unittest.main()
