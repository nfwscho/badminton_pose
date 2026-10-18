# 정답 구간이 있는 합성 33관절/각도/원본 PTS 시계열로 스윙 규칙을 검증한다.
# 정지/노이즈/단일·복수 스윙/좌우 손/신원·시간 단절, 단계 순서 및 A/B 잘림,
# 결정적 ID, 수동 편집·원본 복원·JSON 이력/소스 검증을 검사한다.
# 합성 데이터 통과는 실제 배드민턴 영상의 동작 인식 정확도 검증을 뜻하지 않는다.
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from app.analysis.joint_angles import FrameMeasurement, METRICS, MetricValue
from app.analysis.motion_detector import detect_motion, manual_event
from app.analysis.motion_features import extract_features, temporal_derivative
from app.analysis.motion_models import (MotionConfig, MotionDocument, make_context, normalize_phases,
                                       phase_span, validate_event)
from app.analysis.motion_store import MotionStore
from app.core.player_tracker import FrameSelection
from app.core.pose_detector import FrameResult, Joint, LANDMARK_NAMES, PersonPose
from app.core.video_timeline import VideoTimeline


def synthetic_case(swings=(1.0,), hand="right", duration=6.0, fps=50):
    n, width, height, player = int(duration * fps), 1000, 800, 7
    times = tuple(i / fps for i in range(n))
    tl = VideoTimeline(times, times, ("pts",) * n, duration, width, height, ("synthetic.mp4", 1234, 999))
    frames, selections, measurements = {}, {}, {}
    truth = []
    for onset in swings:
        truth.append((round(onset * fps), round((onset + 1.2) * fps)))
    for i, seconds in enumerate(times):
        rel, elbow, shoulder = np.array((-0.25, 0.6)), 140.0, 35.0
        for onset in swings:
            t = seconds - onset
            knots = ((0, (-0.25, 0.6), 140, 35), (0.35, (-0.65, -1.0), 70, 135),
                     (0.60, (0.95, -0.3), 175, 100), (0.95, (0.25, 0.65), 150, 40),
                     (1.20, (-0.25, 0.6), 140, 35))
            for a, b in zip(knots, knots[1:]):
                if a[0] <= t <= b[0]:
                    weight = (1 - np.cos(np.pi * (t - a[0]) / (b[0] - a[0]))) / 2
                    rel = np.array(a[1]) * (1 - weight) + np.array(b[1]) * weight
                    elbow, shoulder = a[2] * (1 - weight) + b[2] * weight, a[3] * (1 - weight) + b[3] * weight
        points = {11: (450, 320), 12: (550, 320), 23: (450, 520), 24: (550, 520),
                  13: (400, 400), 14: (600, 400), 15: (400, 440), 16: (600, 440)}
        wrist_id = 15 if hand == "left" else 16
        points[wrist_id] = tuple(np.array((500, 320)) + rel * 200)
        pose = PersonPose(tuple(Joint(j, name, points.get(j, (500, 300))[0] / width,
                                      points.get(j, (500, 300))[1] / height, 0, 1, 1)
                                for j, name in enumerate(LANDMARK_NAMES)))
        frames[i] = FrameResult(i, round(seconds * 1000), (pose,), True, "detected", "pts", seconds)
        selections[i] = FrameSelection(i, 0, "tracked", player)
        values = {name: MetricValue(150.0, True, display=150.0) for name in METRICS}
        values[f"{hand}_elbow"] = MetricValue(float(elbow), True, display=float(elbow))
        values[f"{hand}_shoulder"] = MetricValue(float(shoulder), True, display=float(shoulder))
        values[f"{hand}_wrist_height"] = MetricValue(float(-rel[1]), True, display=float(-rel[1]))
        other = "left" if hand == "right" else "right"
        values[f"{other}_wrist_height"] = MetricValue(-0.6, True, display=-0.6)
        values["trunk_tilt"] = MetricValue(0.0, True, display=0.0)
        measurements[i] = FrameMeasurement(i, seconds, "pts", player, "7:0", "tracked", values)
    context = make_context(tl.identity, 0, frames[0].poses[0], hand, player)
    return frames, selections, measurements, tl, context, truth


def analyze(case, hand="right", config=None, scope=None):
    frames, selections, measurements, tl, context, _ = case
    cfg = config or MotionConfig()
    features = extract_features(frames, selections, measurements, tl, tl.width, tl.height, hand, cfg)
    result = detect_motion(features, measurements, tl, context, hand, cfg, scope)
    return result, features


class MotionTests(unittest.TestCase):
    def test_static_and_small_jitter_no_swing(self):
        case = synthetic_case(())
        result, features = analyze(case)
        self.assertFalse(result.events)
        self.assertTrue(all(c.kind == "NO_SWING" for c in result.coverage))
        frames = case[0]
        for i, frame in list(frames.items()):
            joints = list(frame.poses[0].joints)
            joints[16] = replace(joints[16], x=joints[16].x + np.sin(i * 1.5) * 0.001)
            frames[i] = replace(frame, poses=(PersonPose(tuple(joints)),))
        self.assertFalse(analyze(case)[0].events)

    def test_single_swing_overlaps_ground_truth_and_is_provisional(self):
        case = synthetic_case()
        result, features = analyze(case)
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.kind, "OVERHEAD_CANDIDATE")
        a, b = case[-1][0]
        overlap = min(b, event.end_frame) - max(a, event.start_frame) + 1
        self.assertGreater(overlap / (b - a + 1), 0.8)
        self.assertTrue(event.uncertainty)
        self.assertGreater(event.indicators["peak_wrist_speed_torso_per_second"], 1.4)
        self.assertGreater(event.details["wrist_path_torso_lengths"], 0)
        validate_event(event, case[3])

    def test_two_independent_events_and_deterministic_ids(self):
        case = synthetic_case((1.0, 3.4))
        first, _ = analyze(case)
        second, _ = analyze(case)
        self.assertEqual(len(first.events), 2)
        self.assertLess(first.events[0].end_frame, first.events[1].start_frame)
        self.assertEqual([e.event_id for e in first.events], [e.event_id for e in second.events])
        self.assertEqual(first.analysis_key, second.analysis_key)

    def test_left_and_right_hand_are_independent(self):
        case = synthetic_case(hand="left")
        self.assertEqual(len(analyze(case, "left")[0].events), 1)
        self.assertFalse(analyze(case, "right")[0].events)

    def test_missing_tracking_never_bridges_event(self):
        case = synthetic_case()
        for i in range(70, 80):
            case[1][i] = FrameSelection(i, None, "lost", 7, 70, "not_detected")
        result, features = analyze(case)
        self.assertTrue(np.isnan(features.speed[70:80]).all())
        self.assertTrue(any(interval.kind == "UNKNOWN" for interval in result.coverage))
        self.assertFalse(any(event.start_frame < 70 and event.end_frame >= 80 for event in result.events))

    def test_candidate_order_and_foreign_player_do_not_mix(self):
        case = synthetic_case()
        baseline, _ = analyze(case)
        for i, frame in list(case[0].items()):
            other = synthetic_case((), duration=0.02)[0][0].poses[0]
            case[0][i] = replace(frame, poses=(other, frame.poses[0]))
            case[1][i] = replace(case[1][i], selected_pose_index=1)
        self.assertEqual([e.peak_frame for e in baseline.events], [e.peak_frame for e in analyze(case)[0].events])
        for i in range(len(case[0])):
            case[1][i] = replace(case[1][i], player_id=8)
            case[2][i] = replace(case[2][i], player_id=8, track_segment="8:0")
        self.assertFalse(analyze(case)[0].events)

    def test_time_derivative_uses_pts_and_does_not_cross_bad_time(self):
        times = np.array([0, 0.02, 0.10, 0.10, 0.08, 0.8, 0.84])
        values = times * 2
        derivative = temporal_derivative(values, times, np.zeros(len(times)), 0.15)
        np.testing.assert_allclose(derivative[:3], 2)
        self.assertTrue(np.isnan(derivative[3:5]).all())
        np.testing.assert_allclose(derivative[5:], 2)

    def test_phase_order_continuity_and_partial_observation(self):
        case = synthetic_case()
        result, _ = analyze(case)
        event = result.events[0]
        self.assertEqual(event.phases[0].start_frame, event.start_frame)
        self.assertEqual(event.phases[-1].end_frame, event.end_frame)
        self.assertTrue(all(a.end_frame + 1 == b.start_frame for a, b in zip(event.phases, event.phases[1:])))
        self.assertIn("BACKSWING", [p.phase for p in event.phases])
        self.assertIn("FORWARD_SWING", [p.phase for p in event.phases])
        clipped, _ = analyze(case, scope=(75, 90))
        self.assertTrue(clipped.events)
        self.assertTrue(all(e.clipped for e in clipped.events))
        self.assertNotIn("READY", [p.phase for p in clipped.events[0].phases])

    def test_manual_edit_validation_original_restore_and_json_history(self):
        case = synthetic_case()
        result, features = analyze(case)
        doc = MotionDocument(result, result.events)
        event = doc.events[0]
        edited = manual_event(doc, case[3], event.start_frame, event.end_frame, "UNKNOWN", (), "사용자 확인",
                              event, features, case[2])
        self.assertFalse(doc.modified)
        changed = replace(doc, events=(edited,), operations=({"action": "edit"},))
        self.assertTrue(changed.modified)
        self.assertEqual(doc.automatic.events[0].kind, "OVERHEAD_CANDIDATE")
        with tempfile.TemporaryDirectory() as directory:
            store = MotionStore(directory)
            store.save(doc, case[3])
            store.save(changed, case[3])
            restored = store.load(case[3])
            self.assertEqual(restored.events, changed.events)
            self.assertTrue(restored.modified)
            import json
            self.assertEqual(len(json.loads(store.path_for(case[3].identity).read_text(encoding="utf-8"))["history"]), 1)
        wrong = replace(case[3], identity=("different.mp4", 1234, 999))
        from app.analysis.motion_store import document_from_dict
        from dataclasses import asdict
        with self.assertRaises(ValueError):
            document_from_dict(asdict(changed), wrong)
        with self.assertRaises(ValueError):
            manual_event(doc, case[3], 20, 10, "UNKNOWN", ())

    def test_manual_overlap_and_phase_order_rejected(self):
        case = synthetic_case((1.0, 3.4))
        result, features = analyze(case)
        doc = MotionDocument(result, result.events)
        with self.assertRaises(ValueError):
            manual_event(doc, case[3], result.events[0].start_frame, result.events[1].end_frame, "UNKNOWN", ())
        a, b = result.events[0].start_frame, result.events[0].end_frame
        with self.assertRaises(ValueError):
            normalize_phases((phase_span("RECOVERY", a, a + 1, case[3], "manual"),
                              phase_span("BACKSWING", a + 2, b, case[3], "manual")), a, b, case[3])

    def test_isolated_spike_is_rejected_without_changing_landmarks(self):
        case = synthetic_case(())
        frame = case[0][50]
        joints = list(frame.poses[0].joints)
        joints[16] = replace(joints[16], x=joints[16].x + 0.4)
        case[0][50] = replace(frame, poses=(PersonPose(tuple(joints)),))
        before = case[0][50].poses[0].joints
        result, features = analyze(case)
        self.assertFalse(features.valid[50])
        self.assertEqual(features.reasons[50], "coordinate_spike")
        self.assertFalse(result.events)
        self.assertEqual(case[0][50].poses[0].joints, before)

    def test_large_gap_and_duplicate_pts_split_features(self):
        case = list(synthetic_case())
        times = list(case[3].seconds)
        times[90], times[100] = times[89], times[99] + 0.5
        case[3] = replace(case[3], seconds=tuple(times))
        result, features = analyze(case)
        self.assertFalse(features.valid[90])
        self.assertFalse(features.valid[100])
        self.assertTrue(np.isnan(features.speed[90]))
        self.assertNotEqual(features.runs[89], features.runs[91])
        self.assertFalse(any(e.start_frame < 90 < e.end_frame for e in result.events))

    def test_config_minimum_duration_and_insufficient_scope(self):
        case = synthetic_case()
        self.assertFalse(analyze(case, config=MotionConfig(min_duration=3))[0].events)
        result, _ = analyze(case, scope=(0, 1))
        self.assertFalse(result.events)
        self.assertTrue(all(c.kind == "UNKNOWN" for c in result.coverage))
        with self.assertRaises(ValueError):
            MotionConfig(window=2)

    def test_other_and_unknown_candidates_do_not_force_overhead_phases(self):
        for height_value, expected in ((-0.6, "OTHER_SWING"), (0.0, "UNKNOWN")):
            case = synthetic_case()
            for i, frame in list(case[0].items()):
                joints = list(frame.poses[0].joints)
                joints[16] = replace(joints[16], y=(320 - height_value * 200) / 800)
                case[0][i] = replace(frame, poses=(PersonPose(tuple(joints)),))
                values = dict(case[2][i].values)
                values["right_wrist_height"] = replace(values["right_wrist_height"], raw=height_value)
                case[2][i] = replace(case[2][i], values=values)
            result, _ = analyze(case)
            self.assertTrue(result.events)
            self.assertTrue(all(event.kind == expected for event in result.events))
            self.assertTrue(all(phase.phase == "UNKNOWN_PHASE" for event in result.events for phase in event.phases))

    def test_analysis_ignores_display_smoothing_values(self):
        case = synthetic_case()
        original, _ = analyze(case)
        for i, measurement in list(case[2].items()):
            case[2][i] = replace(measurement, values={name: replace(value, display=9999.0) for name, value in measurement.values.items()})
        changed, _ = analyze(case)
        self.assertEqual(original.events, changed.events)
        none_config = MotionConfig(smoothing="none")
        _, features = analyze(case, config=none_config)
        np.testing.assert_allclose(features.wrist_raw, features.wrist_smooth, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
