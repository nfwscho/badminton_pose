# JSON으로 관리하는 로컬 관찰 규칙 명세와 엄격한 설정 로더.
# 실행 코드는 허용된 규칙 ID로만 선택하며 설정의 임의 코드/eval을 실행하지 않는다.
# 각 규칙은 단계/특징/관절/품질/조건/보류 기준과 설명 템플릿·버전을 명시한다.
# 임계값은 관측 충족 조건이며 검증된 표준 자세 기준이나 정확도 확률이 아니다.
from dataclasses import dataclass
import json
from pathlib import Path
from app.analysis.motion_models import fingerprint

RULE_IDS = ("backswing_elbow", "forward_extension", "peak_timing", "follow_motion",
            "trunk_change", "repeat_consistency", "quality_notice")
DEFAULT_RULES = Path(__file__).resolve().parents[2] / "config" / "coach_rules.json"


@dataclass(frozen=True)
class RuleSet:
    version: str
    key: str
    rules: tuple[dict, ...]


def load_rules(path=DEFAULT_RULES):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("version"), str) or not data["version"]:
        raise ValueError("코칭 규칙 설정 버전이 잘못되었습니다.")
    rules = data.get("rules", [])
    if len(rules) != len(RULE_IDS) or {r.get("id") for r in rules} != set(RULE_IDS):
        raise ValueError("코칭 규칙 ID가 누락되거나 중복되었습니다.")
    for rule in rules:
        for field in ("name", "version", "condition", "observation_method", "interpretation", "suggestion", "defer_when"):
            if not isinstance(rule.get(field), str) or not rule[field].strip():
                raise ValueError("코칭 규칙 설명 필드가 잘못되었습니다: " + field)
        if type(rule.get("enabled")) is not bool or rule.get("validated_standard") is not False:
            raise ValueError("V1은 검증되지 않은 관찰 규칙만 허용합니다.")
        for field in ("kinds", "phases", "features", "joints"):
            if not isinstance(rule.get(field), list) or not all(isinstance(v, str) for v in rule[field]):
                raise ValueError("코칭 규칙 목록 필드가 잘못되었습니다: " + field)
        if any(p not in ("READY", "BACKSWING", "FORWARD_SWING", "FOLLOW_THROUGH", "RECOVERY") for p in rule["phases"]):
            raise ValueError("지원되지 않는 코칭 단계")
        minimum = rule.get("min_valid_ratio")
        if isinstance(minimum, bool) or not isinstance(minimum, (float, int)) or not 0.5 <= minimum <= 1:
            raise ValueError("규칙별 최소 관측 비율은 0.5~1이어야 합니다.")
    return RuleSet(data["version"], fingerprint(data), tuple(rules))
