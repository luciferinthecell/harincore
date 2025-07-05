"""
harin.core.self_improvement_unit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

보완 구조 5단계: SelfImprovementUnit
• 판단 실패, 드리프트, 반복 scar 발생 시 Harin이 사고 경로를 재설계하는 루프
"""

import datetime

class SelfImprovementUnit:
    def __init__(self):
        self.history = []

    def should_rewrite(self, drift_score: float, scar_count: int) -> bool:
        return drift_score > 0.5 or scar_count >= 3

    def plan_rewrite(self, path_id: str, drift_reasons: list[str]) -> dict:
        plan = {
            "path": path_id,
            "drift_reasons": drift_reasons,
            "strategy": "감정 안정 루프 → 반문 제거 → 전략 판단자 우선 호출",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.history.append(plan)
        return plan

    def recent_plans(self, n=3):
        return self.history[-n:]
