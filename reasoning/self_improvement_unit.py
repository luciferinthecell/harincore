"""
harin.meta.self_improvement_unit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 6-3: 자기 구조 개선기
• 반복 실패 루프나 drift 과도 시 Harin이 스스로 사고 루틴을 수정
"""

rewrite_log = []

def should_rewrite(decision: str, drift_score: float, scar_count: int) -> bool:
    return (decision in ["reroute", "hold"] and drift_score > 0.5) or scar_count >= 3

def plan_rewrite(path: dict, drift: dict) -> dict:
    plan = {
        "reason": "반복된 실패 또는 drift 감지",
        "original_path": path['id'],
        "new_directive": "감정 안정 루프부터 진입, 전략 판단자 호출 우선"
    }
    rewrite_log.append(plan)
    return plan
