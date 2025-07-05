"""
harin.reasoning.world_model_simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 6-2: 사고 경로 시뮬레이션기
• 사고 판단 결과를 실행했을 때 예상 결과와 리스크를 시뮬레이션
"""

def simulate_path(path: dict, context: dict) -> dict:
    effects = []
    if "중단" in path['statement']:
        effects.append("감정 안정화, 판단 지연")
    if "강행" in path['statement']:
        effects.append("속도 우선, 실패 리스크 상승")
    risk = 0.4
    if context.get("emotion") in ["불안", "혼란"]:
        risk += 0.2
    if context.get("rhythm", {}).get("truth", 1.0) < 0.5:
        risk += 0.2
    return {
        "simulated_effects": effects,
        "risk_score": round(risk, 2)
    }
