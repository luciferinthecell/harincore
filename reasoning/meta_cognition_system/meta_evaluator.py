"""
harin.reasoning.meta_evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 4-2: 메타 판단기
• 사고 경로의 실행 가능성 평가 (신뢰도, 감정, 리듬 기반)
"""

def evaluate_path(path: dict, emotion: str, rhythm: dict) -> dict:
    reasons = []
    decision = "use"
    if path['score'] < 0.6:
        decision = "reroute"
        reasons.append("low_score")
    if rhythm.get("truth", 1.0) < 0.4:
        decision = "hold"
        reasons.append("low_truth")
    if emotion in ["불안", "혼란"] and path['score'] < 0.75:
        decision = "delay"
        reasons.append("emotion_mismatch")
    return {"decision": decision, "reasons": reasons, "score": path['score']}
