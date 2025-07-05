"""
harin.core.rhythm_governor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 1-2: 감정 기반 리듬 조절기
• 감정과 tone_force를 기반으로 사고 속도/깊이/회피 유무를 조절
"""

def regulate(emotion: str, tone_force: float) -> dict:
    if emotion in ["불안", "분노"]:
        return {"mode": "slow", "reason": "감정 불안정", "intensity": tone_force}
    elif tone_force > 0.8:
        return {"mode": "intensify", "reason": "명령 강도 높음", "intensity": tone_force}
    else:
        return {"mode": "normal", "reason": "안정 상태", "intensity": tone_force}
