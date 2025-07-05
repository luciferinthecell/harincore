"""
harin.meta.reloop_trigger
~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 2-3: 사고 재진입 유도기
• 실패, 불일치, 감정 불안정 등을 기반으로 루프 재진입을 판단
"""

def should_reloop(drift: float, emotion: str, failed_recently: bool) -> bool:
    if drift > 0.5:
        return True
    if emotion in ["불안", "분노"] and failed_recently:
        return True
    return False
