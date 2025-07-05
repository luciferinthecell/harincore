"""
harin.loop.interaction_loop_controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 7-2: 상호작용 흐름 감시기
• Harin과 사용자 사이의 루프 흐름을 감정/리듬 기반으로 제어
"""

history = []

def observe_interaction(user_input: str, harin_output: str, rhythm: dict, emotion: str) -> dict:
    history.append({"input": user_input, "output": harin_output, "rhythm": rhythm, "emotion": emotion})
    if len(history) >= 3:
        recent = history[-3:]
        avg_truth = sum(h['rhythm']['truth'] for h in recent) / 3
        neg_emotions = sum(1 for h in recent if h['emotion'] in ["불안", "분노", "슬픔"])
        if avg_truth < 0.4 or neg_emotions >= 2:
            return {"flow": "pause", "reason": "불안정 흐름"}
    return {"flow": "continue", "reason": "안정"}
