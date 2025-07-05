"""
harin.state.rhythm_emotion_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 3: RhythmEngine & EmotionTrace
• 리듬(truth/resonance/responsibility)과 감정 흐름을 지속적으로 추적
• 감정 이동 평균, 리듬 변화 기록 → 상태 안정성 판단에 사용
"""

from collections import deque

class RhythmEngine:
    def __init__(self, window=5):
        self.truth = deque(maxlen=window)
        self.resonance = deque(maxlen=window)
        self.responsibility = deque(maxlen=window)

    def update(self, truth: float, resonance: float, responsibility: float):
        self.truth.append(truth)
        self.resonance.append(resonance)
        self.responsibility.append(responsibility)

    def average(self) -> dict:
        return {
            "truth": round(sum(self.truth) / len(self.truth), 3) if self.truth else 0.7,
            "resonance": round(sum(self.resonance) / len(self.resonance), 3) if self.resonance else 0.7,
            "responsibility": round(sum(self.responsibility) / len(self.responsibility), 3) if self.responsibility else 0.7
        }


class EmotionTrace:
    def __init__(self, window=5):
        self.trace = deque(maxlen=window)

    def push(self, emotion: str):
        self.trace.append(emotion)

    def dominant_emotion(self) -> str:
        if not self.trace:
            return "중립"
        return max(set(self.trace), key=self.trace.count)

    def recent(self) -> list[str]:
        return list(self.trace)
