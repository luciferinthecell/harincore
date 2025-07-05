"""
harin.meta.feedback_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 4: FeedbackEngine
• 사용자 피드백 (명시적/암시적)을 수집하여 루프 흐름, scar, self-improvement에 반영
"""

from collections import deque
from typing import List, Dict

class FeedbackEngine:
    def __init__(self, window: int = 10):
        self.signals = deque(maxlen=window)

    def register(self, feedback: str):
        self.signals.append(feedback)

    def recent(self, n: int = 3) -> List[str]:
        return list(self.signals)[-n:]

    def sentiment_score(self) -> float:
        score = 0.0
        for signal in self.signals:
            if signal in ["👍", "좋아", "고마워"]:
                score += 1.0
            elif signal in ["👎", "틀렸어", "아니야"]:
                score -= 1.0
        return round(score / len(self.signals), 2) if self.signals else 0.0

    def trigger_reroute(self) -> bool:
        # 부정 피드백이 2회 이상 누적될 경우 재루프 유도
        return self.signals.count("👎") + self.signals.count("틀렸어") >= 2
