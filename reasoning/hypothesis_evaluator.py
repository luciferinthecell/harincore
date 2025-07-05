"""
harin.reasoning.hypothesis_evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 4-A: HypothesisEvaluator
• 다관점 사고 분기에서 나온 가설들을 점수화하여 랭킹
기준: 감정 일치도, 리듬 적합도, 메모리 연관도
"""

class HypothesisEvaluator:
    def __init__(self, session):
        self.session = session

    def score(self, hypothesis: str, meta: dict) -> float:
        score = 0.5
        if meta.get("emotion") == self.session.emotion:
            score += 0.2
        if abs(meta.get("rhythm", {}).get("truth", 0.7) - self.session.rhythm["truth"]) < 0.2:
            score += 0.2
        if meta.get("memory_linked", False):
            score += 0.1
        return round(score, 3)
