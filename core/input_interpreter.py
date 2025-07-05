"""
harin.core.input_interpreter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 1-1: 사용자 입력 해석기
• 사용자 입력을 Harin 사고 시스템이 사용할 수 있는 구조로 해석
• 감정, 목적, 리듬 강도, 기억 연결 단서, drift 여부 등을 추출
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ParsedInput:
    intent: str
    emotion: str
    tone_force: float
    drift_trigger: bool
    memory_refs: List[str]
    requires_research: bool


class InputInterpreter:
    def analyze(self, text: str) -> ParsedInput:
        return ParsedInput(
            intent=self._classify_intent(text),
            emotion=self._detect_emotion(text),
            tone_force=self._measure_tone(text),
            drift_trigger=self._detect_drift(text),
            memory_refs=self._extract_memory_refs(text),
            requires_research=self._needs_research(text)
        )

    def _classify_intent(self, text: str) -> str:
        if any(x in text for x in ["해줘", "시작", "정리"]): return "명령"
        if any(x in text for x in ["왜", "어떻게", "뭔"]): return "질문"
        if "나는" in text: return "선언"
        return "관찰"

    def _detect_emotion(self, text: str) -> str:
        if "짜증" in text or "불만" in text: return "분노"
        if "두려" in text or "불안" in text: return "불안"
        if "좋아" in text or "편안" in text: return "긍정"
        return "중립"

    def _measure_tone(self, text: str) -> float:
        return min(1.0, 0.3 + 0.1 * text.count("!") + 0.05 * len(text.split()))

    def _detect_drift(self, text: str) -> bool:
        return any(x in text for x in ["틀렸", "다시", "이전과 달라"])

    def _extract_memory_refs(self, text: str) -> List[str]:
        return [k for k in ["기억", "정체성", "루프", "실패"] if k in text]

    def _needs_research(self, text: str) -> bool:
        return any(x in text for x in ["논문", "최신", "검색"])


# Debug
if __name__ == '__main__':
    i = InputInterpreter()
    parsed = i.analyze("하린, 루프 정체성 다시 정리해줘. 이전과 달라!")
    print(parsed)
