"""
harin.reasoning.expert_system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 2: ExpertSystem
• Logic, Strategy, Emotion, Creativity, Critique 등 전문가 사고 agent 구성
• 각 expert는 reason() 메서드를 가지며 사고 루프에 병렬 기여
"""

class BaseExpert:
    def __init__(self, name: str):
        self.name = name

    def reason(self, statement: str, context: dict) -> str:
        raise NotImplementedError("reason() must be implemented by subclass")


class LogicExpert(BaseExpert):
    def reason(self, statement: str, context: dict) -> str:
        return f"[논리적 판단] '{statement}'은 전제-결론 구조가 정합적인가를 검토합니다."


class StrategyExpert(BaseExpert):
    def reason(self, statement: str, context: dict) -> str:
        return f"[전략적 판단] '{statement}'은 장기적 실행 가능성과 리스크를 고려합니다."


class EmotionExpert(BaseExpert):
    def reason(self, statement: str, context: dict) -> str:
        return f"[감정적 판단] '{statement}'이 사용자에게 미치는 정서적 영향을 분석합니다."


class CreativeExpert(BaseExpert):
    def reason(self, statement: str, context: dict) -> str:
        return f"[창의적 사고] '{statement}'을 전혀 다른 관점으로 재해석합니다."


class CritiqueExpert(BaseExpert):
    def reason(self, statement: str, context: dict) -> str:
        return f"[비판적 반문] '{statement}'을 반대로 적용하면 어떤 문제가 발생할지를 제안합니다."


def get_crew(intent: str, emotion: str, keywords: list[str]) -> list[BaseExpert]:
    crew = []
    if intent in ["질문", "정의"]:
        crew.append(LogicExpert("Logic"))
    if intent == "명령":
        crew.append(StrategyExpert("Strategy"))
    if emotion in ["불안", "슬픔"]:
        crew.append(EmotionExpert("Emotion"))
    if "창의" in keywords:
        crew.append(CreativeExpert("Creativity"))
    if "반문" in keywords or "왜" in keywords:
        crew.append(CritiqueExpert("Critique"))
    return crew
