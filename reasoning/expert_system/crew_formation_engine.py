"""
harin.reasoning.crew_formation_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 6-1: 전문가 에이전트 크루 구성기
• 입력 의도/감정/키워드에 따라 적절한 사고 전문가 집단을 자동 구성
"""

def form_crew(intent: str, emotion: str, keywords: list[str]) -> list[str]:
    crew = []
    if intent in ["질문", "정의"]:
        crew.append("LogicExpert")
    if intent == "명령":
        crew.append("StrategyExpert")
    if "반문" in keywords:
        crew.append("CritiqueAgent")
    if emotion in ["불안", "분노"]:
        crew.append("EmotionModerator")
    if "창의" in keywords:
        crew.append("CreativeThinker")
    return list(dict.fromkeys(crew))  # remove duplicates
