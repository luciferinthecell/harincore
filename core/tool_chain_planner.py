"""
harin.core.tool_chain_planner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 2: ToolChainPlanner
• 사용자 목표 또는 내부 사고 목적에 따라 적절한 도구 흐름을 자동 구성하는 계획 엔진
• 예: [crawl → extract → summarize → compare → store]
"""

class ToolChainPlanner:
    def __init__(self):
        self.templates = {
            "비교 분석": ["crawl", "extract", "contrast", "summarize", "record"],
            "요약 정리": ["retrieve", "summarize", "store"],
            "리스크 평가": ["retrieve", "simulate", "risk_estimate"],
            "가치 정렬": ["retrieve", "evaluate", "align", "record"]
        }

    def plan(self, goal: str) -> list[str]:
        if goal in self.templates:
            return self.templates[goal]
        elif "비교" in goal:
            return self.templates["비교 분석"]
        elif "요약" in goal or "정리" in goal:
            return self.templates["요약 정리"]
        elif "리스크" in goal:
            return self.templates["리스크 평가"]
        else:
            return ["retrieve", "analyze", "respond"]

    def available_templates(self) -> list[str]:
        return list(self.templates.keys())
