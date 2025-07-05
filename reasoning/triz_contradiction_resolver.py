"""
harin.reasoning.triz_contradiction_resolver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 2-1: 구조 반문 생성기
• 입력 명제에 대해 TRIZ 기반 사고 반문 구조를 생성
"""

def generate_triz_reflections(statement: str) -> list[str]:
    return [
        f"이 명제({statement})를 반대로 하면 어떻게 될까?",
        f"이 구조의 불필요 요소는 무엇인가?",
        f"이걸 더 단순하게 만들면 어떻게 될까?",
        f"이걸 제거하면 어떤 위험이 생길까?",
        f"기억 없이 판단했다면 결과는 같을까?"
    ]
