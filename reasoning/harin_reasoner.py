"""
harin.reasoning.harin_reasoner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 4-1: 사고 실행기
• 반문 + 기억 기반 사고 분기 경로 생성 및 최적 선택
"""

def generate_paths(reflections: list[str], memories: list[dict]) -> list[dict]:
    paths = []
    for i, r in enumerate(reflections):
        linked = [m for m in memories if r[:10] in m['content']]
        score = 0.6 + 0.1 * len(linked)
        paths.append({"id": f"path_{i+1:02}", "statement": r, "linked": linked, "score": round(score, 2)})
    return sorted(paths, key=lambda p: -p['score'])

def select_best_path(paths: list[dict]) -> dict:
    return paths[0] if paths else {"id": "default", "statement": "기본 경로", "score": 0.0, "linked": []}
