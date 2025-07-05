"""
harin.research.researcher
~~~~~~~~~~~~~~~~~~~~~~~~~

웹 검색 기반 연구 모듈 (더미 구현)
"""

class WebSearch:
    """더미 웹 검색 클래스"""
    def search(self, query: str, max_results: int):
        return [{"title": f"더미 결과 {i}", "url": f"http://dummy{i}.com", "content": f"더미 내용 {i}"} 
                for i in range(min(max_results, 3))]

class SourceEvaluator:
    """더미 소스 평가 클래스"""
    def evaluate(self, hit, queries):
        return {"score": 0.5, "relevance": 0.5, "credibility": 0.5}

class Researcher:
    def __init__(self, max_results: int = 10):
        self.search = WebSearch()
        self.eval = SourceEvaluator()
        self.max_results = max_results

    def run(self, query: str):
        hits = self.search.search(query, self.max_results)
        rated = []
        for h in hits:
            eval_result = self.eval.evaluate(h, [query])
            rated.append(eval_result | {"content": h})
        rated.sort(key=lambda x: -x.get("score", 0))
        return rated
