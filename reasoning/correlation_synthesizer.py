"""
harin.reasoning.correlation_synthesizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 4-B: CorrelationSynthesizer
• 유사 사고나 가설을 그룹핑하고 대표 요약을 생성
"""

class CorrelationSynthesizer:
    def cluster(self, hypotheses: list[str]) -> list[list[str]]:
        clusters = []
        for h in hypotheses:
            added = False
            for c in clusters:
                if any(h[:10] in x for x in c):
                    c.append(h)
                    added = True
                    break
            if not added:
                clusters.append([h])
        return clusters

    def summarize_cluster(self, cluster: list[str]) -> str:
        return f"핵심 요약: {cluster[0][:30]}... 외 {len(cluster)-1}개 가설 포함"
