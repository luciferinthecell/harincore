"""
harin.reasoning.thought_diversifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 2-2: 다관점 사고 분기기
• 하나의 명제를 논리/감정/전략/창의 관점으로 분리
"""

from typing import List, Dict, Set
import itertools
import numpy as np

class ThoughtElement:
    def __init__(self, text: str, tags: Set[str]):
        self.text = text
        self.tags = set(tags)
        self.embedding = self.fake_embed(text)

    def fake_embed(self, text):
        # 실제 환경에서는 임베딩 모델로 교체
        return np.random.rand(1, 128)

def cosine_similarity(a, b):
    """numpy를 사용한 코사인 유사도 계산"""
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def tag_similarity(tags1, tags2):
    return len(tags1 & tags2) / len(tags1 | tags2) if tags1 | tags2 else 0

def cluster_by_tag_similarity(elements, sim_matrix, threshold):
    from collections import defaultdict
    clusters = []
    used = set()
    for idx, el in enumerate(elements):
        if idx in used:
            continue
        cluster = [idx]
        used.add(idx)
        for j in range(len(elements)):
            if j != idx and sim_matrix[idx][j] > threshold and j not in used:
                cluster.append(j)
                used.add(j)
        clusters.append([elements[i].text for i in cluster])
    return clusters

def process_thought_graph(raw_elements: List[Dict], threshold=0.7):
    elements = [ThoughtElement(e["text"], e["tags"]) for e in raw_elements]
    N = len(elements)
    sim_matrix = np.zeros((N, N))
    for i, j in itertools.combinations(range(N), 2):
        tag_sim = tag_similarity(elements[i].tags, elements[j].tags)
        embed_sim = cosine_similarity(elements[i].embedding, elements[j].embedding)
        sim_matrix[i][j] = sim_matrix[j][i] = 0.6 * tag_sim + 0.4 * embed_sim
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if sim_matrix[i][j] > threshold:
                edges.append((i, j, round(sim_matrix[i][j], 2)))
    clusters = cluster_by_tag_similarity(elements, sim_matrix, threshold)
    return {
        "nodes": [{"text": el.text, "tags": list(el.tags)} for el in elements],
        "edges": edges,
        "clusters": clusters,
        "summary": f"총 {len(clusters)}개의 사고 루프가 생성되었습니다. 공명도 평균: {round(np.mean(sim_matrix), 3)}"
    }

def diversify_thought(statement: str) -> list[dict]:
    return [
        {"type": "logic", "question": f"논리적으로 이 명제({statement})는 타당한가?"},
        {"type": "emotion", "question": f"감정적으로 이 명제({statement})는 어떤 반응을 유발하는가?"},
        {"type": "strategy", "question": f"전략적으로 이 명제({statement})의 실행 이점은?"},
        {"type": "creativity", "question": f"이 명제({statement})를 완전히 다르게 구성하면?"}
    ]
