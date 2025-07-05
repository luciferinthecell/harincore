"""
harin.core.memory_orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

하린 사고 루프용 메모리 오케스트레이터
- cold / hot / cache 기억을 통합 분석
- 프롬프트에 반영할 메모리 추출
- scar 조건 감지 시 재사고 루프 유도
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from scipy.spatial.distance import cosine
from memory.models import MemoryEpisodeNode
from memory.models import HarinThoughtNode, HarinEdge


class MemoryOrchestrator:
    def __init__(self, cold_path="data/memory_data/h2.jsonl", hot_path="data/memory_data/ha1.jsonl"):
        self.h2_path = cold_path
        self.ha1_path = hot_path

    def load_hot_topics(self, query_vector: List[float], top_k: int = 2) -> List[str]:
        """핫 메모리에서 관련 토픽 로드"""
        results = []
        try:
            with open(self.ha1_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    node = MemoryEpisodeNode(**obj)
                    sim = 1 - cosine(query_vector, node.embedding)
                    if sim > 0.6:
                        results.append((sim, node.topic_summary))
        except Exception as e:
            print(f"핫 토픽 로드 오류: {e}")
            return []
        
        results.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in results[:top_k]]

    def detect_scar_violation(self, generated_text: str) -> Optional[str]:
        """SCAR 위반 감지"""
        try:
            with open(self.h2_path, "r", encoding="utf-8") as f:
                for line in f:
                    node = json.loads(line)
                    if node.get("type") == "memory" and "scar." in str(node.get("tags", [])):
                        scar_text = node.get("content", "")
                        if scar_text and scar_text[:50] in generated_text:
                            return node["tags"][0]
        except Exception as e:
            print(f"SCAR 감지 오류: {e}")
        return None

    def assemble_memory_context(self, query_vector: List[float], generated_text: str = None) -> Dict[str, Any]:
        """메모리 컨텍스트 조립"""
        context = {}

        # 핫 메모리 토픽 로드
        hot = self.load_hot_topics(query_vector)
        if hot:
            context["hot_memory_topics"] = hot

        # SCAR 위반 감지
        if generated_text:
            scar = self.detect_scar_violation(generated_text)
            if scar:
                context["scar_triggered"] = scar

        return context

    def create_thought_node(self, node_id: str, node_type: str, tags: List[str], 
                          content: str, relations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """사고 노드 생성"""
        edges = [HarinEdge(**r) for r in (relations or [])]
        node = HarinThoughtNode(
            id=node_id,
            type=node_type,
            tags=tags,
            content=content,
            relations=edges
        )
        return node.model_dump()  # 딕셔너리로 반환 
 