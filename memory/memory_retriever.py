"""
harin.memory.memory_retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

사고루프용 기억 검색 및 추출 시스템
- 주제별 관련 기억 자동 검색
- 맥락 기반 기억 필터링
- 개념 연결 기반 추천
- 사고루프 최적화된 기억 제공
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

from memory.palantirgraph import PalantirGraph, ThoughtNode
from memory.text_importer import TopicExtractor, ConceptExtractor


@dataclass
class MemoryContext:
    """기억 검색 컨텍스트"""
    query: str
    topics: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    emotion: str = "neutral"
    context_tags: List[str] = field(default_factory=list)
    max_results: int = 10
    min_relevance: float = 0.3


@dataclass
class RetrievedMemory:
    """검색된 기억"""
    node: ThoughtNode
    relevance_score: float
    retrieval_reason: str
    context_match: Dict[str, Any] = field(default_factory=dict)


class MemoryRetriever:
    """기억 검색 및 추출기"""
    
    def __init__(self, memory_graph: PalantirGraph):
        self.memory = memory_graph
        self.topic_extractor = TopicExtractor()
        self.concept_extractor = ConceptExtractor()
        
    def retrieve_for_thinking(self, user_input: str, context: MemoryContext = None) -> List[RetrievedMemory]:
        """사고루프용 관련 기억 검색"""
        if context is None:
            context = self._create_context_from_input(user_input)
        
        # 다양한 방법으로 기억 검색
        memories = []
        
        # 1. 직접 유사도 검색
        direct_memories = self._direct_similarity_search(user_input, context)
        memories.extend(direct_memories)
        
        # 2. 주제 기반 검색
        topic_memories = self._topic_based_search(context)
        memories.extend(topic_memories)
        
        # 3. 개념 기반 검색
        concept_memories = self._concept_based_search(context)
        memories.extend(concept_memories)
        
        # 4. 맥락 기반 검색
        context_memories = self._context_based_search(context)
        memories.extend(context_memories)
        
        # 중복 제거 및 정렬
        unique_memories = self._deduplicate_and_rank(memories, context)
        
        return unique_memories[:context.max_results]
    
    def _create_context_from_input(self, user_input: str) -> MemoryContext:
        """입력에서 컨텍스트 생성"""
        return MemoryContext(
            query=user_input,
            topics=self.topic_extractor.extract_multiple_topics(user_input),
            concepts=self.concept_extractor.extract_concepts(user_input),
            context_tags=self.topic_extractor.extract_context_tags(user_input)
        )
    
    def _direct_similarity_search(self, query: str, context: MemoryContext) -> List[RetrievedMemory]:
        """직접 유사도 검색"""
        memories = []
        
        # 텍스트 유사도 검색
        similar_nodes = self.memory.find_similar(query, top_k=5, min_score=context.min_relevance)
        
        for node in similar_nodes:
            score = self._calculate_similarity_score(query, node)
            if score >= context.min_relevance:
                memories.append(RetrievedMemory(
                    node=node,
                    relevance_score=score,
                    retrieval_reason="텍스트 유사도",
                    context_match={"similarity": score}
                ))
        
        return memories
    
    def _topic_based_search(self, context: MemoryContext) -> List[RetrievedMemory]:
        """주제 기반 검색"""
        memories = []
        
        for topic in context.topics:
            # 주제와 관련된 노드들 검색
            topic_nodes = self._find_nodes_by_topic(topic)
            
            for node in topic_nodes:
                score = self._calculate_topic_relevance(topic, node)
                if score >= context.min_relevance:
                    memories.append(RetrievedMemory(
                        node=node,
                        relevance_score=score,
                        retrieval_reason=f"주제 매칭: {topic}",
                        context_match={"topic": topic, "topic_score": score}
                    ))
        
        return memories
    
    def _concept_based_search(self, context: MemoryContext) -> List[RetrievedMemory]:
        """개념 기반 검색"""
        memories = []
        
        for concept in context.concepts:
            # 개념과 직접 연결된 노드들 검색
            concept_nodes = self._find_nodes_by_concept(concept)
            
            for node in concept_nodes:
                score = self._calculate_concept_relevance(concept, node)
                if score >= context.min_relevance:
                    memories.append(RetrievedMemory(
                        node=node,
                        relevance_score=score,
                        retrieval_reason=f"개념 연결: {concept}",
                        context_match={"concept": concept, "concept_score": score}
                    ))
        
        return memories
    
    def _context_based_search(self, context: MemoryContext) -> List[RetrievedMemory]:
        """맥락 기반 검색"""
        memories = []
        
        for tag in context.context_tags:
            # 맥락 태그와 관련된 노드들 검색
            context_nodes = self._find_nodes_by_context_tag(tag)
            
            for node in context_nodes:
                score = self._calculate_context_relevance(tag, node)
                if score >= context.min_relevance:
                    memories.append(RetrievedMemory(
                        node=node,
                        relevance_score=score,
                        retrieval_reason=f"맥락 매칭: {tag}",
                        context_match={"context_tag": tag, "context_score": score}
                    ))
        
        return memories
    
    def _find_nodes_by_topic(self, topic: str) -> List[ThoughtNode]:
        """주제로 노드 검색"""
        nodes = []
        topic_lower = topic.lower()
        
        for node in self.memory.nodes.values():
            # 메타데이터에서 주제 확인
            if (node.meta.get("topic", "").lower() == topic_lower or
                topic_lower in node.content.lower()):
                nodes.append(node)
        
        return nodes
    
    def _find_nodes_by_concept(self, concept: str) -> List[ThoughtNode]:
        """개념으로 노드 검색"""
        nodes = []
        concept_lower = concept.lower()
        
        for node in self.memory.nodes.values():
            # 메타데이터에서 개념 태그 확인
            concept_tags = node.meta.get("concept_tags", [])
            if (concept_lower in node.content.lower() or
                any(concept_lower in tag.lower() for tag in concept_tags)):
                nodes.append(node)
        
        return nodes
    
    def _find_nodes_by_context_tag(self, tag: str) -> List[ThoughtNode]:
        """맥락 태그로 노드 검색"""
        nodes = []
        tag_lower = tag.lower()
        
        for node in self.memory.nodes.values():
            # 메타데이터에서 맥락 태그 확인
            context_tags = node.meta.get("context_tags", [])
            if any(tag_lower in ctx_tag.lower() for ctx_tag in context_tags):
                nodes.append(node)
        
        return nodes
    
    def _calculate_similarity_score(self, query: str, node: ThoughtNode) -> float:
        """유사도 점수 계산"""
        # 간단한 키워드 기반 유사도
        query_words = set(query.lower().split())
        node_words = set(node.content.lower().split())
        
        if not query_words or not node_words:
            return 0.0
        
        intersection = query_words & node_words
        union = query_words | node_words
        
        return len(intersection) / len(union)
    
    def _calculate_topic_relevance(self, topic: str, node: ThoughtNode) -> float:
        """주제 관련성 점수 계산"""
        base_score = 0.5
        
        # 메타데이터의 주제와 매칭
        if node.meta.get("topic") == topic:
            base_score += 0.3
        
        # 전체 주제 목록에 포함
        overall_topics = node.meta.get("topics", [])
        if topic in overall_topics:
            base_score += 0.2
        
        # 내용에서 주제 키워드 확인
        topic_keywords = self.topic_extractor.topic_keywords.get(topic, [])
        for keyword in topic_keywords:
            if keyword in node.content.lower():
                base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_concept_relevance(self, concept: str, node: ThoughtNode) -> float:
        """개념 관련성 점수 계산"""
        base_score = 0.4
        
        # 개념 태그에 포함
        concept_tags = node.meta.get("concept_tags", [])
        if concept in concept_tags:
            base_score += 0.4
        
        # 내용에서 개념 언급
        if concept.lower() in node.content.lower():
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _calculate_context_relevance(self, tag: str, node: ThoughtNode) -> float:
        """맥락 관련성 점수 계산"""
        base_score = 0.4
        
        # 맥락 태그에 포함
        context_tags = node.meta.get("context_tags", [])
        if tag in context_tags:
            base_score += 0.4
        
        # 내용에서 맥락 패턴 확인
        if self._has_context_pattern(tag, node.content):
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _has_context_pattern(self, tag: str, content: str) -> bool:
        """맥락 패턴 확인"""
        patterns = {
            "질문": ["?", "무엇", "어떻게", "왜"],
            "설명요청": ["설명", "이해", "알려줘"],
            "예시요청": ["예시", "예제", "사례"],
            "비교": ["비교", "차이", "다른", "vs"],
            "문제해결": ["문제", "해결", "방법", "해결책"]
        }
        
        if tag in patterns:
            return any(pattern in content for pattern in patterns[tag])
        
        return False
    
    def _deduplicate_and_rank(self, memories: List[RetrievedMemory], context: MemoryContext) -> List[RetrievedMemory]:
        """중복 제거 및 정렬"""
        # 노드 ID 기준으로 중복 제거
        seen_nodes = set()
        unique_memories = []
        
        for memory in memories:
            if memory.node.id not in seen_nodes:
                seen_nodes.add(memory.node.id)
                unique_memories.append(memory)
        
        # 점수 기반 정렬 (높은 점수 우선)
        unique_memories.sort(key=lambda m: m.relevance_score, reverse=True)
        
        return unique_memories
    
    def get_memory_summary(self, memories: List[RetrievedMemory]) -> str:
        """검색된 기억들의 요약 생성"""
        if not memories:
            return "관련된 기억이 없습니다."
        
        summary_parts = []
        
        # 주제별 그룹화
        topic_groups = defaultdict(list)
        for memory in memories:
            topic = memory.node.meta.get("topic", "일반")
            topic_groups[topic].append(memory)
        
        for topic, topic_memories in topic_groups.items():
            summary_parts.append(f"\n[{topic}]")
            for memory in topic_memories[:3]:  # 주제당 최대 3개
                summary_parts.append(f"- {memory.node.content[:100]}...")
        
        return "\n".join(summary_parts)


# 사고루프 연동을 위한 헬퍼 클래스
class ThinkingMemoryHelper:
    """사고루프에서 사용할 기억 헬퍼"""
    
    def __init__(self, retriever: MemoryRetriever):
        self.retriever = retriever
    
    def get_relevant_memories_for_loop(self, user_input: str, loop_type: str = "general") -> List[RetrievedMemory]:
        """사고루프 타입별 관련 기억 검색"""
        context = MemoryContext(
            query=user_input,
            max_results=15 if loop_type == "creative" else 10,
            min_relevance=0.2 if loop_type == "creative" else 0.3
        )
        
        return self.retriever.retrieve_for_thinking(user_input, context)
    
    def format_memories_for_prompt(self, memories: List[RetrievedMemory]) -> str:
        """프롬프트용 기억 포맷팅"""
        if not memories:
            return ""
        
        formatted = "\n[관련 기억들]\n"
        
        for i, memory in enumerate(memories[:5], 1):  # 상위 5개만
            formatted += f"{i}. {memory.node.content[:150]}...\n"
            formatted += f"   (관련도: {memory.relevance_score:.2f}, 이유: {memory.retrieval_reason})\n\n"
        
        return formatted


# TopicExtractor와 ConceptExtractor는 text_importer.py에서 import
from memory.text_importer import TopicExtractor, ConceptExtractor 
