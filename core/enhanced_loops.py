"""
harin.core.enhanced_loops
~~~~~~~~~~~~~~~~~~~~~~~~

기억을 활용하는 향상된 사고루프
- PalantirGraph 기억과 연동
- 관련 기억 자동 검색 및 활용
- 맥락 기반 추론
- 개념 연결 기반 사고
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import uuid

from core.judgment import Judgment, ScoreVector
from core.context import UserContext
from memory.palantirgraph import PalantirGraph
from memory.memory_retriever import MemoryRetriever, ThinkingMemoryHelper, RetrievedMemory


@dataclass
class EnhancedJudgment:
    """향상된 판단 결과"""
    loop_id: str
    output_text: str
    score: ScoreVector
    rationale: str
    used_memories: List[RetrievedMemory] = field(default_factory=list)
    memory_influence: Dict[str, float] = field(default_factory=dict)


class MemoryAwareLoop:
    """기억을 인식하는 기본 루프 클래스"""
    
    def __init__(self, name: str, memory_retriever: MemoryRetriever):
        self.name = name
        self.retriever = memory_retriever
        self.helper = ThinkingMemoryHelper(memory_retriever)
    
    def run(self, *, memory: PalantirGraph, context: UserContext, user_input: str) -> EnhancedJudgment:
        """기본 실행 메서드 - 하위 클래스에서 오버라이드"""
        raise NotImplementedError


class EnhancedRetrievalLoop(MemoryAwareLoop):
    """기억 기반 검색 루프"""
    
    def __init__(self, memory_retriever: MemoryRetriever):
        super().__init__("enhanced_retrieval", memory_retriever)
    
    def run(self, *, memory: PalantirGraph, context: UserContext, user_input: str) -> EnhancedJudgment:
        # 관련 기억 검색
        relevant_memories = self.helper.get_relevant_memories_for_loop(user_input, "retrieval")
        
        if not relevant_memories:
            output = "관련된 기억이 없어 새로운 정보를 제공할 수 없습니다."
            score = ScoreVector(persuasiveness=0.3, consistency=0.5, credibility=0.4, affect_match=0.5)
            return EnhancedJudgment(
                loop_id=self.name,
                output_text=output,
                score=score,
                rationale="기억 없음",
                used_memories=[]
            )
        
        # 기억 기반 답변 생성
        memory_summary = self._create_memory_summary(relevant_memories)
        output = self._generate_response_from_memories(user_input, relevant_memories, memory_summary)
        
        # 점수 계산
        score = self._calculate_score(relevant_memories, output)
        
        # 기억 영향도 분석
        memory_influence = self._analyze_memory_influence(relevant_memories)
        
        return EnhancedJudgment(
            loop_id=self.name,
            output_text=output,
            score=score,
            rationale="기억 기반 검색",
            used_memories=relevant_memories,
            memory_influence=memory_influence
        )
    
    def _create_memory_summary(self, memories: List[RetrievedMemory]) -> str:
        """기억 요약 생성"""
        summary_parts = []
        
        # 주제별 그룹화
        topic_groups = {}
        for memory in memories:
            topic = memory.node.meta.get("topic", "일반")
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(memory)
        
        for topic, topic_memories in topic_groups.items():
            summary_parts.append(f"\n[{topic} 관련 기억]")
            for memory in topic_memories[:3]:  # 주제당 최대 3개
                content = memory.node.content[:100] + "..." if len(memory.node.content) > 100 else memory.node.content
                summary_parts.append(f"- {content}")
        
        return "\n".join(summary_parts)
    
    def _generate_response_from_memories(self, user_input: str, memories: List[RetrievedMemory], summary: str) -> str:
        """기억을 바탕으로 답변 생성"""
        if not memories:
            return "관련된 기억이 없습니다."
        
        # 가장 관련성 높은 기억들 선택
        top_memories = sorted(memories, key=lambda m: m.relevance_score, reverse=True)[:3]
        
        response_parts = []
        response_parts.append("기억을 바탕으로 답변드리겠습니다:")
        
        for i, memory in enumerate(top_memories, 1):
            content = memory.node.content
            if len(content) > 200:
                content = content[:200] + "..."
            
            response_parts.append(f"\n{i}. {content}")
            response_parts.append(f"   (관련도: {memory.relevance_score:.2f})")
        
        # 추가 컨텍스트
        if len(memories) > 3:
            response_parts.append(f"\n\n총 {len(memories)}개의 관련 기억이 있습니다.")
        
        return "\n".join(response_parts)
    
    def _calculate_score(self, memories: List[RetrievedMemory], output: str) -> ScoreVector:
        """점수 계산"""
        if not memories:
            return ScoreVector(persuasiveness=0.3, consistency=0.5, credibility=0.4, affect_match=0.5)
        
        # 기억 기반 점수 계산
        avg_relevance = sum(m.relevance_score for m in memories) / len(memories)
        memory_count = len(memories)
        
        return ScoreVector(
            persuasiveness=min(0.9, 0.5 + avg_relevance * 0.4),
            consistency=min(0.9, 0.6 + memory_count * 0.05),
            credibility=min(0.9, 0.7 + avg_relevance * 0.2),
            affect_match=min(0.9, 0.6 + avg_relevance * 0.3)
        )
    
    def _analyze_memory_influence(self, memories: List[RetrievedMemory]) -> Dict[str, float]:
        """기억 영향도 분석"""
        influence = {
            "total_memories": len(memories),
            "avg_relevance": 0.0,
            "topic_diversity": 0.0,
            "concept_coverage": 0.0
        }
        
        if memories:
            influence["avg_relevance"] = sum(m.relevance_score for m in memories) / len(memories)
            
            # 주제 다양성
            topics = set(m.node.meta.get("topic", "일반") for m in memories)
            influence["topic_diversity"] = len(topics) / max(len(memories), 1)
            
            # 개념 커버리지
            all_concepts = set()
            for memory in memories:
                concepts = memory.node.meta.get("concept_tags", [])
                all_concepts.update(concepts)
            influence["concept_coverage"] = len(all_concepts)
        
        return influence


class EnhancedCreativeLoop(MemoryAwareLoop):
    """기억 기반 창의적 루프"""
    
    def __init__(self, memory_retriever: MemoryRetriever):
        super().__init__("enhanced_creative", memory_retriever)
    
    def run(self, *, memory: PalantirGraph, context: UserContext, user_input: str) -> EnhancedJudgment:
        # 창의적 사고를 위한 다양한 기억 검색
        relevant_memories = self.helper.get_relevant_memories_for_loop(user_input, "creative")
        
        # 창의적 연결 생성
        creative_connections = self._find_creative_connections(relevant_memories, user_input)
        
        # 창의적 답변 생성
        output = self._generate_creative_response(user_input, relevant_memories, creative_connections)
        
        # 점수 계산
        score = self._calculate_creative_score(relevant_memories, creative_connections)
        
        return EnhancedJudgment(
            loop_id=self.name,
            output_text=output,
            score=score,
            rationale="기억 기반 창의적 사고",
            used_memories=relevant_memories,
            memory_influence={"creative_connections": len(creative_connections)}
        )
    
    def _find_creative_connections(self, memories: List[RetrievedMemory], user_input: str) -> List[Dict[str, Any]]:
        """창의적 연결 찾기"""
        connections = []
        
        # 개념 간 연결 찾기
        concepts = set()
        for memory in memories:
            concepts.update(memory.node.meta.get("concept_tags", []))
        
        # 관련 개념들로 추가 기억 검색
        for concept in list(concepts)[:5]:  # 상위 5개 개념만
            related_memories = self.retriever._find_nodes_by_concept(concept)
            for node in related_memories[:3]:  # 개념당 최대 3개
                if node not in [m.node for m in memories]:
                    connections.append({
                        "concept": concept,
                        "memory": node,
                        "connection_type": "concept_link"
                    })
        
        return connections
    
    def _generate_creative_response(self, user_input: str, memories: List[RetrievedMemory], connections: List[Dict[str, Any]]) -> str:
        """창의적 답변 생성"""
        response_parts = []
        response_parts.append(f"'{user_input}'에 대한 창의적인 관점을 제시해보겠습니다:")
        
        # 기존 기억들의 창의적 재해석
        if memories:
            response_parts.append("\n[기존 지식의 창의적 활용]")
            for memory in memories[:3]:
                creative_interpretation = self._create_creative_interpretation(memory, user_input)
                response_parts.append(f"- {creative_interpretation}")
        
        # 새로운 연결 제시
        if connections:
            response_parts.append("\n[새로운 관점과 연결]")
            for connection in connections[:3]:
                response_parts.append(f"- {connection['concept']} 관점에서: {connection['memory'].content[:100]}...")
        
        response_parts.append("\n이러한 다양한 관점을 종합하면 새로운 해결책이나 아이디어를 발견할 수 있습니다.")
        
        return "\n".join(response_parts)
    
    def _create_creative_interpretation(self, memory: RetrievedMemory, user_input: str) -> str:
        """기억의 창의적 재해석"""
        content = memory.node.content
        topic = memory.node.meta.get("topic", "일반")
        
        interpretations = {
            "기술": f"기술적 관점에서 '{content[:50]}...'를 다른 분야에 적용해볼 수 있습니다.",
            "과학": f"과학적 원리를 바탕으로 '{content[:50]}...'를 실험적으로 검증해볼 수 있습니다.",
            "예술": f"예술적 창작의 관점에서 '{content[:50]}...'를 표현해볼 수 있습니다.",
            "경제": f"경제적 가치의 관점에서 '{content[:50]}...'를 분석해볼 수 있습니다."
        }
        
        return interpretations.get(topic, f"'{content[:50]}...'를 새로운 관점에서 재해석해볼 수 있습니다.")
    
    def _calculate_creative_score(self, memories: List[RetrievedMemory], connections: List[Dict[str, Any]]) -> ScoreVector:
        """창의적 점수 계산"""
        base_score = 0.6
        
        # 기억 다양성
        topics = set(m.node.meta.get("topic", "일반") for m in memories)
        topic_diversity = len(topics) / max(len(memories), 1)
        
        # 연결 수
        connection_bonus = min(0.2, len(connections) * 0.05)
        
        return ScoreVector(
            persuasiveness=min(0.9, base_score + topic_diversity * 0.2),
            consistency=min(0.8, base_score + 0.1),
            credibility=min(0.7, base_score - 0.1),
            affect_match=min(0.9, base_score + connection_bonus)
        )


class EnhancedAnalyticalLoop(MemoryAwareLoop):
    """기억 기반 분석적 루프"""
    
    def __init__(self, memory_retriever: MemoryRetriever):
        super().__init__("enhanced_analytical", memory_retriever)
    
    def run(self, *, memory: PalantirGraph, context: UserContext, user_input: str) -> EnhancedJudgment:
        # 분석을 위한 체계적 기억 검색
        relevant_memories = self.helper.get_relevant_memories_for_loop(user_input, "analytical")
        
        # 분석적 구조 생성
        analysis_structure = self._create_analysis_structure(relevant_memories, user_input)
        
        # 분석적 답변 생성
        output = self._generate_analytical_response(user_input, relevant_memories, analysis_structure)
        
        # 점수 계산
        score = self._calculate_analytical_score(relevant_memories, analysis_structure)
        
        return EnhancedJudgment(
            loop_id=self.name,
            output_text=output,
            score=score,
            rationale="기억 기반 분석적 사고",
            used_memories=relevant_memories,
            memory_influence={"analysis_depth": len(analysis_structure)}
        )
    
    def _create_analysis_structure(self, memories: List[RetrievedMemory], user_input: str) -> Dict[str, List[RetrievedMemory]]:
        """분석 구조 생성"""
        structure = {
            "supporting_evidence": [],
            "contrasting_views": [],
            "related_concepts": [],
            "historical_context": []
        }
        
        for memory in memories:
            # 증거 분류
            if memory.relevance_score > 0.7:
                structure["supporting_evidence"].append(memory)
            elif memory.relevance_score < 0.4:
                structure["contrasting_views"].append(memory)
            
            # 개념 분류
            if memory.node.meta.get("concept_tags"):
                structure["related_concepts"].append(memory)
            
            # 시간적 컨텍스트
            if memory.node.meta.get("timestamp"):
                structure["historical_context"].append(memory)
        
        return structure
    
    def _generate_analytical_response(self, user_input: str, memories: List[RetrievedMemory], structure: Dict[str, List[RetrievedMemory]]) -> str:
        """분석적 답변 생성"""
        response_parts = []
        response_parts.append(f"'{user_input}'에 대한 체계적 분석을 제공하겠습니다:")
        
        # 지지하는 증거
        if structure["supporting_evidence"]:
            response_parts.append("\n[지지하는 증거]")
            for memory in structure["supporting_evidence"][:3]:
                response_parts.append(f"- {memory.node.content[:100]}...")
        
        # 대조되는 관점
        if structure["contrasting_views"]:
            response_parts.append("\n[대조되는 관점]")
            for memory in structure["contrasting_views"][:2]:
                response_parts.append(f"- {memory.node.content[:100]}...")
        
        # 관련 개념
        if structure["related_concepts"]:
            response_parts.append("\n[관련 개념들]")
            concepts = set()
            for memory in structure["related_concepts"]:
                concepts.update(memory.node.meta.get("concept_tags", []))
            response_parts.append(f"- {', '.join(list(concepts)[:5])}")
        
        # 결론
        response_parts.append("\n[분석 결론]")
        response_parts.append("위의 증거들을 종합하여 체계적인 분석 결과를 제시했습니다.")
        
        return "\n".join(response_parts)
    
    def _calculate_analytical_score(self, memories: List[RetrievedMemory], structure: Dict[str, List[RetrievedMemory]]) -> ScoreVector:
        """분석적 점수 계산"""
        evidence_count = len(structure["supporting_evidence"])
        concept_count = len(structure["related_concepts"])
        
        return ScoreVector(
            persuasiveness=min(0.9, 0.6 + evidence_count * 0.1),
            consistency=min(0.9, 0.8 + concept_count * 0.05),
            credibility=min(0.9, 0.8 + evidence_count * 0.1),
            affect_match=min(0.8, 0.6 + evidence_count * 0.05)
        )


# 향상된 루프 매니저
class EnhancedLoopManager:
    """향상된 루프 매니저"""
    
    def __init__(self, memory_retriever: MemoryRetriever):
        self.retriever = memory_retriever
        self.registry = {
            "enhanced_retrieval": EnhancedRetrievalLoop(memory_retriever),
            "enhanced_creative": EnhancedCreativeLoop(memory_retriever),
            "enhanced_analytical": EnhancedAnalyticalLoop(memory_retriever)
        }
    
    def run_all(self, text: str, context: UserContext, memory: PalantirGraph) -> List[EnhancedJudgment]:
        """모든 향상된 루프 실행"""
        judgments = []
        
        for loop in self.registry.values():
            try:
                judgment = loop.run(memory=memory, context=context, user_input=text)
                judgments.append(judgment)
            except Exception as e:
                # 에러 발생 시 기본 판단 생성
                judgment = EnhancedJudgment(
                    loop_id=loop.name,
                    output_text=f"루프 실행 중 오류: {str(e)}",
                    score=ScoreVector(persuasiveness=0.3, consistency=0.3, credibility=0.3, affect_match=0.3),
                    rationale="오류 발생"
                )
                judgments.append(judgment)
        
        return judgments
    
    def get_best_judgment(self, judgments: List[EnhancedJudgment]) -> EnhancedJudgment:
        """최적 판단 선택"""
        if not judgments:
            return None
        
        # 점수 기반 선택
        return max(judgments, key=lambda j: j.score.overall()) 
