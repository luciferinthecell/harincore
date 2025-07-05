"""
harin.memory.contextual_memory_system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

맥락 기반 기억 시스템
- 키워드 매칭이 아닌 상황 인식
- 주제별, 문맥별, 관계별 인식
- 벡터화 기반 의미 기억
- 자기 진화하는 기억 구조
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from memory.palantirgraph import PalantirGraph, ThoughtNode, Relationship


class ContextType(Enum):
    """맥락 타입"""
    TOPIC = "topic"           # 주제
    SITUATION = "situation"   # 상황
    RELATIONSHIP = "relationship"  # 관계
    EMOTION = "emotion"       # 감정
    COGNITION = "cognition"   # 인지
    EVOLUTION = "evolution"   # 진화


@dataclass
class ContextualMemory:
    """맥락 기반 기억"""
    id: str
    content: str
    context_type: ContextType
    context_vector: Dict[str, float]  # 맥락 벡터
    semantic_meaning: str  # 의미적 의미
    relationships: List[str] = field(default_factory=list)
    evolution_trace: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ContextualQuery:
    """맥락 기반 쿼리"""
    query_text: str
    context_focus: ContextType
    semantic_intent: str
    relationship_scope: List[str] = field(default_factory=list)
    evolution_level: float = 0.5


class ContextualMemorySystem:
    """맥락 기반 기억 시스템"""
    
    def __init__(self, memory_graph: PalantirGraph):
        self.memory = memory_graph
        self.context_analyzer = ContextAnalyzer()
        self.semantic_processor = SemanticProcessor()
        self.relationship_mapper = RelationshipMapper()
        self.evolution_tracker = EvolutionTracker()
        
    def store_contextually(self, text: str, context_info: Dict[str, Any] = None) -> ContextualMemory:
        """맥락 기반 기억 저장"""
        # 1. 맥락 분석
        context_analysis = self.context_analyzer.analyze_context(text, context_info)
        
        # 2. 의미 처리
        semantic_meaning = self.semantic_processor.extract_meaning(text, context_analysis)
        
        # 3. 관계 매핑
        relationships = self.relationship_mapper.map_relationships(text, context_analysis)
        
        # 4. 진화 추적
        evolution_trace = self.evolution_tracker.track_evolution(text, context_analysis)
        
        # 5. 맥락 벡터 생성
        context_vector = self._create_context_vector(context_analysis, semantic_meaning, relationships)
        
        # 6. 맥락 기억 생성
        contextual_memory = ContextualMemory(
            id=str(uuid.uuid4()),
            content=text,
            context_type=context_analysis.primary_context,
            context_vector=context_vector,
            semantic_meaning=semantic_meaning,
            relationships=relationships,
            evolution_trace=evolution_trace
        )
        
        # 7. 메모리 그래프에 저장
        self._store_in_graph(contextual_memory)
        
        return contextual_memory
    
    def retrieve_contextually(self, query: ContextualQuery) -> List[ContextualMemory]:
        """맥락 기반 기억 검색"""
        # 1. 쿼리 맥락 분석
        query_context = self.context_analyzer.analyze_context(query.query_text)
        
        # 2. 맥락 기반 검색
        candidates = self._find_contextual_candidates(query_context, query)
        
        # 3. 의미적 유사도 계산
        scored_candidates = []
        for candidate in candidates:
            similarity = self._calculate_contextual_similarity(query_context, candidate)
            if similarity > 0.3:  # 임계값
                scored_candidates.append((similarity, candidate))
        
        # 4. 정렬 및 반환
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [candidate for _, candidate in scored_candidates[:10]]
    
    def evolve_memory(self, memory_id: str, new_context: Dict[str, Any]) -> ContextualMemory:
        """기억 진화"""
        # 기존 기억 찾기
        if memory_id not in self.memory.nodes:
            raise ValueError(f"기억을 찾을 수 없습니다: {memory_id}")
        
        node = self.memory.nodes[memory_id]
        
        # 새로운 맥락 정보로 진화
        evolution_result = self.evolution_tracker.evolve_memory(node, new_context)
        
        # 진화된 기억 업데이트
        node.meta["evolution_trace"] = evolution_result.evolution_trace
        node.meta["updated_at"] = datetime.now().isoformat()
        node.meta["evolution_count"] = node.meta.get("evolution_count", 0) + 1
        
        # 벡터 업데이트
        node.vectors.update(evolution_result.new_vectors)
        
        # 메모리 저장
        self.memory.save()
        
        return self._node_to_contextual_memory(node)
    
    def _create_context_vector(self, context_analysis: ContextAnalysis, 
                             semantic_meaning: str, relationships: List[str]) -> Dict[str, float]:
        """맥락 벡터 생성"""
        vector = {
            "T": 0.5,  # Topic
            "C": 0.5,  # Context
            "R": 0.5,  # Relationship
            "E": 0.5,  # Emotion
            "V": 0.5   # Evolution
        }
        
        # 주제 기반 조정
        if context_analysis.primary_context == ContextType.TOPIC:
            vector["T"] = 0.8
        
        # 상황 기반 조정
        if context_analysis.primary_context == ContextType.SITUATION:
            vector["C"] = 0.8
        
        # 관계 기반 조정
        if context_analysis.primary_context == ContextType.RELATIONSHIP:
            vector["R"] = 0.8
        
        # 감정 기반 조정
        if context_analysis.primary_context == ContextType.EMOTION:
            vector["E"] = 0.8
        
        # 진화 기반 조정
        if context_analysis.primary_context == ContextType.EVOLUTION:
            vector["V"] = 0.8
        
        # 관계 수에 따른 조정
        if relationships:
            vector["R"] = min(1.0, vector["R"] + len(relationships) * 0.1)
        
        return vector
    
    def _store_in_graph(self, contextual_memory: ContextualMemory):
        """그래프에 저장"""
        # 맥락 기억 노드 생성
        memory_node = ThoughtNode.create(
            content=contextual_memory.content,
            node_type="contextual_memory",
            vectors=contextual_memory.context_vector,
            meta={
                "memory_id": contextual_memory.id,
                "context_type": contextual_memory.context_type.value,
                "semantic_meaning": contextual_memory.semantic_meaning,
                "relationships": contextual_memory.relationships,
                "evolution_trace": contextual_memory.evolution_trace,
                "created_at": contextual_memory.created_at
            }
        )
        self.memory.add_node(memory_node)
        
        # 관계 노드들과 연결
        for relationship in contextual_memory.relationships:
            relation_node = self._get_or_create_relationship_node(relationship)
            relation_edge = Relationship.create(
                source=memory_node.id,
                target=relation_node.id,
                predicate="relates_to",
                weight=0.8
            )
            self.memory.add_edge(relation_edge)
        
        # 메모리 저장
        self.memory.save()
    
    def _get_or_create_relationship_node(self, relationship: str) -> ThoughtNode:
        """관계 노드 생성 또는 반환"""
        # 기존 관계 노드 찾기
        for node in self.memory.nodes.values():
            if (node.node_type == "relationship" and 
                node.content.lower() == relationship.lower()):
                return node
        
        # 새 관계 노드 생성
        relation_node = ThoughtNode.create(
            content=relationship,
            node_type="relationship",
            vectors={"R": 0.7},  # Relationship importance
            meta={"relationship_type": "contextual", "usage_count": 1}
        )
        self.memory.add_node(relation_node)
        return relation_node
    
    def _find_contextual_candidates(self, query_context: ContextAnalysis, 
                                  query: ContextualQuery) -> List[ContextualMemory]:
        """맥락 기반 후보 검색"""
        candidates = []
        
        for node in self.memory.nodes.values():
            if node.node_type != "contextual_memory":
                continue
            
            # 맥락 타입 매칭
            if query.context_focus == ContextType.TOPIC:
                if "topic" in node.meta.get("context_type", ""):
                    candidates.append(self._node_to_contextual_memory(node))
            elif query.context_focus == ContextType.SITUATION:
                if "situation" in node.meta.get("context_type", ""):
                    candidates.append(self._node_to_contextual_memory(node))
            elif query.context_focus == ContextType.RELATIONSHIP:
                if "relationship" in node.meta.get("context_type", ""):
                    candidates.append(self._node_to_contextual_memory(node))
            else:
                # 모든 맥락 타입 고려
                candidates.append(self._node_to_contextual_memory(node))
        
        return candidates
    
    def _calculate_contextual_similarity(self, query_context: ContextAnalysis, 
                                       candidate: ContextualMemory) -> float:
        """맥락적 유사도 계산"""
        similarity = 0.0
        
        # 맥락 타입 유사도
        if query_context.primary_context == candidate.context_type:
            similarity += 0.3
        
        # 의미적 유사도
        semantic_sim = self._calculate_semantic_similarity(
            query_context.semantic_summary, 
            candidate.semantic_meaning
        )
        similarity += semantic_sim * 0.4
        
        # 관계 유사도
        relationship_sim = self._calculate_relationship_similarity(
            query_context.relationships, 
            candidate.relationships
        )
        similarity += relationship_sim * 0.3
        
        return similarity
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산"""
        # 간단한 단어 기반 유사도 (실제로는 더 정교한 NLP 필요)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _calculate_relationship_similarity(self, rels1: List[str], rels2: List[str]) -> float:
        """관계 유사도 계산"""
        if not rels1 or not rels2:
            return 0.0
        
        set1 = set(rels1)
        set2 = set(rels2)
        
        intersection = set1 & set2
        union = set1 | set2
        
        return len(intersection) / len(union)
    
    def _node_to_contextual_memory(self, node: ThoughtNode) -> ContextualMemory:
        """노드를 맥락 기억으로 변환"""
        return ContextualMemory(
            id=node.id,
            content=node.content,
            context_type=ContextType(node.meta.get("context_type", "topic")),
            context_vector=node.vectors,
            semantic_meaning=node.meta.get("semantic_meaning", ""),
            relationships=node.meta.get("relationships", []),
            evolution_trace=node.meta.get("evolution_trace", []),
            created_at=node.meta.get("created_at", "")
        )


@dataclass
class ContextAnalysis:
    """맥락 분석 결과"""
    primary_context: ContextType
    semantic_summary: str
    relationships: List[str]
    emotional_context: str
    cognitive_level: float
    evolution_potential: float


class ContextAnalyzer:
    """맥락 분석기"""
    
    def analyze_context(self, text: str, context_info: Dict[str, Any] = None) -> ContextAnalysis:
        """맥락 분석"""
        # 주 맥락 타입 판별
        primary_context = self._identify_primary_context(text)
        
        # 의미적 요약
        semantic_summary = self._create_semantic_summary(text)
        
        # 관계 추출
        relationships = self._extract_relationships(text)
        
        # 감정적 맥락
        emotional_context = self._analyze_emotional_context(text)
        
        # 인지 수준
        cognitive_level = self._assess_cognitive_level(text)
        
        # 진화 잠재력
        evolution_potential = self._assess_evolution_potential(text)
        
        return ContextAnalysis(
            primary_context=primary_context,
            semantic_summary=semantic_summary,
            relationships=relationships,
            emotional_context=emotional_context,
            cognitive_level=cognitive_level,
            evolution_potential=evolution_potential
        )
    
    def _identify_primary_context(self, text: str) -> ContextType:
        """주 맥락 타입 식별"""
        text_lower = text.lower()
        
        # 주제 기반 판별
        topic_indicators = ["주제", "관련", "분야", "영역", "카테고리"]
        if any(indicator in text_lower for indicator in topic_indicators):
            return ContextType.TOPIC
        
        # 상황 기반 판별
        situation_indicators = ["상황", "환경", "조건", "상태", "맥락"]
        if any(indicator in text_lower for indicator in situation_indicators):
            return ContextType.SITUATION
        
        # 관계 기반 판별
        relationship_indicators = ["관계", "연결", "상호작용", "소통", "이해"]
        if any(indicator in text_lower for indicator in relationship_indicators):
            return ContextType.RELATIONSHIP
        
        # 감정 기반 판별
        emotion_indicators = ["느끼다", "생각하다", "감정", "마음", "기분"]
        if any(indicator in text_lower for indicator in emotion_indicators):
            return ContextType.EMOTION
        
        # 진화 기반 판별
        evolution_indicators = ["발전", "성장", "변화", "진화", "향상"]
        if any(indicator in text_lower for indicator in evolution_indicators):
            return ContextType.EVOLUTION
        
        return ContextType.TOPIC  # 기본값
    
    def _create_semantic_summary(self, text: str) -> str:
        """의미적 요약 생성"""
        # 간단한 요약 (실제로는 더 정교한 NLP 필요)
        sentences = text.split('.')
        if len(sentences) > 1:
            return sentences[0] + "."
        return text[:100] + "..." if len(text) > 100 else text
    
    def _extract_relationships(self, text: str) -> List[str]:
        """관계 추출"""
        relationships = []
        
        # 관계 패턴 매칭
        relationship_patterns = [
            r"(\w+)와\s+(\w+)의\s+관계",
            r"(\w+)가\s+(\w+)에\s+영향",
            r"(\w+)와\s+(\w+)의\s+연결",
            r"(\w+)가\s+(\w+)를\s+이해"
        ]
        
        for pattern in relationship_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    relationships.append(f"{match[0]}-{match[1]}")
                else:
                    relationships.append(match)
        
        return relationships[:5]  # 상위 5개만
    
    def _analyze_emotional_context(self, text: str) -> str:
        """감정적 맥락 분석"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["좋아", "행복", "즐거워"]):
            return "positive"
        elif any(word in text_lower for word in ["걱정", "불안", "어려워"]):
            return "concerned"
        elif any(word in text_lower for word in ["궁금", "알고싶어", "흥미"]):
            return "curious"
        else:
            return "neutral"
    
    def _assess_cognitive_level(self, text: str) -> float:
        """인지 수준 평가"""
        level = 0.5  # 기본값
        
        # 복잡한 개념이 많을수록 높은 수준
        complex_concepts = ["알고리즘", "데이터베이스", "아키텍처", "패턴", "원리"]
        for concept in complex_concepts:
            if concept in text:
                level += 0.1
        
        # 추상적 사고가 많을수록 높은 수준
        abstract_indicators = ["생각하다", "분석하다", "이해하다", "통찰"]
        for indicator in abstract_indicators:
            if indicator in text:
                level += 0.05
        
        return min(1.0, level)
    
    def _assess_evolution_potential(self, text: str) -> float:
        """진화 잠재력 평가"""
        potential = 0.3  # 기본값
        
        # 새로운 관점이나 이해가 있으면 높은 잠재력
        evolution_indicators = ["새롭게", "처음", "이제야", "변화", "발전"]
        for indicator in evolution_indicators:
            if indicator in text:
                potential += 0.2
        
        # 질문이나 탐구가 있으면 높은 잠재력
        if "?" in text or any(word in text for word in ["어떻게", "왜", "무엇"]):
            potential += 0.2
        
        return min(1.0, potential)


class SemanticProcessor:
    """의미 처리기"""
    
    def extract_meaning(self, text: str, context_analysis: ContextAnalysis) -> str:
        """의미 추출"""
        # 맥락 타입에 따른 의미 추출
        if context_analysis.primary_context == ContextType.TOPIC:
            return self._extract_topic_meaning(text)
        elif context_analysis.primary_context == ContextType.SITUATION:
            return self._extract_situation_meaning(text)
        elif context_analysis.primary_context == ContextType.RELATIONSHIP:
            return self._extract_relationship_meaning(text)
        elif context_analysis.primary_context == ContextType.EMOTION:
            return self._extract_emotional_meaning(text)
        elif context_analysis.primary_context == ContextType.EVOLUTION:
            return self._extract_evolution_meaning(text)
        else:
            return self._extract_general_meaning(text)
    
    def _extract_topic_meaning(self, text: str) -> str:
        """주제 의미 추출"""
        return f"주제 관련: {text[:50]}..."
    
    def _extract_situation_meaning(self, text: str) -> str:
        """상황 의미 추출"""
        return f"상황 인식: {text[:50]}..."
    
    def _extract_relationship_meaning(self, text: str) -> str:
        """관계 의미 추출"""
        return f"관계 이해: {text[:50]}..."
    
    def _extract_emotional_meaning(self, text: str) -> str:
        """감정 의미 추출"""
        return f"감정 체험: {text[:50]}..."
    
    def _extract_evolution_meaning(self, text: str) -> str:
        """진화 의미 추출"""
        return f"진화 과정: {text[:50]}..."
    
    def _extract_general_meaning(self, text: str) -> str:
        """일반 의미 추출"""
        return f"일반 내용: {text[:50]}..."


class RelationshipMapper:
    """관계 매퍼"""
    
    def map_relationships(self, text: str, context_analysis: ContextAnalysis) -> List[str]:
        """관계 매핑"""
        relationships = []
        
        # 명시적 관계
        explicit_rels = self._extract_explicit_relationships(text)
        relationships.extend(explicit_rels)
        
        # 암시적 관계
        implicit_rels = self._extract_implicit_relationships(text, context_analysis)
        relationships.extend(implicit_rels)
        
        return relationships[:10]  # 상위 10개만
    
    def _extract_explicit_relationships(self, text: str) -> List[str]:
        """명시적 관계 추출"""
        relationships = []
        
        # 관계 패턴 매칭
        patterns = [
            r"(\w+)와\s+(\w+)의\s+관계",
            r"(\w+)가\s+(\w+)에\s+영향",
            r"(\w+)와\s+(\w+)의\s+연결"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    relationships.append(f"{match[0]}-{match[1]}")
        
        return relationships
    
    def _extract_implicit_relationships(self, text: str, context_analysis: ContextAnalysis) -> List[str]:
        """암시적 관계 추출"""
        relationships = []
        
        # 맥락 기반 관계 추론
        if context_analysis.primary_context == ContextType.TOPIC:
            # 주제 관련 관계
            topics = ["기술", "학습", "문제해결", "개발", "분석"]
            for topic in topics:
                if topic in text:
                    relationships.append(f"주제-{topic}")
        
        elif context_analysis.primary_context == ContextType.SITUATION:
            # 상황 관련 관계
            situations = ["대화", "학습", "문제해결", "성찰"]
            for situation in situations:
                if situation in text:
                    relationships.append(f"상황-{situation}")
        
        return relationships


class EvolutionTracker:
    """진화 추적기"""
    
    def track_evolution(self, text: str, context_analysis: ContextAnalysis) -> List[str]:
        """진화 추적"""
        evolution_points = []
        
        # 새로운 이해
        if self._contains_new_understanding(text):
            evolution_points.append("새로운 이해 획득")
        
        # 관점 변화
        if self._contains_perspective_change(text):
            evolution_points.append("관점 변화")
        
        # 성장 인식
        if self._contains_growth_recognition(text):
            evolution_points.append("성장 인식")
        
        return evolution_points
    
    def evolve_memory(self, node: ThoughtNode, new_context: Dict[str, Any]) -> EvolutionResult:
        """기억 진화"""
        evolution_trace = node.meta.get("evolution_trace", [])
        
        # 새로운 진화 포인트 추가
        if "new_understanding" in new_context:
            evolution_trace.append(f"새로운 이해: {new_context['new_understanding']}")
        
        if "perspective_change" in new_context:
            evolution_trace.append(f"관점 변화: {new_context['perspective_change']}")
        
        # 벡터 업데이트
        new_vectors = node.vectors.copy()
        if "evolution_level" in new_context:
            new_vectors["V"] = min(1.0, new_vectors["V"] + new_context["evolution_level"] * 0.1)
        
        return EvolutionResult(
            evolution_trace=evolution_trace,
            new_vectors=new_vectors
        )
    
    def _contains_new_understanding(self, text: str) -> bool:
        """새로운 이해 포함 여부"""
        understanding_words = ["처음", "새롭게", "이제야", "비로소", "드디어"]
        return any(word in text for word in understanding_words)
    
    def _contains_perspective_change(self, text: str) -> bool:
        """관점 변화 포함 여부"""
        perspective_words = ["다른", "새로운", "변화", "달라", "이전과"]
        return any(word in text for word in perspective_words)
    
    def _contains_growth_recognition(self, text: str) -> bool:
        """성장 인식 포함 여부"""
        growth_words = ["성장", "발전", "향상", "나아지", "개선"]
        return any(word in text for word in growth_words)


@dataclass
class EvolutionResult:
    """진화 결과"""
    evolution_trace: List[str]
    new_vectors: Dict[str, float] 
