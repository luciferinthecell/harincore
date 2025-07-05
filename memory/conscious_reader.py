"""
harin.memory.conscious_reader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

인간처럼 정독하면서 상황을 인식하는 기억 시스템
- 스캔이 아닌 정독 방식
- 상황 인식 기반 기억
- 자기 진화하는 기억 구조
- 맥락 기반 의미 추출
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from memory.palantirgraph import PalantirGraph, ThoughtNode, Relationship


class ReadingMode(Enum):
    """읽기 모드"""
    SKIM = "skim"           # 빠른 훑어보기
    READ = "read"           # 일반 읽기
    STUDY = "study"         # 정독/학습
    REFLECT = "reflect"     # 반성/성찰
    INTEGRATE = "integrate" # 통합/융합


class MemoryType(Enum):
    """기억 타입"""
    EXPERIENCE = "experience"    # 경험
    KNOWLEDGE = "knowledge"      # 지식
    INSIGHT = "insight"          # 통찰
    EMOTION = "emotion"          # 감정
    RELATIONSHIP = "relationship" # 관계
    CONTEXT = "context"          # 맥락
    EVOLUTION = "evolution"      # 진화


@dataclass
class ReadingContext:
    """읽기 컨텍스트"""
    mode: ReadingMode = ReadingMode.READ
    focus_areas: List[str] = field(default_factory=list)
    emotional_state: str = "neutral"
    cognitive_load: float = 0.5  # 0.0 ~ 1.0
    attention_span: float = 0.8  # 0.0 ~ 1.0
    integration_level: float = 0.6  # 0.0 ~ 1.0


@dataclass
class SituationalMemory:
    """상황 인식 기억"""
    id: str
    situation_type: str  # "conversation", "learning", "reflection", "problem_solving"
    context_summary: str
    key_participants: List[str] = field(default_factory=list)
    emotional_atmosphere: str = "neutral"
    cognitive_challenge: float = 0.5
    learning_outcome: str = ""
    personal_growth: str = ""
    related_concepts: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConsciousReadingResult:
    """정독 결과"""
    original_text: str
    situational_memories: List[SituationalMemory]
    emotional_journey: List[Dict[str, Any]]
    cognitive_insights: List[str]
    personal_evolution: List[str]
    integrated_knowledge: Dict[str, Any]
    reading_quality: float  # 0.0 ~ 1.0


class ConsciousReader:
    """의식적 정독 시스템"""
    
    def __init__(self, memory_graph: PalantirGraph):
        self.memory = memory_graph
        self.situation_analyzer = SituationAnalyzer()
        self.emotion_tracker = EmotionTracker()
        self.cognitive_processor = CognitiveProcessor()
        self.evolution_tracker = EvolutionTracker()
        
    def read_consciously(self, text: str, context: ReadingContext = None) -> ConsciousReadingResult:
        """의식적으로 정독"""
        if context is None:
            context = ReadingContext()
        
        # 1. 텍스트를 의미 단위로 분할
        segments = self._segment_by_meaning(text)
        
        # 2. 각 세그먼트를 정독하면서 상황 인식
        situational_memories = []
        emotional_journey = []
        cognitive_insights = []
        personal_evolution = []
        
        for segment in segments:
            # 세그먼트별 정독
            segment_result = self._read_segment_consciously(segment, context)
            
            if segment_result.situational_memory:
                situational_memories.append(segment_result.situational_memory)
            
            emotional_journey.extend(segment_result.emotional_states)
            cognitive_insights.extend(segment_result.insights)
            personal_evolution.extend(segment_result.evolution_points)
            
            # 컨텍스트 업데이트 (진화)
            context = self._evolve_context(context, segment_result)
        
        # 3. 전체 통합 및 진화
        integrated_knowledge = self._integrate_knowledge(situational_memories, cognitive_insights)
        reading_quality = self._assess_reading_quality(situational_memories, emotional_journey, cognitive_insights)
        
        # 4. 기억에 저장
        self._store_conscious_reading(situational_memories, integrated_knowledge)
        
        return ConsciousReadingResult(
            original_text=text,
            situational_memories=situational_memories,
            emotional_journey=emotional_journey,
            cognitive_insights=cognitive_insights,
            personal_evolution=personal_evolution,
            integrated_knowledge=integrated_knowledge,
            reading_quality=reading_quality
        )
    
    def _segment_by_meaning(self, text: str) -> List[str]:
        """의미 단위로 텍스트 분할"""
        # 단순한 문장 단위 분할 (실제로는 더 정교한 의미 분석 필요)
        sentences = re.split(r'[.!?]+', text)
        segments = []
        
        current_segment = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 의미적 연결성 확인
            if self._is_meaningfully_connected(current_segment, sentence):
                current_segment += " " + sentence if current_segment else sentence
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = sentence
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _is_meaningfully_connected(self, prev: str, current: str) -> bool:
        """의미적 연결성 확인"""
        if not prev:
            return True
        
        # 간단한 연결성 체크 (실제로는 더 정교한 분석 필요)
        prev_words = set(prev.lower().split())
        current_words = set(current.lower().split())
        
        # 공통 단어가 있거나, 대명사/연결어가 있으면 연결된 것으로 판단
        common_words = prev_words & current_words
        if len(common_words) > 0:
            return True
        
        # 연결어 체크
        connectors = ["그리고", "또한", "또는", "하지만", "그런데", "따라서", "그래서", "이제", "그러면"]
        if any(connector in current for connector in connectors):
            return True
        
        return False
    
    def _read_segment_consciously(self, segment: str, context: ReadingContext) -> SegmentReadingResult:
        """세그먼트 정독"""
        # 1. 상황 분석
        situation = self.situation_analyzer.analyze(segment, context)
        
        # 2. 감정 추적
        emotional_states = self.emotion_tracker.track_emotions(segment, context.emotional_state)
        
        # 3. 인지 처리
        insights = self.cognitive_processor.process(segment, context)
        
        # 4. 진화 포인트 식별
        evolution_points = self.evolution_tracker.identify_evolution(segment, context)
        
        # 5. 상황 기억 생성
        situational_memory = None
        if situation.is_significant:
            situational_memory = SituationalMemory(
                id=str(uuid.uuid4()),
                situation_type=situation.type,
                context_summary=situation.summary,
                key_participants=situation.participants,
                emotional_atmosphere=emotional_states[-1]["emotion"] if emotional_states else "neutral",
                cognitive_challenge=situation.complexity,
                learning_outcome=", ".join(insights) if insights else "",
                personal_growth=", ".join(evolution_points) if evolution_points else "",
                related_concepts=situation.concepts
            )
        
        return SegmentReadingResult(
            situational_memory=situational_memory,
            emotional_states=emotional_states,
            insights=insights,
            evolution_points=evolution_points
        )
    
    def _evolve_context(self, context: ReadingContext, result: SegmentReadingResult) -> ReadingContext:
        """컨텍스트 진화"""
        # 감정 상태 진화
        if result.emotional_states:
            latest_emotion = result.emotional_states[-1]["emotion"]
            context.emotional_state = latest_emotion
        
        # 인지 부하 조정
        if result.insights:
            context.cognitive_load = min(1.0, context.cognitive_load + 0.1)
        else:
            context.cognitive_load = max(0.0, context.cognitive_load - 0.05)
        
        # 주의력 조정
        if result.evolution_points:
            context.attention_span = min(1.0, context.attention_span + 0.1)
        else:
            context.attention_span = max(0.3, context.attention_span - 0.02)
        
        # 통합 수준 진화
        if result.insights or result.evolution_points:
            context.integration_level = min(1.0, context.integration_level + 0.05)
        
        return context
    
    def _integrate_knowledge(self, memories: List[SituationalMemory], insights: List[str]) -> Dict[str, Any]:
        """지식 통합"""
        integration = {
            "core_concepts": set(),
            "emotional_patterns": {},
            "learning_themes": [],
            "growth_areas": [],
            "relationship_insights": []
        }
        
        # 핵심 개념 추출
        for memory in memories:
            integration["core_concepts"].update(memory.related_concepts)
        
        # 감정 패턴 분석
        emotion_counts = {}
        for memory in memories:
            emotion = memory.emotional_atmosphere
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        integration["emotional_patterns"] = emotion_counts
        
        # 학습 테마 추출
        for insight in insights:
            if "학습" in insight or "알게" in insight:
                integration["learning_themes"].append(insight)
        
        # 성장 영역 식별
        for memory in memories:
            if memory.personal_growth:
                integration["growth_areas"].append(memory.personal_growth)
        
        return integration
    
    def _assess_reading_quality(self, memories: List[SituationalMemory], 
                              emotions: List[Dict[str, Any]], insights: List[str]) -> float:
        """읽기 품질 평가"""
        quality = 0.5  # 기본값
        
        # 기억의 깊이
        if memories:
            avg_complexity = sum(m.cognitive_challenge for m in memories) / len(memories)
            quality += avg_complexity * 0.2
        
        # 감정적 참여도
        if emotions:
            emotion_diversity = len(set(e["emotion"] for e in emotions))
            quality += min(0.2, emotion_diversity * 0.05)
        
        # 통찰의 수
        quality += min(0.2, len(insights) * 0.05)
        
        return min(1.0, quality)
    
    def _store_conscious_reading(self, memories: List[SituationalMemory], 
                               integrated_knowledge: Dict[str, Any]):
        """의식적 읽기 결과 저장"""
        # 상황 기억들을 노드로 저장
        for memory in memories:
            memory_node = ThoughtNode.create(
                content=memory.context_summary,
                node_type="situational_memory",
                vectors={
                    "T": memory.cognitive_challenge,  # Topic complexity
                    "C": 0.8,  # Context richness
                    "I": 0.9,  # Insight value
                    "E": self._emotion_to_vector(memory.emotional_atmosphere),
                    "M": 0.8   # Memory importance
                },
                meta={
                    "memory_id": memory.id,
                    "situation_type": memory.situation_type,
                    "participants": memory.key_participants,
                    "emotional_atmosphere": memory.emotional_atmosphere,
                    "cognitive_challenge": memory.cognitive_challenge,
                    "learning_outcome": memory.learning_outcome,
                    "personal_growth": memory.personal_growth,
                    "related_concepts": memory.related_concepts,
                    "created_at": memory.created_at
                }
            )
            self.memory.add_node(memory_node)
        
        # 통합 지식을 메타 노드로 저장
        if integrated_knowledge["core_concepts"]:
            integration_node = ThoughtNode.create(
                content=f"통합 지식: {', '.join(list(integrated_knowledge['core_concepts'])[:5])}",
                node_type="knowledge_integration",
                vectors={"M": 0.9},  # High memory importance
                meta={
                    "integration_type": "conscious_reading",
                    "core_concepts": list(integrated_knowledge["core_concepts"]),
                    "emotional_patterns": integrated_knowledge["emotional_patterns"],
                    "learning_themes": integrated_knowledge["learning_themes"],
                    "growth_areas": integrated_knowledge["growth_areas"],
                    "created_at": datetime.now().isoformat()
                }
            )
            self.memory.add_node(integration_node)
        
        # 메모리 저장
        self.memory.save()


@dataclass
class SegmentReadingResult:
    """세그먼트 읽기 결과"""
    situational_memory: Optional[SituationalMemory]
    emotional_states: List[Dict[str, Any]]
    insights: List[str]
    evolution_points: List[str]


@dataclass
class SituationAnalysis:
    """상황 분석 결과"""
    type: str
    summary: str
    participants: List[str]
    complexity: float
    concepts: List[str]
    is_significant: bool


class SituationAnalyzer:
    """상황 분석기"""
    
    def analyze(self, text: str, context: ReadingContext) -> SituationAnalysis:
        """상황 분석"""
        # 상황 타입 판별
        situation_type = self._identify_situation_type(text)
        
        # 참여자 식별
        participants = self._identify_participants(text)
        
        # 복잡도 계산
        complexity = self._calculate_complexity(text, context)
        
        # 개념 추출
        concepts = self._extract_concepts(text)
        
        # 중요도 판단
        is_significant = self._assess_significance(text, complexity, concepts)
        
        return SituationAnalysis(
            type=situation_type,
            summary=text[:100] + "..." if len(text) > 100 else text,
            participants=participants,
            complexity=complexity,
            concepts=concepts,
            is_significant=is_significant
        )
    
    def _identify_situation_type(self, text: str) -> str:
        """상황 타입 식별"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["대화", "말씀", "질문", "답변"]):
            return "conversation"
        elif any(word in text_lower for word in ["학습", "배우", "알게", "이해"]):
            return "learning"
        elif any(word in text_lower for word in ["생각", "반성", "성찰", "느낌"]):
            return "reflection"
        elif any(word in text_lower for word in ["문제", "해결", "방법", "해결책"]):
            return "problem_solving"
        else:
            return "general"
    
    def _identify_participants(self, text: str) -> List[str]:
        """참여자 식별"""
        participants = []
        
        # 간단한 패턴 매칭
        if "사용자" in text or "User" in text:
            participants.append("사용자")
        if "어시스턴트" in text or "Assistant" in text or "GPT" in text:
            participants.append("어시스턴트")
        
        return participants
    
    def _calculate_complexity(self, text: str, context: ReadingContext) -> float:
        """복잡도 계산"""
        complexity = 0.3  # 기본값
        
        # 문장 길이
        if len(text) > 100:
            complexity += 0.2
        
        # 개념 수
        concepts = self._extract_concepts(text)
        complexity += len(concepts) * 0.05
        
        # 감정적 강도
        if any(word in text for word in ["중요", "핵심", "필요", "꼭"]):
            complexity += 0.1
        
        return min(1.0, complexity)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """개념 추출 (키워드 매칭이 아닌 맥락 기반)"""
        concepts = []
        
        # 맥락 기반 개념 추출 (실제로는 더 정교한 NLP 필요)
        context_patterns = {
            "기술": ["프로그래밍", "코딩", "개발", "알고리즘", "데이터베이스"],
            "학습": ["배우다", "이해하다", "알다", "학습", "교육"],
            "문제해결": ["문제", "해결", "방법", "해결책", "접근"],
            "감정": ["느끼다", "생각하다", "감사하다", "궁금하다", "걱정하다"]
        }
        
        for concept, keywords in context_patterns.items():
            if any(keyword in text for keyword in keywords):
                concepts.append(concept)
        
        return concepts
    
    def _assess_significance(self, text: str, complexity: float, concepts: List[str]) -> bool:
        """중요도 평가"""
        # 복잡도가 높거나 개념이 많으면 중요
        if complexity > 0.6 or len(concepts) > 2:
            return True
        
        # 특정 키워드가 있으면 중요
        significant_words = ["중요", "핵심", "필요", "꼭", "반드시", "확실히"]
        if any(word in text for word in significant_words):
            return True
        
        return False


class EmotionTracker:
    """감정 추적기"""
    
    def track_emotions(self, text: str, current_state: str) -> List[Dict[str, Any]]:
        """감정 추적"""
        emotions = []
        
        # 감정 변화 감지
        detected_emotion = self._detect_emotion(text)
        
        if detected_emotion != current_state:
            emotions.append({
                "emotion": detected_emotion,
                "intensity": self._calculate_intensity(text),
                "trigger": self._identify_trigger(text),
                "timestamp": datetime.now().isoformat()
            })
        
        return emotions
    
    def _detect_emotion(self, text: str) -> str:
        """감정 감지"""
        text_lower = text.lower()
        
        emotion_patterns = {
            "happy": ["좋아", "행복", "즐거워", "재미있어", "감사해"],
            "excited": ["흥미", "신기", "놀라워", "대단해", "멋져"],
            "curious": ["궁금", "알고싶어", "어떻게", "왜", "무엇"],
            "concerned": ["걱정", "염려", "불안", "어려워", "힘들어"],
            "sad": ["슬퍼", "우울", "실망", "아쉬워", "후회"],
            "angry": ["화나", "짜증", "분노", "열받", "불만"]
        }
        
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return "neutral"
    
    def _calculate_intensity(self, text: str) -> float:
        """감정 강도 계산"""
        intensity = 0.5  # 기본값
        
        # 강조어 체크
        intensifiers = ["매우", "정말", "너무", "완전히", "절대적으로"]
        if any(word in text for word in intensifiers):
            intensity += 0.3
        
        # 반복 체크
        if "!" in text:
            intensity += 0.1
        
        return min(1.0, intensity)
    
    def _identify_trigger(self, text: str) -> str:
        """감정 트리거 식별"""
        # 간단한 트리거 식별
        if "감사" in text:
            return "appreciation"
        elif "궁금" in text:
            return "curiosity"
        elif "걱정" in text:
            return "concern"
        else:
            return "general"


class CognitiveProcessor:
    """인지 처리기"""
    
    def process(self, text: str, context: ReadingContext) -> List[str]:
        """인지 처리"""
        insights = []
        
        # 학습 인사이트
        if self._contains_learning(text):
            insights.append(self._extract_learning_insight(text))
        
        # 문제 해결 인사이트
        if self._contains_problem_solving(text):
            insights.append(self._extract_problem_insight(text))
        
        # 관계 인사이트
        if self._contains_relationship(text):
            insights.append(self._extract_relationship_insight(text))
        
        return [insight for insight in insights if insight]
    
    def _contains_learning(self, text: str) -> bool:
        """학습 내용 포함 여부"""
        learning_words = ["배우", "알게", "이해", "학습", "알다"]
        return any(word in text for word in learning_words)
    
    def _contains_problem_solving(self, text: str) -> bool:
        """문제 해결 내용 포함 여부"""
        problem_words = ["문제", "해결", "방법", "해결책", "접근"]
        return any(word in text for word in problem_words)
    
    def _contains_relationship(self, text: str) -> bool:
        """관계 내용 포함 여부"""
        relationship_words = ["관계", "연결", "상호작용", "소통", "이해"]
        return any(word in text for word in relationship_words)
    
    def _extract_learning_insight(self, text: str) -> str:
        """학습 인사이트 추출"""
        return f"학습 인사이트: {text[:50]}..."
    
    def _extract_problem_insight(self, text: str) -> str:
        """문제 해결 인사이트 추출"""
        return f"문제 해결 인사이트: {text[:50]}..."
    
    def _extract_relationship_insight(self, text: str) -> str:
        """관계 인사이트 추출"""
        return f"관계 인사이트: {text[:50]}..."


class EvolutionTracker:
    """진화 추적기"""
    
    def identify_evolution(self, text: str, context: ReadingContext) -> List[str]:
        """진화 포인트 식별"""
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


def _emotion_to_vector(emotion: str) -> float:
    """감정을 벡터 값으로 변환"""
    emotion_map = {
        "neutral": 0.5,
        "happy": 0.8,
        "excited": 0.9,
        "curious": 0.7,
        "concerned": 0.3,
        "sad": 0.2,
        "angry": 0.1
    }
    return emotion_map.get(emotion, 0.5) 
