"""
Harin Core Perception System - LIDA Integration
자각(Perception) 시스템: 자극에서 패턴을 인식하고 의미를 해석하는 시스템
PMM의 sensation_evaluation을 참고하여 구현
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import re
import random

# 순환 import 방지를 위해 필요한 클래스들을 직접 정의
class StimulusType(Enum):
    """자극 유형들"""
    UserMessage = "UserMessage"
    SystemMessage = "SystemMessage"
    UserInactivity = "UserInactivity"
    TimeOfDayChange = "TimeOfDayChange"
    LowNeedTrigger = "LowNeedTrigger"
    WakeUp = "WakeUp"
    EngagementOpportunity = "EngagementOpportunity"

class Stimulus(BaseModel):
    """자극 - 사용자 입력이나 시스템 이벤트"""
    id: int = -1
    tick_id: int = -1
    content: str
    source: str = ""
    stimulus_type: StimulusType
    timestamp_creation: datetime = Field(default_factory=datetime.now)

class FeatureType(Enum):
    """특성 유형들"""
    Dialogue = "Dialogue"
    Feeling = "Feeling"
    SituationalModel = "SituationalModel"
    AttentionFocus = "AttentionFocus"
    ConsciousWorkspace = "ConsciousWorkspace"
    MemoryRecall = "MemoryRecall"
    SubjectiveExperience = "SubjectiveExperience"
    ActionSimulation = "ActionSimulation"
    ActionRating = "ActionRating"
    Action = "Action"
    ActionExpectation = "ActionExpectation"
    NarrativeUpdate = "NarrativeUpdate"
    ExpectationOutcome = "ExpectationOutcome"
    StoryWildcard = "StoryWildcard"
    Expectation = "Expectation"
    Goal = "Goal"
    Narrative = "Narrative"
    WorldEvent = "WorldEvent"
    Thought = "Thought"
    ExternalThought = "ExternalThought"
    MetaInsight = "MetaInsight"
    SystemMessage = "SystemMessage"

class Feature(BaseModel):
    """특성 - 스토리의 개별 인과적 사건"""
    id: int = -1
    tick_id: int = -1
    content: str
    feature_type: FeatureType
    source: str
    affective_valence: Optional[float] = None
    incentive_salience: Optional[float] = None
    interlocus: float = 0  # -1 내부, +1 외부, 0 혼합/중립
    causal: bool = False  # 스토리 생성에 영향을 미치는가?
    timestamp_creation: datetime = Field(default_factory=datetime.now)

# from core.stimulus_classifier import StimulusClassifier


class PatternType(Enum):
    """패턴 유형"""
    SEMANTIC = "semantic"      # 의미적 패턴
    TEMPORAL = "temporal"      # 시간적 패턴
    SPATIAL = "spatial"        # 공간적 패턴
    CAUSAL = "causal"          # 인과적 패턴
    EMOTIONAL = "emotional"    # 감정적 패턴
    BEHAVIORAL = "behavioral"  # 행동적 패턴


class PerceptionResult(BaseModel):
    """자각 결과"""
    stimulus_id: str
    pattern_type: PatternType
    confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_patterns: Dict[str, Any]
    interpretation: str
    emotional_impact: float = Field(..., ge=-1.0, le=1.0)
    urgency_level: float = Field(..., ge=0.0, le=1.0)
    complexity_score: float = Field(..., ge=0.0, le=1.0)
    novelty_score: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class SemanticPattern(BaseModel):
    """의미적 패턴"""
    keywords: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    intent: str = ""
    entities: List[str] = Field(default_factory=list)


class TemporalPattern(BaseModel):
    """시간적 패턴"""
    sequence_order: int = 0
    frequency: float = 0.0
    duration: Optional[float] = None
    timing: str = "immediate"  # immediate, delayed, periodic
    urgency: float = Field(..., ge=0.0, le=1.0)


class SpatialPattern(BaseModel):
    """공간적 패턴"""
    location_context: str = ""
    spatial_relationships: List[str] = Field(default_factory=list)
    proximity: float = 0.0
    direction: str = ""


class CausalPattern(BaseModel):
    """인과적 패턴"""
    cause_effect_pairs: List[Dict[str, str]] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    consequences: List[str] = Field(default_factory=list)


class PerceptionSystem:
    """자각 시스템 - LIDA의 Perception 단계 구현"""
    
    def __init__(self, stimulus_classifier: Optional[Any] = None):
        self.stimulus_classifier = stimulus_classifier
        self.pattern_recognition_history: List[PerceptionResult] = []
        
        # 패턴 인식 규칙 초기화
        self._initialize_pattern_rules()
        
    def _initialize_pattern_rules(self):
        """패턴 인식 규칙 초기화"""
        self.semantic_patterns = {
            "question_keywords": ["어떻게", "무엇", "왜", "언제", "어디", "누가", "뭐", "?"],
            "emotion_keywords": {
                "positive": ["기쁘", "좋", "감사", "사랑", "행복", "즐거", "만족", "희망"],
                "negative": ["화나", "슬프", "걱정", "불안", "미워", "싫", "실망", "두려움"],
                "neutral": ["보통", "평범", "일반", "중간", "보통"]
            },
            "action_keywords": ["하고", "하려", "할까", "해야", "필요", "원해"],
            "temporal_keywords": ["지금", "나중", "언제", "항상", "가끔", "자주", "이제", "곧"]
        }
        
        self.causal_patterns = {
            "cause_indicators": ["때문에", "덕분에", "결과로", "로 인해", "때문"],
            "effect_indicators": ["그래서", "따라서", "결과적으로", "결과", "효과"],
            "conditional_indicators": ["만약", "만일", "혹시", "만", "면", "면"]
        }
        
    def recognize_patterns(self, stimuli: Dict[str, Any]) -> Dict[PatternType, Any]:
        """자극에서 패턴 인식"""
        patterns = {}
        
        # 텍스트 자극이 있는 경우
        if 'textual' in stimuli and stimuli['textual']:
            text_content = stimuli['textual']
            
            # 의미적 패턴 추출
            semantic_pattern = self._extract_semantic_patterns(text_content)
            patterns[PatternType.SEMANTIC] = semantic_pattern
            
            # 시간적 패턴 추출
            temporal_pattern = self._extract_temporal_patterns(text_content)
            patterns[PatternType.TEMPORAL] = temporal_pattern
            
            # 인과적 패턴 추출
            causal_pattern = self._extract_causal_patterns(text_content)
            patterns[PatternType.CAUSAL] = causal_pattern
            
            # 감정적 패턴 추출
            emotional_pattern = self._extract_emotional_patterns(text_content)
            patterns[PatternType.EMOTIONAL] = emotional_pattern
        
        # 시각적 자극이 있는 경우
        if 'visual' in stimuli and stimuli['visual']:
            spatial_pattern = self._extract_spatial_patterns(stimuli['visual'])
            patterns[PatternType.SPATIAL] = spatial_pattern
        
        # 맥락적 자극이 있는 경우
        if 'contextual' in stimuli and stimuli['contextual']:
            contextual_pattern = self._extract_contextual_patterns(stimuli['contextual'])
            patterns[PatternType.BEHAVIORAL] = contextual_pattern
        
        return patterns
    
    def _extract_semantic_patterns(self, text: str) -> SemanticPattern:
        """의미적 패턴 추출"""
        text_lower = text.lower()
        
        # 키워드 추출
        keywords = []
        for category, words in self.semantic_patterns.items():
            if category != "emotion_keywords":
                for word in words:
                    if word in text_lower:
                        keywords.append(word)
        
        # 주제 추출
        topics = self._identify_topics(text)
        
        # 감정 분석
        sentiment = self._analyze_sentiment(text_lower)
        
        # 의도 추출
        intent = self._identify_intent(text_lower)
        
        # 개체 추출
        entities = self._extract_entities(text)
        
        return SemanticPattern(
            keywords=keywords,
            topics=topics,
            sentiment=sentiment,
            intent=intent,
            entities=entities
        )
    
    def _extract_temporal_patterns(self, text: str) -> TemporalPattern:
        """시간적 패턴 추출"""
        text_lower = text.lower()
        
        # 긴급도 분석
        urgency = 0.0
        if any(word in text_lower for word in ["지금", "바로", "즉시", "긴급", "빨리"]):
            urgency = 0.9
        elif any(word in text_lower for word in ["곧", "나중", "언제"]):
            urgency = 0.5
        else:
            urgency = 0.2
        
        # 타이밍 분석
        timing = "immediate"
        if any(word in text_lower for word in ["나중", "언제", "가끔"]):
            timing = "delayed"
        elif any(word in text_lower for word in ["항상", "자주", "매번"]):
            timing = "periodic"
        
        # 빈도 분석
        frequency = 0.0
        if "항상" in text_lower:
            frequency = 1.0
        elif "자주" in text_lower:
            frequency = 0.8
        elif "가끔" in text_lower:
            frequency = 0.4
        elif "거의" in text_lower:
            frequency = 0.1
        
        return TemporalPattern(
            urgency=urgency,
            timing=timing,
            frequency=frequency
        )
    
    def _extract_spatial_patterns(self, visual_data: Any) -> SpatialPattern:
        """공간적 패턴 추출"""
        # 시각적 데이터에서 공간적 관계 추출
        # 실제 구현에서는 이미지 분석, 위치 정보 등을 활용
        
        return SpatialPattern(
            location_context="current",
            spatial_relationships=[],
            proximity=0.5,
            direction="center"
        )
    
    def _extract_causal_patterns(self, text: str) -> CausalPattern:
        """인과적 패턴 추출"""
        text_lower = text.lower()
        
        cause_effect_pairs = []
        dependencies = []
        triggers = []
        consequences = []
        
        # 인과 관계 추출
        for cause_indicator in self.causal_patterns["cause_indicators"]:
            if cause_indicator in text_lower:
                # 간단한 인과 관계 추출 로직
                parts = text_lower.split(cause_indicator)
                if len(parts) >= 2:
                    cause_effect_pairs.append({
                        "cause": parts[0].strip(),
                        "effect": parts[1].strip()
                    })
        
        # 조건부 관계 추출
        for conditional_indicator in self.causal_patterns["conditional_indicators"]:
            if conditional_indicator in text_lower:
                triggers.append(conditional_indicator)
        
        return CausalPattern(
            cause_effect_pairs=cause_effect_pairs,
            dependencies=dependencies,
            triggers=triggers,
            consequences=consequences
        )
    
    def _extract_emotional_patterns(self, text: str) -> Dict[str, Any]:
        """감정적 패턴 추출"""
        text_lower = text.lower()
        
        emotional_intensity = 0.0
        emotional_valence = 0.0
        dominant_emotion = "neutral"
        
        # 감정 키워드 분석
        positive_count = 0
        negative_count = 0
        
        for emotion_type, keywords in self.semantic_patterns["emotion_keywords"].items():
            for keyword in keywords:
                if keyword in text_lower:
                    if emotion_type == "positive":
                        positive_count += 1
                        emotional_valence += 0.3
                    elif emotion_type == "negative":
                        negative_count += 1
                        emotional_valence -= 0.3
        
        # 감정 강도 계산
        total_emotion_words = positive_count + negative_count
        if total_emotion_words > 0:
            emotional_intensity = min(1.0, total_emotion_words * 0.2)
            if positive_count > negative_count:
                dominant_emotion = "positive"
            elif negative_count > positive_count:
                dominant_emotion = "negative"
        
        return {
            "intensity": emotional_intensity,
            "valence": max(-1.0, min(1.0, emotional_valence)),
            "dominant_emotion": dominant_emotion,
            "emotion_words": total_emotion_words
        }
    
    def _extract_contextual_patterns(self, contextual_data: Dict[str, Any]) -> Dict[str, Any]:
        """맥락적 패턴 추출"""
        # 시간, 위치, 상태 등의 맥락 정보에서 패턴 추출
        
        return {
            "time_context": contextual_data.get("time", "unknown"),
            "location_context": contextual_data.get("location", "unknown"),
            "user_state": contextual_data.get("user_state", "unknown"),
            "system_state": contextual_data.get("system_state", "unknown")
        }
    
    def interpret_meaning(self, patterns: Dict[PatternType, Any]) -> Dict[str, Any]:
        """패턴의 의미 해석"""
        interpretations = {}
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_type == PatternType.SEMANTIC:
                interpretations["semantic"] = self._interpret_semantic_meaning(pattern_data)
            elif pattern_type == PatternType.TEMPORAL:
                interpretations["temporal"] = self._interpret_temporal_meaning(pattern_data)
            elif pattern_type == PatternType.SPATIAL:
                interpretations["spatial"] = self._interpret_spatial_meaning(pattern_data)
            elif pattern_type == PatternType.CAUSAL:
                interpretations["causal"] = self._interpret_causal_meaning(pattern_data)
            elif pattern_type == PatternType.EMOTIONAL:
                interpretations["emotional"] = self._interpret_emotional_meaning(pattern_data)
            elif pattern_type == PatternType.BEHAVIORAL:
                interpretations["behavioral"] = self._interpret_behavioral_meaning(pattern_data)
        
        return interpretations
    
    def _interpret_semantic_meaning(self, semantic_pattern: SemanticPattern) -> Dict[str, Any]:
        """의미적 패턴 해석"""
        return {
            "primary_intent": semantic_pattern.intent,
            "sentiment_analysis": {
                "overall_sentiment": semantic_pattern.sentiment,
                "sentiment_strength": abs(semantic_pattern.sentiment)
            },
            "topic_analysis": {
                "main_topics": semantic_pattern.topics,
                "topic_count": len(semantic_pattern.topics)
            },
            "entity_analysis": {
                "entities": semantic_pattern.entities,
                "entity_count": len(semantic_pattern.entities)
            },
            "keyword_analysis": {
                "keywords": semantic_pattern.keywords,
                "keyword_count": len(semantic_pattern.keywords)
            }
        }
    
    def _interpret_temporal_meaning(self, temporal_pattern: TemporalPattern) -> Dict[str, Any]:
        """시간적 패턴 해석"""
        return {
            "urgency_assessment": {
                "level": temporal_pattern.urgency,
                "requires_immediate_attention": temporal_pattern.urgency > 0.7
            },
            "timing_analysis": {
                "timing_type": temporal_pattern.timing,
                "frequency": temporal_pattern.frequency
            },
            "temporal_priority": temporal_pattern.urgency * temporal_pattern.frequency
        }
    
    def _interpret_spatial_meaning(self, spatial_pattern: SpatialPattern) -> Dict[str, Any]:
        """공간적 패턴 해석"""
        return {
            "location_context": spatial_pattern.location_context,
            "spatial_relationships": spatial_pattern.spatial_relationships,
            "proximity_analysis": {
                "proximity_level": spatial_pattern.proximity,
                "is_close": spatial_pattern.proximity > 0.7
            }
        }
    
    def _interpret_causal_meaning(self, causal_pattern: CausalPattern) -> Dict[str, Any]:
        """인과적 패턴 해석"""
        return {
            "causal_relationships": {
                "cause_effect_pairs": len(causal_pattern.cause_effect_pairs),
                "dependencies": len(causal_pattern.dependencies)
            },
            "trigger_analysis": {
                "triggers": causal_pattern.triggers,
                "trigger_count": len(causal_pattern.triggers)
            },
            "consequence_analysis": {
                "consequences": causal_pattern.consequences,
                "consequence_count": len(causal_pattern.consequences)
            }
        }
    
    def _interpret_emotional_meaning(self, emotional_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """감정적 패턴 해석"""
        return {
            "emotional_state": {
                "intensity": emotional_pattern["intensity"],
                "valence": emotional_pattern["valence"],
                "dominant_emotion": emotional_pattern["dominant_emotion"]
            },
            "emotional_impact": {
                "requires_emotional_response": emotional_pattern["intensity"] > 0.5,
                "emotional_urgency": emotional_pattern["intensity"] * abs(emotional_pattern["valence"])
            }
        }
    
    def _interpret_behavioral_meaning(self, behavioral_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """행동적 패턴 해석"""
        return {
            "context_analysis": {
                "time_context": behavioral_pattern.get("time_context"),
                "location_context": behavioral_pattern.get("location_context"),
                "user_state": behavioral_pattern.get("user_state")
            },
            "behavioral_implications": {
                "context_awareness": True,
                "adaptive_response_needed": True
            }
        }
    
    def _identify_topics(self, text: str) -> List[str]:
        """주제 식별"""
        topics = []
        
        # 간단한 키워드 기반 주제 식별
        topic_keywords = {
            "일상": ["일상", "생활", "하루", "일과"],
            "감정": ["기분", "감정", "마음", "느낌"],
            "문제": ["문제", "고민", "어려움", "힘들"],
            "학습": ["배우", "공부", "학습", "지식"],
            "관계": ["친구", "가족", "사람", "관계"],
            "취미": ["취미", "관심", "좋아", "즐거"]
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _analyze_sentiment(self, text: str) -> float:
        """감정 분석"""
        positive_score = 0
        negative_score = 0
        
        for emotion_type, keywords in self.semantic_patterns["emotion_keywords"].items():
            for keyword in keywords:
                count = text.count(keyword)
                if emotion_type == "positive":
                    positive_score += count * 0.3
                elif emotion_type == "negative":
                    negative_score += count * 0.3
        
        if positive_score == 0 and negative_score == 0:
            return 0.0
        
        total_score = positive_score + negative_score
        sentiment = (positive_score - negative_score) / total_score
        
        return max(-1.0, min(1.0, sentiment))
    
    def _identify_intent(self, text: str) -> str:
        """의도 식별"""
        if any(word in text for word in self.semantic_patterns["question_keywords"]):
            return "question"
        elif any(word in text for word in self.semantic_patterns["action_keywords"]):
            return "action_request"
        elif any(word in text for word in self.semantic_patterns["emotion_keywords"]["positive"] + self.semantic_patterns["emotion_keywords"]["negative"]):
            return "emotional_expression"
        else:
            return "information_sharing"
    
    def _extract_entities(self, text: str) -> List[str]:
        """개체 추출"""
        entities = []
        
        # 간단한 개체 추출 (실제로는 NER 모델 사용)
        # 사람 이름, 장소, 시간 등
        
        return entities
    
    def process_stimulus(self, stimulus: Stimulus) -> PerceptionResult:
        """자극 처리 및 자각 결과 생성"""
        # 자극을 패턴 인식 형식으로 변환
        stimuli = {
            'textual': stimulus.content if stimulus.stimulus_type == StimulusType.UserMessage else None,
            'contextual': {
                'time': datetime.now().isoformat(),
                'stimulus_type': stimulus.stimulus_type.value,
                'user_state': 'active'
            }
        }
        
        # 패턴 인식
        patterns = self.recognize_patterns(stimuli)
        
        # 의미 해석
        interpretations = self.interpret_meaning(patterns)
        
        # 자각 결과 생성
        perception_result = PerceptionResult(
            stimulus_id=stimulus.id,
            pattern_type=PatternType.SEMANTIC,  # 기본값
            confidence=self._calculate_confidence(patterns),
            extracted_patterns=patterns,
            interpretation=self._generate_interpretation_summary(interpretations),
            emotional_impact=interpretations.get("emotional", {}).get("emotional_state", {}).get("valence", 0.0),
            urgency_level=interpretations.get("temporal", {}).get("urgency_assessment", {}).get("level", 0.0),
            complexity_score=self._calculate_complexity(patterns),
            novelty_score=self._calculate_novelty(patterns)
        )
        
        # 히스토리에 추가
        self.pattern_recognition_history.append(perception_result)
        
        return perception_result
    
    def _calculate_confidence(self, patterns: Dict[PatternType, Any]) -> float:
        """패턴 인식 신뢰도 계산"""
        if not patterns:
            return 0.0
        
        # 패턴의 품질과 일관성을 기반으로 신뢰도 계산
        confidence_scores = []
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_type == PatternType.SEMANTIC and hasattr(pattern_data, 'keywords'):
                # 키워드 수에 따른 신뢰도
                keyword_confidence = min(1.0, len(pattern_data.keywords) * 0.2)
                confidence_scores.append(keyword_confidence)
            
            elif pattern_type == PatternType.EMOTIONAL:
                # 감정 강도에 따른 신뢰도
                emotional_confidence = pattern_data.get("intensity", 0.0)
                confidence_scores.append(emotional_confidence)
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
    
    def _calculate_complexity(self, patterns: Dict[PatternType, Any]) -> float:
        """복잡도 계산"""
        complexity = 0.0
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_type == PatternType.SEMANTIC and hasattr(pattern_data, 'keywords'):
                complexity += len(pattern_data.keywords) * 0.1
            elif pattern_type == PatternType.CAUSAL and hasattr(pattern_data, 'cause_effect_pairs'):
                complexity += len(pattern_data.cause_effect_pairs) * 0.2
        
        return min(1.0, complexity)
    
    def _calculate_novelty(self, patterns: Dict[PatternType, Any]) -> float:
        """새로움 계산"""
        # 히스토리와 비교하여 새로움 계산
        if not self.pattern_recognition_history:
            return 0.8  # 첫 번째 패턴은 높은 새로움
        
        # 간단한 새로움 계산 (실제로는 더 정교한 로직 필요)
        return random.uniform(0.3, 0.7)
    
    def _generate_interpretation_summary(self, interpretations: Dict[str, Any]) -> str:
        """해석 요약 생성"""
        summary_parts = []
        
        if "semantic" in interpretations:
            semantic = interpretations["semantic"]
            summary_parts.append(f"의도: {semantic.get('primary_intent', 'unknown')}")
        
        if "emotional" in interpretations:
            emotional = interpretations["emotional"]
            emotion_state = emotional.get("emotional_state", {})
            summary_parts.append(f"감정: {emotion_state.get('dominant_emotion', 'neutral')}")
        
        if "temporal" in interpretations:
            temporal = interpretations["temporal"]
            urgency = temporal.get("urgency_assessment", {})
            summary_parts.append(f"긴급도: {urgency.get('level', 0.0):.2f}")
        
        return "; ".join(summary_parts) if summary_parts else "패턴 인식 완료"
    
    def get_perception_history(self, limit: int = 10) -> List[PerceptionResult]:
        """자각 히스토리 조회"""
        return self.pattern_recognition_history[-limit:]
    
    def get_perception_statistics(self) -> Dict[str, Any]:
        """자각 통계 조회"""
        if not self.pattern_recognition_history:
            return {}
        
        total_patterns = len(self.pattern_recognition_history)
        avg_confidence = sum(p.confidence for p in self.pattern_recognition_history) / total_patterns
        avg_emotional_impact = sum(p.emotional_impact for p in self.pattern_recognition_history) / total_patterns
        
        return {
            "total_patterns_recognized": total_patterns,
            "average_confidence": avg_confidence,
            "average_emotional_impact": avg_emotional_impact,
            "pattern_types_distribution": self._get_pattern_type_distribution()
        }
    
    def _get_pattern_type_distribution(self) -> Dict[str, int]:
        """패턴 유형 분포"""
        distribution = {}
        for result in self.pattern_recognition_history:
            pattern_type = result.pattern_type.value
            distribution[pattern_type] = distribution.get(pattern_type, 0) + 1
        return distribution 