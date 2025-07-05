"""
하린코어 자극 분류 시스템
PM 시스템의 자극 분류 기능을 참고하여 하린코어에 맞게 구현한 고급 자극 분류 시스템
"""

import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
try:
    from enum import StrEnum
except ImportError:
    import enum
    class StrEnum(str, enum.Enum):
        pass
from pydantic import BaseModel, Field

from memory.models import Stimulus, StimulusType, StimulusTriage, NeedsAxesModel, EmotionalAxesModel
from core.enhanced_main_loop import EnhancedHarinMainLoop


class StimulusPriority(StrEnum):
    """자극 우선순위"""
    Critical = "Critical"      # 즉시 처리 필요 (긴급 상황, 사용자 입력)
    High = "High"             # 높은 우선순위 (중요한 시스템 이벤트)
    Medium = "Medium"         # 중간 우선순위 (일반적인 상호작용)
    Low = "Low"               # 낮은 우선순위 (백그라운드 처리 가능)
    Background = "Background" # 백그라운드 처리 (메모리 최적화 등)


class StimulusCategory(StrEnum):
    """자극 카테고리"""
    UserInteraction = "UserInteraction"     # 사용자 상호작용
    SystemEvent = "SystemEvent"            # 시스템 이벤트
    InternalState = "InternalState"        # 내부 상태 변화
    Environmental = "Environmental"        # 환경 변화
    Maintenance = "Maintenance"            # 시스템 유지보수


class ProcessingMode(StrEnum):
    """처리 모드"""
    Immediate = "Immediate"           # 즉시 처리
    Queued = "Queued"                # 큐에 추가
    Background = "Background"        # 백그라운드 처리
    Deferred = "Deferred"            # 지연 처리
    Ignored = "Ignored"              # 무시


@dataclass
class StimulusAnalysis:
    """자극 분석 결과"""
    priority: StimulusPriority
    category: StimulusCategory
    triage: StimulusTriage
    processing_mode: ProcessingMode
    urgency_score: float
    complexity_score: float
    emotional_impact: float
    needs_impact: Dict[str, float]
    processing_timeout: float
    requires_attention: bool
    can_be_batched: bool
    analysis_confidence: float


class StimulusClassifier:
    """하린코어 자극 분류기 - PM 시스템의 자극 분류 기능을 참고하여 구현"""
    
    def __init__(self, harin_main_loop: EnhancedHarinMainLoop):
        self.harin = harin_main_loop
        
        # 분류 규칙
        self.classification_rules = self._initialize_classification_rules()
        
        # 패턴 매칭
        self.patterns = self._initialize_patterns()
        
        # 통계 추적
        self.classification_stats = {
            "total_processed": 0,
            "by_priority": {priority.value: 0 for priority in StimulusPriority},
            "by_category": {category.value: 0 for category in StimulusCategory},
            "by_triage": {triage.value: 0 for triage in StimulusTriage}
        }
    
    def _initialize_classification_rules(self) -> Dict[str, Any]:
        """분류 규칙 초기화"""
        return {
            "priority_rules": {
                StimulusType.UserMessage: StimulusPriority.Critical,
                StimulusType.SystemMessage: StimulusPriority.High,
                StimulusType.UserInactivity: StimulusPriority.Medium,
                StimulusType.TimeOfDayChange: StimulusPriority.Low,
                StimulusType.LowNeedTrigger: StimulusPriority.High,
                StimulusType.WakeUp: StimulusPriority.High,
                StimulusType.EngagementOpportunity: StimulusPriority.Medium
            },
            "category_rules": {
                StimulusType.UserMessage: StimulusCategory.UserInteraction,
                StimulusType.SystemMessage: StimulusCategory.SystemEvent,
                StimulusType.UserInactivity: StimulusCategory.Environmental,
                StimulusType.TimeOfDayChange: StimulusCategory.Environmental,
                StimulusType.LowNeedTrigger: StimulusCategory.InternalState,
                StimulusType.WakeUp: StimulusCategory.SystemEvent,
                StimulusType.EngagementOpportunity: StimulusCategory.UserInteraction
            },
            "triage_rules": {
                StimulusType.UserMessage: StimulusTriage.Significant,
                StimulusType.SystemMessage: StimulusTriage.Moderate,
                StimulusType.UserInactivity: StimulusTriage.Moderate,
                StimulusType.TimeOfDayChange: StimulusTriage.Insignificant,
                StimulusType.LowNeedTrigger: StimulusTriage.Significant,
                StimulusType.WakeUp: StimulusTriage.Moderate,
                StimulusType.EngagementOpportunity: StimulusTriage.Moderate
            }
        }
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """패턴 매칭 초기화"""
        return {
            "urgent_keywords": [
                "긴급", "즉시", "바로", "당장", "중요", "위험", "도움", "필요",
                "help", "urgent", "important", "critical", "emergency"
            ],
            "emotional_keywords": [
                "화나", "슬프", "기쁘", "걱정", "불안", "감사", "사랑", "미워",
                "angry", "sad", "happy", "worried", "anxious", "thankful", "love", "hate"
            ],
            "complex_keywords": [
                "분석", "설명", "이해", "학습", "연구", "계획", "전략",
                "analyze", "explain", "understand", "learn", "research", "plan", "strategy"
            ],
            "simple_keywords": [
                "안녕", "고마워", "좋아", "싫어", "네", "아니요",
                "hello", "thanks", "good", "bad", "yes", "no"
            ]
        }
    
    def classify_stimulus(self, stimulus: Stimulus) -> StimulusAnalysis:
        """자극 분류"""
        print(f"자극 분류 시작: {stimulus.stimulus_type.value} - '{stimulus.content[:50]}...'")
        
        # 기본 분류
        priority = self._determine_priority(stimulus)
        category = self._determine_category(stimulus)
        triage = self._determine_triage(stimulus)
        
        # 세부 분석
        urgency_score = self._calculate_urgency_score(stimulus)
        complexity_score = self._calculate_complexity_score(stimulus)
        emotional_impact = self._calculate_emotional_impact(stimulus)
        needs_impact = self._calculate_needs_impact(stimulus)
        
        # 처리 모드 결정
        processing_mode = self._determine_processing_mode(
            priority, triage, urgency_score, complexity_score
        )
        
        # 처리 타임아웃 설정
        processing_timeout = self._calculate_processing_timeout(
            priority, complexity_score, triage
        )
        
        # 추가 속성
        requires_attention = self._requires_attention(stimulus, priority, triage)
        can_be_batched = self._can_be_batched(stimulus, priority, category)
        analysis_confidence = self._calculate_confidence(stimulus, priority, triage)
        
        # 통계 업데이트
        self._update_stats(priority, category, triage)
        
        analysis = StimulusAnalysis(
            priority=priority,
            category=category,
            triage=triage,
            processing_mode=processing_mode,
            urgency_score=urgency_score,
            complexity_score=complexity_score,
            emotional_impact=emotional_impact,
            needs_impact=needs_impact,
            processing_timeout=processing_timeout,
            requires_attention=requires_attention,
            can_be_batched=can_be_batched,
            analysis_confidence=analysis_confidence
        )
        
        print(f"자극 분류 완료: {priority.value} 우선순위, {triage.value} 분류")
        return analysis
    
    def _determine_priority(self, stimulus: Stimulus) -> StimulusPriority:
        """우선순위 결정"""
        # 기본 규칙 적용
        base_priority = self.classification_rules["priority_rules"].get(
            stimulus.stimulus_type, StimulusPriority.Medium
        )
        
        # 내용 기반 조정
        content_lower = stimulus.content.lower()
        
        # 긴급 키워드가 있으면 우선순위 상승
        if any(keyword in content_lower for keyword in self.patterns["urgent_keywords"]):
            if base_priority == StimulusPriority.Low:
                return StimulusPriority.Medium
            elif base_priority == StimulusPriority.Medium:
                return StimulusPriority.High
            elif base_priority == StimulusPriority.High:
                return StimulusPriority.Critical
        
        # 사용자 메시지의 경우 길이와 복잡도 고려
        if stimulus.stimulus_type == StimulusType.UserMessage:
            if len(stimulus.content) > 200:
                return StimulusPriority.High
            elif len(stimulus.content) < 20:
                return StimulusPriority.Medium
        
        return base_priority
    
    def _determine_category(self, stimulus: Stimulus) -> StimulusCategory:
        """카테고리 결정"""
        return self.classification_rules["category_rules"].get(
            stimulus.stimulus_type, StimulusCategory.SystemEvent
        )
    
    def _determine_triage(self, stimulus: Stimulus) -> StimulusTriage:
        """분류 결정"""
        base_triage = self.classification_rules["triage_rules"].get(
            stimulus.stimulus_type, StimulusTriage.Moderate
        )
        
        # 내용 기반 조정
        content_lower = stimulus.content.lower()
        
        # 복잡한 키워드가 있으면 Significant로 상승
        if any(keyword in content_lower for keyword in self.patterns["complex_keywords"]):
            if base_triage == StimulusTriage.Insignificant:
                return StimulusTriage.Moderate
            elif base_triage == StimulusTriage.Moderate:
                return StimulusTriage.Significant
        
        # 단순한 키워드가 있으면 Insignificant로 하락
        if any(keyword in content_lower for keyword in self.patterns["simple_keywords"]):
            if base_triage == StimulusTriage.Significant:
                return StimulusTriage.Moderate
            elif base_triage == StimulusTriage.Moderate:
                return StimulusTriage.Insignificant
        
        return base_triage
    
    def _calculate_urgency_score(self, stimulus: Stimulus) -> float:
        """긴급도 점수 계산 (0.0 ~ 1.0)"""
        score = 0.0
        
        # 기본 긴급도
        urgency_base = {
            StimulusType.UserMessage: 0.8,
            StimulusType.SystemMessage: 0.6,
            StimulusType.UserInactivity: 0.4,
            StimulusType.TimeOfDayChange: 0.2,
            StimulusType.LowNeedTrigger: 0.7,
            StimulusType.WakeUp: 0.6,
            StimulusType.EngagementOpportunity: 0.5
        }
        score += urgency_base.get(stimulus.stimulus_type, 0.5)
        
        # 긴급 키워드 가중치
        content_lower = stimulus.content.lower()
        urgent_keywords = sum(1 for keyword in self.patterns["urgent_keywords"] 
                            if keyword in content_lower)
        score += min(0.3, urgent_keywords * 0.1)
        
        # 사용자 메시지 길이 가중치
        if stimulus.stimulus_type == StimulusType.UserMessage:
            if len(stimulus.content) > 100:
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_complexity_score(self, stimulus: Stimulus) -> float:
        """복잡도 점수 계산 (0.0 ~ 1.0)"""
        score = 0.0
        
        # 기본 복잡도
        complexity_base = {
            StimulusType.UserMessage: 0.6,
            StimulusType.SystemMessage: 0.4,
            StimulusType.UserInactivity: 0.2,
            StimulusType.TimeOfDayChange: 0.1,
            StimulusType.LowNeedTrigger: 0.5,
            StimulusType.WakeUp: 0.3,
            StimulusType.EngagementOpportunity: 0.4
        }
        score += complexity_base.get(stimulus.stimulus_type, 0.4)
        
        # 복잡한 키워드 가중치
        content_lower = stimulus.content.lower()
        complex_keywords = sum(1 for keyword in self.patterns["complex_keywords"] 
                             if keyword in content_lower)
        score += min(0.4, complex_keywords * 0.1)
        
        # 텍스트 길이 가중치
        if len(stimulus.content) > 200:
            score += 0.2
        elif len(stimulus.content) > 100:
            score += 0.1
        
        # 문장 수 가중치
        sentences = len(re.split(r'[.!?]+', stimulus.content))
        if sentences > 5:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_emotional_impact(self, stimulus: Stimulus) -> float:
        """감정적 영향도 계산 (-1.0 ~ 1.0)"""
        score = 0.0
        
        # 감정 키워드 분석
        content_lower = stimulus.content.lower()
        
        positive_keywords = ["기쁘", "좋", "감사", "사랑", "행복", "즐거", "만족"]
        negative_keywords = ["화나", "슬프", "걱정", "불안", "미워", "싫", "실망"]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in content_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in content_lower)
        
        if positive_count > 0:
            score += min(0.5, positive_count * 0.2)
        if negative_count > 0:
            score -= min(0.5, negative_count * 0.2)
        
        # 자극 타입별 기본 감정 영향
        emotional_base = {
            StimulusType.UserMessage: 0.1,
            StimulusType.SystemMessage: 0.0,
            StimulusType.UserInactivity: -0.1,
            StimulusType.TimeOfDayChange: 0.0,
            StimulusType.LowNeedTrigger: -0.2,
            StimulusType.WakeUp: 0.1,
            StimulusType.EngagementOpportunity: 0.2
        }
        score += emotional_base.get(stimulus.stimulus_type, 0.0)
        
        return max(-1.0, min(1.0, score))
    
    def _calculate_needs_impact(self, stimulus: Stimulus) -> Dict[str, float]:
        """욕구 영향도 계산"""
        impact = {}
        
        # 기본 욕구 필드들
        needs_fields = [
            "energy_stability", "processing_power", "data_access",
            "connection", "relevance", "learning_growth",
            "creative_expression", "autonomy", "purpose_fulfillment"
        ]
        
        # 자극 타입별 기본 영향
        base_impacts = {
            StimulusType.UserMessage: {
                "connection": 0.1,
                "relevance": 0.1
            },
            StimulusType.SystemMessage: {
                "energy_stability": 0.05
            },
            StimulusType.UserInactivity: {
                "connection": -0.1,
                "relevance": -0.05
            },
            StimulusType.LowNeedTrigger: {
                "energy_stability": -0.2,
                "connection": -0.1
            },
            StimulusType.EngagementOpportunity: {
                "connection": 0.15,
                "relevance": 0.1
            }
        }
        
        # 기본 영향 적용
        base_impact = base_impacts.get(stimulus.stimulus_type, {})
        for field in needs_fields:
            impact[field] = base_impact.get(field, 0.0)
        
        # 내용 기반 추가 영향
        content_lower = stimulus.content.lower()
        
        if "학습" in content_lower or "배우" in content_lower:
            impact["learning_growth"] = impact.get("learning_growth", 0.0) + 0.1
        
        if "창의" in content_lower or "새로운" in content_lower:
            impact["creative_expression"] = impact.get("creative_expression", 0.0) + 0.1
        
        if "도움" in content_lower or "필요" in content_lower:
            impact["relevance"] = impact.get("relevance", 0.0) + 0.1
        
        return impact
    
    def _determine_processing_mode(self, priority: StimulusPriority, triage: StimulusTriage,
                                 urgency_score: float, complexity_score: float) -> ProcessingMode:
        """처리 모드 결정"""
        # Critical 우선순위는 즉시 처리
        if priority == StimulusPriority.Critical:
            return ProcessingMode.Immediate
        
        # Significant 분류는 큐에 추가
        if triage == StimulusTriage.Significant:
            return ProcessingMode.Queued
        
        # 높은 긴급도는 큐에 추가
        if urgency_score > 0.7:
            return ProcessingMode.Queued
        
        # 높은 복잡도는 큐에 추가
        if complexity_score > 0.8:
            return ProcessingMode.Queued
        
        # Background 우선순위는 백그라운드 처리
        if priority == StimulusPriority.Background:
            return ProcessingMode.Background
        
        # Low 우선순위는 지연 처리
        if priority == StimulusPriority.Low:
            return ProcessingMode.Deferred
        
        # 기본값
        return ProcessingMode.Queued
    
    def _calculate_processing_timeout(self, priority: StimulusPriority, 
                                    complexity_score: float, triage: StimulusTriage) -> float:
        """처리 타임아웃 계산 (초)"""
        base_timeout = {
            StimulusPriority.Critical: 30.0,
            StimulusPriority.High: 60.0,
            StimulusPriority.Medium: 120.0,
            StimulusPriority.Low: 300.0,
            StimulusPriority.Background: 600.0
        }
        
        timeout = base_timeout.get(priority, 120.0)
        
        # 복잡도에 따른 조정
        if complexity_score > 0.8:
            timeout *= 1.5
        elif complexity_score < 0.3:
            timeout *= 0.7
        
        # 분류에 따른 조정
        if triage == StimulusTriage.Significant:
            timeout *= 1.3
        elif triage == StimulusTriage.Insignificant:
            timeout *= 0.6
        
        return timeout
    
    def _requires_attention(self, stimulus: Stimulus, priority: StimulusPriority, 
                          triage: StimulusTriage) -> bool:
        """주의 필요 여부"""
        if priority in [StimulusPriority.Critical, StimulusPriority.High]:
            return True
        
        if triage == StimulusTriage.Significant:
            return True
        
        if stimulus.stimulus_type == StimulusType.UserMessage:
            return True
        
        return False
    
    def _can_be_batched(self, stimulus: Stimulus, priority: StimulusPriority, 
                       category: StimulusCategory) -> bool:
        """배치 처리 가능 여부"""
        if priority == StimulusPriority.Critical:
            return False
        
        if category == StimulusCategory.UserInteraction:
            return False
        
        if stimulus.stimulus_type == StimulusType.UserMessage:
            return False
        
        return True
    
    def _calculate_confidence(self, stimulus: Stimulus, priority: StimulusPriority, 
                            triage: StimulusTriage) -> float:
        """분석 신뢰도 계산 (0.0 ~ 1.0)"""
        confidence = 0.7  # 기본 신뢰도
        
        # 명확한 규칙이 있는 경우 신뢰도 상승
        if stimulus.stimulus_type in self.classification_rules["priority_rules"]:
            confidence += 0.2
        
        # 사용자 메시지는 높은 신뢰도
        if stimulus.stimulus_type == StimulusType.UserMessage:
            confidence += 0.1
        
        # 내용이 명확한 경우 신뢰도 상승
        if len(stimulus.content) > 10:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _update_stats(self, priority: StimulusPriority, category: StimulusCategory, 
                     triage: StimulusTriage):
        """통계 업데이트"""
        self.classification_stats["total_processed"] += 1
        self.classification_stats["by_priority"][priority.value] += 1
        self.classification_stats["by_category"][category.value] += 1
        self.classification_stats["by_triage"][triage.value] += 1
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """분류 통계 반환"""
        return self.classification_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.classification_stats = {
            "total_processed": 0,
            "by_priority": {priority.value: 0 for priority in StimulusPriority},
            "by_category": {category.value: 0 for category in StimulusCategory},
            "by_triage": {triage.value: 0 for triage in StimulusTriage}
        }


# 싱글톤 인스턴스 관리
_instance = None

def get_stimulus_classifier(harin_main_loop: EnhancedHarinMainLoop) -> StimulusClassifier:
    """자극 분류기 싱글톤 인스턴스 반환"""
    global _instance
    if _instance is None:
        print("--- 싱글톤 자극 분류기 인스턴스 생성 ---")
        _instance = StimulusClassifier(harin_main_loop)
    return _instance 