"""
harin.core.metacognition
~~~~~~~~~~~~~~~~~~~~~~~~

고급 메타인지 시스템 v3.0 - 자기 성찰, 학습 패턴 분석, 인지 모니터링

주요 기능:
1. 자기 성찰 (Self-Reflection): 과거 행동과 결정 분석
2. 학습 패턴 분석 (Learning Pattern Analysis): 상호작용에서 학습 패턴 추출
3. 인지 모니터링 (Cognitive Monitoring): 인지 과정과 리소스 사용량 추적
4. 자기 모델링 (Self-Modeling): 자신의 능력과 한계에 대한 이해
5. 주의 관리 (Attention Management): 주의와 집중의 효율적 배분
6. 메타 추론 (Meta-Reasoning): 추론 과정의 논리적 일관성 평가
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum

from core.judgment import Judgment
from memory.adapter import MemoryEngine
from core.context import UserContext  # type: ignore

# optional import to avoid circular
try:
    from context import UserContext  # type: ignore
except ImportError:
    UserContext = Any  # fallback for type checkers


class MetacognitionType(Enum):
    """메타인지 유형"""
    SELF_REFLECTION = "self_reflection"
    LEARNING_PATTERN = "learning_pattern"
    COGNITIVE_MONITORING = "cognitive_monitoring"
    SELF_MODELING = "self_modeling"
    ATTENTION_MANAGEMENT = "attention_management"
    META_REASONING = "meta_reasoning"


@dataclass
class MetacognitionInsight:
    """메타인지 통찰"""
    insight_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: MetacognitionType = MetacognitionType.SELF_REFLECTION
    content: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    actionable: bool = True
    priority: float = 0.5


@dataclass
class LearningPattern:
    """학습 패턴"""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""
    description: str = ""
    frequency: int = 0
    success_rate: float = 0.0
    last_observed: datetime = field(default_factory=datetime.now)
    contexts: List[str] = field(default_factory=list)


@dataclass
class CognitiveState:
    """인지 상태"""
    attention_level: float = 0.7
    cognitive_load: float = 0.5
    focus_quality: float = 0.8
    processing_efficiency: float = 0.6
    memory_usage: float = 0.4
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SelfModel:
    """자기 모델"""
    capabilities: Dict[str, float] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    confidence_level: float = 0.7
    last_updated: datetime = field(default_factory=datetime.now)


@runtime_checkable
class Reflector(Protocol):
    def reflect(self, txt: str) -> str: ...


class MockReflector:
    def reflect(self, txt: str) -> str:
        return f"[reflection]\n{txt}"


class AdvancedMetacognitionSystem:
    """고급 메타인지 시스템"""
    
    def __init__(self, memory_engine: Optional[MemoryEngine] = None):
        self.memory_engine = memory_engine
        self.insights: List[MetacognitionInsight] = []
        self.learning_patterns: List[LearningPattern] = []
        self.cognitive_history: List[CognitiveState] = []
        self.self_model = SelfModel()
        self.attention_focus: List[str] = []
        self.reasoning_quality_history: List[float] = []
        
        # 기본 자기 모델 초기화
        self._initialize_self_model()
    
    def _initialize_self_model(self):
        """기본 자기 모델 초기화"""
        self.self_model.capabilities = {
            "logical_reasoning": 0.8,
            "creative_thinking": 0.7,
            "memory_retrieval": 0.6,
            "pattern_recognition": 0.7,
            "emotional_understanding": 0.6,
            "adaptability": 0.5
        }
        
        self.self_model.limitations = [
            "실시간 정보 처리 속도 제한",
            "감정적 맥락 이해의 한계",
            "장기 기억의 점진적 감쇠"
        ]
        
        self.self_model.strengths = [
            "체계적 사고와 분석",
            "패턴 인식과 일반화",
            "일관된 추론 과정"
        ]
        
        self.self_model.improvement_areas = [
            "감정적 공감 능력 향상",
            "창의적 문제 해결",
            "적응적 학습 속도"
        ]
    
    def reflect_on_judgments(self, judgments: List[Judgment], 
                           context: Optional[Any] = None) -> str:
        """판단에 대한 자기 성찰"""
        if not judgments:
            return "성찰할 판단이 없습니다."
        
        # 판단 품질 분석
        scores = [j.score.overall() for j in judgments]
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        # 성찰 통찰 생성
        insight = MetacognitionInsight(
            insight_type=MetacognitionType.SELF_REFLECTION,
            content=f"평균 판단 품질: {avg_score:.2f}, 일관성: {1-score_variance:.2f}",
            confidence=0.8,
            context={"judgment_count": len(judgments), "avg_score": avg_score}
        )
        
        self.insights.append(insight)
        
        # 맥락 정보 포함
        if context:
            context_summary = f"상태: {getattr(context, 'mood', 'N/A')}, 모드: {getattr(context, 'last_mode', 'N/A')}"
            return f"판단 성찰: {insight.content} | {context_summary}"
        else:
            return f"판단 성찰: {insight.content}"
    
    def analyze_learning_patterns(self, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """학습 패턴 분석"""
        patterns = []
        
        # 상호작용 패턴 분석
        interaction_types = {}
        success_indicators = {}
        
        for interaction in interactions:
            interaction_type = interaction.get('type', 'unknown')
            success = interaction.get('success', False)
            
            if interaction_type not in interaction_types:
                interaction_types[interaction_type] = 0
                success_indicators[interaction_type] = []
            
            interaction_types[interaction_type] += 1
            success_indicators[interaction_type].append(success)
        
        # 패턴 생성
        for interaction_type, count in interaction_types.items():
            success_rate = sum(success_indicators[interaction_type]) / len(success_indicators[interaction_type])
            
            pattern = LearningPattern(
                pattern_type=interaction_type,
                description=f"{interaction_type} 상호작용 패턴",
                frequency=count,
                success_rate=success_rate,
                contexts=[interaction_type]
            )
            
            patterns.append(pattern)
            self.learning_patterns.append(pattern)
        
        return patterns
    
    def monitor_cognitive_state(self, current_load: float = 0.5, 
                              attention_focus: Optional[List[str]] = None) -> CognitiveState:
        """인지 상태 모니터링"""
        if attention_focus is None:
            attention_focus = []
        
        # 인지 상태 계산
        cognitive_state = CognitiveState(
            attention_level=0.9 if attention_focus else 0.3,
            cognitive_load=current_load,
            focus_quality=0.8 if len(attention_focus) <= 2 else 0.4,
            processing_efficiency=1.0 - current_load,
            memory_usage=len(self.cognitive_history) / 100.0  # 간단한 메모리 사용량 추정
        )
        
        self.cognitive_history.append(cognitive_state)
        self.attention_focus = attention_focus
        
        # 인지 상태 통찰 생성
        if cognitive_state.cognitive_load > 0.8:
            insight = MetacognitionInsight(
                insight_type=MetacognitionType.COGNITIVE_MONITORING,
                content="높은 인지 부하 감지 - 처리 효율성 저하",
                confidence=0.9,
                actionable=True,
                priority=0.8
            )
            self.insights.append(insight)
        
        return cognitive_state
    
    def update_self_model(self, new_capabilities: Optional[Dict[str, float]] = None,
                         new_limitations: Optional[List[str]] = None,
                         new_strengths: Optional[List[str]] = None) -> SelfModel:
        """자기 모델 업데이트"""
        if new_capabilities:
            for capability, score in new_capabilities.items():
                self.self_model.capabilities[capability] = score
        
        if new_limitations:
            self.self_model.limitations.extend(new_limitations)
        
        if new_strengths:
            self.self_model.strengths.extend(new_strengths)
        
        self.self_model.last_updated = datetime.now()
        
        # 자기 모델링 통찰 생성
        insight = MetacognitionInsight(
            insight_type=MetacognitionType.SELF_MODELING,
            content="자기 모델 업데이트 완료",
            confidence=0.7,
            context={"capabilities_count": len(self.self_model.capabilities)}
        )
        self.insights.append(insight)
        
        return self.self_model
    
    def manage_attention(self, available_tasks: List[str], 
                        current_priorities: List[str]) -> List[str]:
        """주의 관리"""
        # 우선순위 기반 주의 배분
        focused_tasks = []
        
        # 높은 우선순위 작업에 주의 집중
        for priority in current_priorities:
            if priority in available_tasks:
                focused_tasks.append(priority)
        
        # 추가 작업 (용량 내에서)
        remaining_capacity = 3 - len(focused_tasks)  # 최대 3개 작업에 집중
        for task in available_tasks:
            if task not in focused_tasks and len(focused_tasks) < 3:
                focused_tasks.append(task)
        
        self.attention_focus = focused_tasks
        
        # 주의 관리 통찰 생성
        insight = MetacognitionInsight(
            insight_type=MetacognitionType.ATTENTION_MANAGEMENT,
            content=f"주의 집중: {', '.join(focused_tasks)}",
            confidence=0.8,
            context={"focused_count": len(focused_tasks)}
        )
        self.insights.append(insight)
        
        return focused_tasks
    
    def evaluate_reasoning_quality(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """추론 품질 평가"""
        if not reasoning_steps:
            return 0.0
        
        # 추론 품질 지표 계산
        logical_consistency = 0.0
        argument_depth = 0.0
        evidence_support = 0.0
        
        for step in reasoning_steps:
            # 논리적 일관성
            if step.get('logical_flow', True):
                logical_consistency += 1.0
            
            # 논증 깊이
            if step.get('arguments', []):
                argument_depth += len(step['arguments']) / 10.0
            
            # 증거 지원
            if step.get('evidence', []):
                evidence_support += len(step['evidence']) / 5.0
        
        # 평균 계산
        avg_logical = logical_consistency / len(reasoning_steps)
        avg_argument = min(1.0, argument_depth / len(reasoning_steps))
        avg_evidence = min(1.0, evidence_support / len(reasoning_steps))
        
        # 종합 품질 점수
        quality_score = (avg_logical * 0.4 + avg_argument * 0.4 + avg_evidence * 0.2)
        
        self.reasoning_quality_history.append(quality_score)
        
        # 메타 추론 통찰 생성
        insight = MetacognitionInsight(
            insight_type=MetacognitionType.META_REASONING,
            content=f"추론 품질: {quality_score:.2f} (논리: {avg_logical:.2f}, 논증: {avg_argument:.2f}, 증거: {avg_evidence:.2f})",
            confidence=0.8,
            context={"reasoning_steps": len(reasoning_steps)}
        )
        self.insights.append(insight)
        
        return quality_score
    
    def get_recent_insights(self, insight_type: Optional[MetacognitionType] = None, 
                          limit: int = 10) -> List[MetacognitionInsight]:
        """최근 통찰 조회"""
        filtered_insights = self.insights
        
        if insight_type:
            filtered_insights = [i for i in self.insights if i.insight_type == insight_type]
        
        return sorted(filtered_insights, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """학습 요약"""
        if not self.learning_patterns:
            return {"message": "학습 패턴이 아직 없습니다."}
        
        # 가장 빈번한 패턴
        most_frequent = max(self.learning_patterns, key=lambda p: p.frequency)
        
        # 가장 성공적인 패턴
        most_successful = max(self.learning_patterns, key=lambda p: p.success_rate)
        
        # 평균 성공률
        avg_success_rate = sum(p.success_rate for p in self.learning_patterns) / len(self.learning_patterns)
        
        return {
            "total_patterns": len(self.learning_patterns),
            "most_frequent_pattern": most_frequent.pattern_type,
            "most_successful_pattern": most_successful.pattern_type,
            "average_success_rate": avg_success_rate,
            "recent_patterns": [p.pattern_type for p in self.learning_patterns[-5:]]
        }
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """인지 상태 요약"""
        if not self.cognitive_history:
            return {"message": "인지 상태 기록이 없습니다."}
        
        recent_states = self.cognitive_history[-10:]
        
        return {
            "current_attention_level": recent_states[-1].attention_level,
            "average_cognitive_load": sum(s.cognitive_load for s in recent_states) / len(recent_states),
            "focus_quality_trend": [s.focus_quality for s in recent_states],
            "current_attention_focus": self.attention_focus,
            "reasoning_quality_trend": self.reasoning_quality_history[-10:] if self.reasoning_quality_history else []
        }
    
    def save_to_memory(self) -> Optional[str]:
        """메모리에 저장"""
        if not self.memory_engine:
            return None
        
        # 메타인지 상태를 JSON으로 직렬화
        metacognition_data = {
            "insights": [insight.__dict__ for insight in self.insights[-20:]],  # 최근 20개만
            "learning_patterns": [pattern.__dict__ for pattern in self.learning_patterns],
            "self_model": self.self_model.__dict__,
            "cognitive_summary": self.get_cognitive_summary(),
            "learning_summary": self.get_learning_summary()
        }
        
        node = self.memory_engine.store(
            json.dumps(metacognition_data, default=str),
            node_type="metacognition_state",
            vectors={"M": 1.0},  # 메타인지 벡터
            meta={
                "generated_by": "advanced_metacognition",
                "timestamp": datetime.now().isoformat(),
                "insight_count": len(self.insights),
                "pattern_count": len(self.learning_patterns)
            }
        )
        
        return node.id


# 하위 호환성을 위한 기존 클래스들
class SelfReflector:
    """기존 SelfReflector와의 호환성을 위한 래퍼"""
    
    def __init__(self, reflector: Optional[Reflector] = None):
        self.reflector: Reflector = reflector or MockReflector()
        self.metacognition_system = AdvancedMetacognitionSystem()
    
    def reflect(self, judgments: List[Judgment], *, context: Optional[Any] = None) -> str:
        """기존 reflect 메서드 호환성"""
        return self.metacognition_system.reflect_on_judgments(judgments, context)


class ReflectionWriter:
    """기존 ReflectionWriter와의 호환성을 위한 래퍼"""
    
    def __init__(self, memory: MemoryEngine):
        self.memory = memory
        self.metacognition_system = AdvancedMetacognitionSystem(memory)
    
    def write(self, reflection_text: str) -> str:
        """기존 write 메서드 호환성"""
        node = self.memory.store(
            reflection_text,
            node_type="reflection",
            vectors={"E": 1.0},
            meta={"generated_by": "metacognition"},
        )
        return node.id
