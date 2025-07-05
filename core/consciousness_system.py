"""
Harin Core Consciousness System - LIDA Integration
의식(Consciousness) 시스템: 작업 공간 활성화 및 의식적 정보 처리 시스템
PMM의 ghost 시스템을 참고하여 구현
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import math
import random

from memory.models import Stimulus, Feature, Action, Intention, Narrative, GhostState
from core.perception_system import PerceptionResult, PatternType


class ConsciousnessLevel(Enum):
    """의식 수준"""
    UNCONSCIOUS = "unconscious"      # 무의식
    PRE_CONSCIOUS = "pre_conscious"  # 전의식
    CONSCIOUS = "conscious"          # 의식
    FOCUSED = "focused"             # 집중
    FLOW = "flow"                   # 몰입


class WorkspaceComponent(Enum):
    """작업 공간 구성 요소"""
    CURRENT_FOCUS = "current_focus"
    ACTIVE_MEMORIES = "active_memories"
    CURRENT_GOALS = "current_goals"
    CONTEXT = "context"
    EMOTIONAL_STATE = "emotional_state"
    COGNITIVE_LOAD = "cognitive_load"


class ContextType(Enum):
    """맥락 유형"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SOCIAL = "social"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    ENVIRONMENTAL = "environmental"


class WorkspaceState(BaseModel):
    """작업 공간 상태"""
    consciousness_level: ConsciousnessLevel = ConsciousnessLevel.CONSCIOUS
    current_focus: Optional[Any] = None
    active_memories: List[Any] = Field(default_factory=list)
    current_goals: List[Any] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    emotional_state: Dict[str, float] = Field(default_factory=dict)
    cognitive_load: float = Field(default=0.5, ge=0.0, le=1.0)
    attention_allocation: Dict[str, float] = Field(default_factory=dict)
    processing_capacity: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class ConsciousnessSystem:
    """의식 시스템 - LIDA의 Consciousness 단계 구현"""
    
    def __init__(self, memory_system=None, goal_system=None):
        self.memory_system = memory_system
        self.goal_system = goal_system
        self.workspace_history: List[WorkspaceState] = []
        self.current_workspace: Optional[WorkspaceState] = None
        
        # 의식 시스템 설정
        self.max_workspace_capacity = 7  # Miller's Law: 7±2
        self.attention_threshold = 0.1
        self.cognitive_load_threshold = 0.8
        
        # 초기 작업 공간 생성
        self._initialize_workspace()
    
    def _initialize_workspace(self):
        """초기 작업 공간 초기화"""
        self.current_workspace = WorkspaceState(
            consciousness_level=ConsciousnessLevel.CONSCIOUS,
            cognitive_load=0.3,
            processing_capacity=1.0
        )
    
    def activate_workspace(self, attention_allocation: Dict[str, float]) -> WorkspaceState:
        """작업 공간 활성화"""
        if not self.current_workspace:
            self._initialize_workspace()
        
        # 현재 주의 집중 대상 결정
        current_focus = self._determine_current_focus(attention_allocation)
        
        # 관련 활성 메모리 검색
        active_memories = self._retrieve_active_memories(attention_allocation)
        
        # 현재 목표 검색
        current_goals = self._retrieve_current_goals()
        
        # 현재 맥락 구성
        context = self._build_context(attention_allocation)
        
        # 감정 상태 업데이트
        emotional_state = self._get_emotional_context()
        
        # 인지 부하 계산
        cognitive_load = self._calculate_cognitive_load(attention_allocation, active_memories)
        
        # 의식 수준 결정
        consciousness_level = self._determine_consciousness_level(cognitive_load, attention_allocation)
        
        # 처리 용량 조정
        processing_capacity = self._adjust_processing_capacity(consciousness_level, cognitive_load)
        
        # 작업 공간 업데이트
        self.current_workspace = WorkspaceState(
            consciousness_level=consciousness_level,
            current_focus=current_focus,
            active_memories=active_memories,
            current_goals=current_goals,
            context=context,
            emotional_state=emotional_state,
            cognitive_load=cognitive_load,
            attention_allocation=attention_allocation,
            processing_capacity=processing_capacity
        )
        
        # 히스토리에 추가
        self.workspace_history.append(self.current_workspace)
        
        return self.current_workspace
    
    def _determine_current_focus(self, attention_allocation: Dict[str, float]) -> Optional[Any]:
        """현재 주의 집중 대상 결정"""
        if not attention_allocation:
            return None
        
        # 가장 높은 주의를 받는 자극 찾기
        max_attention = max(attention_allocation.values())
        focus_candidates = []
        
        for modality, attention in attention_allocation.items():
            if attention >= max_attention * 0.8:  # 80% 이상의 주의를 받는 것들
                focus_candidates.append(modality)
        
        if focus_candidates:
            # 여러 후보가 있으면 랜덤 선택 (실제로는 더 정교한 로직 필요)
            return random.choice(focus_candidates)
        
        return None
    
    def _retrieve_active_memories(self, attention_allocation: Dict[str, float]) -> List[Any]:
        """관련 활성 메모리 검색"""
        active_memories = []
        
        if not self.memory_system:
            return active_memories
        
        for modality, attention in attention_allocation.items():
            if attention > self.attention_threshold:
                # 임계값 이상의 주의를 받는 모달리티에 관련된 메모리 검색
                try:
                    memories = self.memory_system.search_related_memories(modality, limit=5)
                    active_memories.extend(memories)
                except Exception as e:
                    print(f"메모리 검색 오류: {e}")
        
        # 작업 공간 용량 제한
        if len(active_memories) > self.max_workspace_capacity:
            # 중요도에 따라 정렬하고 상위 항목만 유지
            active_memories = self._prioritize_memories(active_memories)[:self.max_workspace_capacity]
        
        return active_memories
    
    def _retrieve_current_goals(self) -> List[Any]:
        """현재 목표 검색"""
        current_goals = []
        
        if self.goal_system:
            try:
                current_goals = self.goal_system.get_active_goals()
            except Exception as e:
                print(f"목표 검색 오류: {e}")
        
        return current_goals
    
    def _build_context(self, attention_allocation: Dict[str, float]) -> Dict[str, Any]:
        """현재 맥락 구성"""
        context = {}
        
        # 시간적 맥락
        context[ContextType.TEMPORAL.value] = self._get_temporal_context()
        
        # 공간적 맥락
        context[ContextType.SPATIAL.value] = self._get_spatial_context()
        
        # 사회적 맥락
        context[ContextType.SOCIAL.value] = self._get_social_context()
        
        # 감정적 맥락
        context[ContextType.EMOTIONAL.value] = self._get_emotional_context()
        
        # 인지적 맥락
        context[ContextType.COGNITIVE.value] = self._get_cognitive_context(attention_allocation)
        
        # 환경적 맥락
        context[ContextType.ENVIRONMENTAL.value] = self._get_environmental_context()
        
        return context
    
    def _get_temporal_context(self) -> Dict[str, Any]:
        """시간적 맥락"""
        now = datetime.now()
        return {
            "current_time": now.isoformat(),
            "time_of_day": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": now.weekday() >= 5,
            "session_duration": self._calculate_session_duration()
        }
    
    def _get_spatial_context(self) -> Dict[str, Any]:
        """공간적 맥락"""
        return {
            "location": "virtual_environment",
            "environment_type": "digital",
            "spatial_orientation": "centered"
        }
    
    def _get_social_context(self) -> Dict[str, Any]:
        """사회적 맥락"""
        return {
            "interaction_partner": "user",
            "interaction_mode": "conversational",
            "social_distance": "close",
            "relationship_type": "assistant_user"
        }
    
    def _get_emotional_context(self) -> Dict[str, float]:
        """감정적 맥락"""
        # 현재 감정 상태 (실제로는 감정 시스템에서 가져옴)
        return {
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.5,
            "emotional_stability": 0.7
        }
    
    def _get_cognitive_context(self, attention_allocation: Dict[str, float]) -> Dict[str, Any]:
        """인지적 맥락"""
        total_attention = sum(attention_allocation.values()) if attention_allocation else 0.0
        
        return {
            "attention_distribution": attention_allocation,
            "total_attention": total_attention,
            "attention_focus": len([a for a in attention_allocation.values() if a > self.attention_threshold]),
            "cognitive_resources_available": 1.0 - total_attention
        }
    
    def _get_environmental_context(self) -> Dict[str, Any]:
        """환경적 맥락"""
        return {
            "system_state": "active",
            "available_resources": "sufficient",
            "environmental_stability": "stable"
        }
    
    def _calculate_cognitive_load(self, attention_allocation: Dict[str, float], active_memories: List[Any]) -> float:
        """인지 부하 계산"""
        # 주의 분배에 따른 부하
        attention_load = sum(attention_allocation.values()) if attention_allocation else 0.0
        
        # 활성 메모리 수에 따른 부하
        memory_load = min(1.0, len(active_memories) / self.max_workspace_capacity)
        
        # 복잡도에 따른 부하
        complexity_load = self._calculate_complexity_load(active_memories)
        
        # 종합 인지 부하
        total_load = (attention_load * 0.4 + memory_load * 0.3 + complexity_load * 0.3)
        
        return min(1.0, total_load)
    
    def _calculate_complexity_load(self, active_memories: List[Any]) -> float:
        """복잡도 부하 계산"""
        if not active_memories:
            return 0.0
        
        # 메모리 유형별 복잡도 가중치
        complexity_weights = {
            'stimulus': 0.2,
            'feature': 0.3,
            'action': 0.4,
            'intention': 0.5,
            'narrative': 0.6
        }
        
        total_complexity = 0.0
        for memory in active_memories:
            memory_type = type(memory).__name__.lower()
            weight = complexity_weights.get(memory_type, 0.3)
            total_complexity += weight
        
        return min(1.0, total_complexity / len(active_memories))
    
    def _determine_consciousness_level(self, cognitive_load: float, attention_allocation: Dict[str, float]) -> ConsciousnessLevel:
        """의식 수준 결정"""
        total_attention = sum(attention_allocation.values()) if attention_allocation else 0.0
        
        if cognitive_load > self.cognitive_load_threshold:
            return ConsciousnessLevel.FOCUSED
        elif total_attention > 0.8:
            return ConsciousnessLevel.FLOW
        elif total_attention > 0.5:
            return ConsciousnessLevel.CONSCIOUS
        elif total_attention > 0.2:
            return ConsciousnessLevel.PRE_CONSCIOUS
        else:
            return ConsciousnessLevel.UNCONSCIOUS
    
    def _adjust_processing_capacity(self, consciousness_level: ConsciousnessLevel, cognitive_load: float) -> float:
        """처리 용량 조정"""
        # 의식 수준에 따른 기본 용량
        base_capacity = {
            ConsciousnessLevel.UNCONSCIOUS: 0.1,
            ConsciousnessLevel.PRE_CONSCIOUS: 0.3,
            ConsciousnessLevel.CONSCIOUS: 0.7,
            ConsciousnessLevel.FOCUSED: 0.9,
            ConsciousnessLevel.FLOW: 1.0
        }
        
        base = base_capacity.get(consciousness_level, 0.5)
        
        # 인지 부하에 따른 조정
        load_factor = 1.0 - (cognitive_load * 0.3)
        
        return min(1.0, base * load_factor)
    
    def _prioritize_memories(self, memories: List[Any]) -> List[Any]:
        """메모리 우선순위 결정"""
        # 간단한 우선순위 결정 (실제로는 더 정교한 로직 필요)
        prioritized = sorted(memories, key=lambda x: self._calculate_memory_priority(x), reverse=True)
        return prioritized
    
    def _calculate_memory_priority(self, memory: Any) -> float:
        """메모리 우선순위 계산"""
        # 메모리 유형별 우선순위
        type_priorities = {
            'Stimulus': 1.0,
            'Intention': 0.9,
            'Action': 0.8,
            'Feature': 0.7,
            'Narrative': 0.6
        }
        
        memory_type = type(memory).__name__
        base_priority = type_priorities.get(memory_type, 0.5)
        
        # 시간적 최신성 (최근일수록 높은 우선순위)
        if hasattr(memory, 'timestamp'):
            time_factor = 1.0  # 실제로는 시간 차이 계산
        else:
            time_factor = 0.8
        
        return base_priority * time_factor
    
    def _calculate_session_duration(self) -> float:
        """세션 지속 시간 계산 (분)"""
        if not self.workspace_history:
            return 0.0
        
        start_time = self.workspace_history[0].timestamp
        current_time = datetime.now()
        duration = (current_time - start_time).total_seconds() / 60.0
        
        return duration
    
    def conscious_processing(self, workspace: WorkspaceState) -> Dict[str, Any]:
        """의식적 정보 처리"""
        if not workspace:
            return {}
        
        # 정보 통합
        integrated_understanding = self._integrate_information(workspace)
        
        # 의사결정 요구사항 식별
        decision_requirements = self._identify_decision_requirements(workspace)
        
        # 행동 계획 생성
        action_plan = self._generate_action_plan(workspace)
        
        # 학습 기회 식별
        learning_opportunities = self._identify_learning_opportunities(workspace)
        
        return {
            "integrated_understanding": integrated_understanding,
            "decision_requirements": decision_requirements,
            "action_plan": action_plan,
            "learning_opportunities": learning_opportunities,
            "consciousness_level": workspace.consciousness_level.value,
            "cognitive_load": workspace.cognitive_load,
            "processing_capacity": workspace.processing_capacity
        }
    
    def _integrate_information(self, workspace: WorkspaceState) -> Dict[str, Any]:
        """정보 통합"""
        integration_result = {
            "current_focus": workspace.current_focus,
            "active_memory_count": len(workspace.active_memories),
            "goal_count": len(workspace.current_goals),
            "context_completeness": len(workspace.context),
            "emotional_state": workspace.emotional_state,
            "attention_distribution": workspace.attention_allocation
        }
        
        # 메모리 간 연관성 분석
        memory_associations = self._analyze_memory_associations(workspace.active_memories)
        integration_result["memory_associations"] = memory_associations
        
        # 맥락 일관성 평가
        context_consistency = self._evaluate_context_consistency(workspace.context)
        integration_result["context_consistency"] = context_consistency
        
        return integration_result
    
    def _identify_decision_requirements(self, workspace: WorkspaceState) -> Dict[str, Any]:
        """의사결정 요구사항 식별"""
        requirements = {
            "requires_decision": False,
            "decision_type": None,
            "available_options": [],
            "constraints": [],
            "urgency": 0.0
        }
        
        # 현재 주의 집중 대상이 의사결정을 요구하는지 확인
        if workspace.current_focus:
            requirements["requires_decision"] = True
            requirements["decision_type"] = "attention_based"
            requirements["urgency"] = workspace.attention_allocation.get(workspace.current_focus, 0.0)
        
        # 목표 기반 의사결정 요구사항
        if workspace.current_goals:
            requirements["requires_decision"] = True
            requirements["decision_type"] = "goal_based"
            requirements["available_options"] = [goal for goal in workspace.current_goals]
        
        return requirements
    
    def _generate_action_plan(self, workspace: WorkspaceState) -> Dict[str, Any]:
        """행동 계획 생성"""
        action_plan = {
            "planned_actions": [],
            "priority_order": [],
            "estimated_duration": 0.0,
            "resource_requirements": {},
            "success_criteria": []
        }
        
        # 현재 주의 집중 대상에 따른 행동 계획
        if workspace.current_focus:
            action_plan["planned_actions"].append({
                "action_type": "focus_processing",
                "target": workspace.current_focus,
                "priority": "high"
            })
        
        # 목표 달성을 위한 행동 계획
        for goal in workspace.current_goals:
            action_plan["planned_actions"].append({
                "action_type": "goal_pursuit",
                "target": goal,
                "priority": "medium"
            })
        
        return action_plan
    
    def _identify_learning_opportunities(self, workspace: WorkspaceState) -> List[Dict[str, Any]]:
        """학습 기회 식별"""
        opportunities = []
        
        # 새로운 패턴 발견 기회
        if workspace.active_memories:
            opportunities.append({
                "type": "pattern_recognition",
                "description": "활성 메모리에서 새로운 패턴 발견",
                "priority": "medium"
            })
        
        # 맥락 학습 기회
        if workspace.context:
            opportunities.append({
                "type": "context_learning",
                "description": "현재 맥락에서 학습 가능한 정보",
                "priority": "low"
            })
        
        return opportunities
    
    def _analyze_memory_associations(self, memories: List[Any]) -> Dict[str, Any]:
        """메모리 간 연관성 분석"""
        if len(memories) < 2:
            return {"association_count": 0, "association_strength": 0.0}
        
        # 간단한 연관성 분석 (실제로는 더 정교한 로직 필요)
        association_count = len(memories) - 1
        association_strength = min(1.0, association_count / 10.0)
        
        return {
            "association_count": association_count,
            "association_strength": association_strength,
            "clustering_coefficient": random.uniform(0.3, 0.7)
        }
    
    def _evaluate_context_consistency(self, context: Dict[str, Any]) -> float:
        """맥락 일관성 평가"""
        if not context:
            return 0.0
        
        # 맥락 요소들의 일관성 점수 (간단한 구현)
        consistency_scores = []
        
        for context_type, context_data in context.items():
            if isinstance(context_data, dict) and context_data:
                consistency_scores.append(0.8)  # 기본 일관성 점수
            else:
                consistency_scores.append(0.3)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """의식 시스템 요약"""
        if not self.current_workspace:
            return {}
        
        return {
            "consciousness_level": self.current_workspace.consciousness_level.value,
            "cognitive_load": self.current_workspace.cognitive_load,
            "processing_capacity": self.current_workspace.processing_capacity,
            "active_memory_count": len(self.current_workspace.active_memories),
            "current_focus": self.current_workspace.current_focus,
            "attention_distribution": self.current_workspace.attention_allocation,
            "workspace_history_length": len(self.workspace_history)
        }
    
    def get_workspace_history(self, limit: int = 10) -> List[WorkspaceState]:
        """작업 공간 히스토리 조회"""
        return self.workspace_history[-limit:]
    
    def clear_workspace(self):
        """작업 공간 초기화"""
        self._initialize_workspace()
        self.workspace_history.clear() 