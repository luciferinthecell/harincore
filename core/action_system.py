"""
하린코어 행동 시스템 - Enhanced with Simulation & Multi-Agent Collaboration
PM 시스템의 Action/ActionType을 참고하여 하린코어에 맞게 구현한 고급 행동 시스템
+ 행동 시뮬레이션 및 멀티 에이전트 협업 기능 추가
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import StrEnum
from pydantic import BaseModel, Field
import asyncio
import random
import json

from memory.models import Stimulus, StimulusType, EmotionalAxesModel, NeedsAxesModel, CognitionAxesModel
from core.enhanced_main_loop import EnhancedHarinMainLoop


class AgentType(StrEnum):
    """에이전트 유형"""
    REASONING = "reasoning"           # 추론 에이전트
    EMOTIONAL = "emotional"           # 감정 에이전트
    MEMORY = "memory"                 # 메모리 에이전트
    PLANNING = "planning"             # 계획 에이전트
    EXECUTION = "execution"           # 실행 에이전트
    MONITORING = "monitoring"         # 모니터링 에이전트
    LEARNING = "learning"             # 학습 에이전트


class AgentState(BaseModel):
    """에이전트 상태"""
    agent_id: str
    agent_type: AgentType
    is_active: bool = True
    current_task: Optional[str] = None
    performance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    specialization: List[str] = Field(default_factory=list)
    collaboration_history: List[str] = Field(default_factory=list)
    last_activity: datetime = Field(default_factory=datetime.now)


class CollaborationProtocol(BaseModel):
    """협업 프로토콜"""
    protocol_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    participants: List[str] = Field(default_factory=list)
    task_description: str
    coordination_method: str = "sequential"  # sequential, parallel, hierarchical
    communication_channels: List[str] = Field(default_factory=list)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ActionType(StrEnum):
    """행동 유형 시스템 - PM 시스템의 ActionType을 참고하여 확장"""
    # 기본 행동 유형
    Reply = "Reply"                           # 사용자에게 응답
    ToolCall = "ToolCall"                     # 도구/API 호출
    Ignore = "Ignore"                         # 입력 무시
    Sleep = "Sleep"                           # 시스템 유지보수/최적화
    
    # 적극적 상호작용
    InitiateUserConversation = "InitiateUserConversation"     # 사용자와 대화 시작
    InitiateInternalContemplation = "InitiateInternalContemplation"  # 내부 성찰/사고
    InitiateIdleMode = "InitiateIdleMode"     # 유휴 모드
    
    # 고급 인지 행동
    Think = "Think"                           # 깊은 사고
    RecallMemory = "RecallMemory"            # 기억 회상
    ManageIntent = "ManageIntent"            # 의도 관리
    Plan = "Plan"                            # 계획 수립
    ManageAwarenessFocus = "ManageAwarenessFocus"  # 주의 집중 관리
    ReflectThoughts = "ReflectThoughts"      # 사고 반성
    
    # 확장 행동 유형 (하린코어 전용)
    AnalyzeEmotion = "AnalyzeEmotion"        # 감정 분석
    ProcessMemory = "ProcessMemory"          # 기억 처리
    GenerateInsight = "GenerateInsight"      # 통찰 생성
    AdaptBehavior = "AdaptBehavior"          # 행동 적응
    LearnFromExperience = "LearnFromExperience"  # 경험 학습
    
    # 시스템 행동
    SystemMaintenance = "SystemMaintenance"  # 시스템 유지보수
    DataOptimization = "DataOptimization"    # 데이터 최적화
    PerformanceMonitoring = "PerformanceMonitoring"  # 성능 모니터링
    
    # 멀티 에이전트 협업 행동
    CoordinateAgents = "CoordinateAgents"    # 에이전트 조율
    DelegateTask = "DelegateTask"            # 작업 위임
    SynthesizeResults = "SynthesizeResults"  # 결과 종합
    ResolveConflict = "ResolveConflict"      # 충돌 해결


class ActionPriority(StrEnum):
    """행동 우선순위"""
    Critical = "Critical"      # 즉시 실행 필요
    High = "High"             # 높은 우선순위
    Medium = "Medium"         # 중간 우선순위
    Low = "Low"               # 낮은 우선순위
    Background = "Background" # 백그라운드 실행


class ActionStatus(StrEnum):
    """행동 상태"""
    Pending = "Pending"       # 대기 중
    Planning = "Planning"     # 계획 중
    Executing = "Executing"   # 실행 중
    Completed = "Completed"   # 완료
    Failed = "Failed"         # 실패
    Cancelled = "Cancelled"   # 취소됨


class ActionContext(BaseModel):
    """행동 컨텍스트"""
    user_input: Optional[str] = None
    current_emotion: EmotionalAxesModel = Field(default_factory=EmotionalAxesModel)
    current_needs: NeedsAxesModel = Field(default_factory=NeedsAxesModel)
    current_cognition: CognitionAxesModel = Field(default_factory=CognitionAxesModel)
    recent_memories: List[Dict[str, Any]] = Field(default_factory=list)
    system_state: Dict[str, Any] = Field(default_factory=dict)
    environmental_factors: Dict[str, Any] = Field(default_factory=dict)
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)


class ActionSimulation(BaseModel):
    """행동 시뮬레이션 결과"""
    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType
    action_content: str
    predicted_user_reaction: str
    predicted_emotional_impact: EmotionalAxesModel
    predicted_needs_impact: NeedsAxesModel
    predicted_cognitive_impact: CognitionAxesModel
    success_probability: float = Field(..., ge=0.0, le=1.0)
    expected_benefit: float = Field(..., ge=0.0, le=1.0)
    risk_level: float = Field(..., ge=0.0, le=1.0)
    cognitive_load: float = Field(..., ge=0.0, le=1.0)
    execution_time: float = Field(..., ge=0.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    agent_contributions: Dict[str, float] = Field(default_factory=dict)


class ActionRating(BaseModel):
    """행동 평가 결과"""
    action_id: str
    action_type: ActionType
    execution_time: float
    success_score: float = Field(..., ge=0.0, le=1.0)
    user_satisfaction: float = Field(..., ge=0.0, le=1.0)
    emotional_impact: float = Field(..., ge=-1.0, le=1.0)
    needs_fulfillment: float = Field(..., ge=0.0, le=1.0)
    cognitive_efficiency: float = Field(..., ge=0.0, le=1.0)
    learning_value: float = Field(..., ge=0.0, le=1.0)
    collaboration_effectiveness: float = Field(..., ge=0.0, le=1.0)
    overall_rating: float = Field(..., ge=0.0, le=1.0)
    feedback_notes: str = ""


class Action(BaseModel):
    """행동 모델"""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType
    priority: ActionPriority = ActionPriority.Medium
    status: ActionStatus = ActionStatus.Pending
    content: str
    context: ActionContext
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    rating: Optional[ActionRating] = None
    collaborating_agents: List[str] = Field(default_factory=list)


class MultiAgentSystem:
    """멀티 에이전트 시스템"""
    
    def __init__(self):
        self.agents: Dict[str, AgentState] = {}
        self.collaboration_protocols: Dict[str, CollaborationProtocol] = {}
        self.agent_specializations = {
            AgentType.REASONING: ["logical_analysis", "problem_solving", "decision_making"],
            AgentType.EMOTIONAL: ["emotion_analysis", "empathy", "emotional_regulation"],
            AgentType.MEMORY: ["memory_retrieval", "pattern_recognition", "context_analysis"],
            AgentType.PLANNING: ["goal_setting", "strategy_development", "resource_allocation"],
            AgentType.EXECUTION: ["action_execution", "performance_monitoring", "adaptation"],
            AgentType.MONITORING: ["quality_assessment", "progress_tracking", "feedback_analysis"],
            AgentType.LEARNING: ["knowledge_integration", "skill_development", "behavior_adaptation"]
        }
        self._initialize_agents()
    
    def _initialize_agents(self):
        """에이전트 초기화"""
        for agent_type in AgentType:
            agent_id = f"{agent_type.value}_agent_{uuid.uuid4().hex[:8]}"
            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                agent_type=agent_type,
                specialization=self.agent_specializations[agent_type]
            )
    
    def get_available_agents(self, required_specializations: List[str]) -> List[AgentState]:
        """필요한 전문성을 가진 사용 가능한 에이전트 조회"""
        available_agents = []
        
        for agent in self.agents.values():
            if not agent.is_active:
                continue
            
            # 전문성 매칭 확인
            matching_specializations = set(agent.specialization) & set(required_specializations)
            if matching_specializations:
                available_agents.append(agent)
        
        return available_agents
    
    def create_collaboration_protocol(self, task_description: str, 
                                    required_agents: List[str],
                                    coordination_method: str = "sequential") -> CollaborationProtocol:
        """협업 프로토콜 생성"""
        protocol = CollaborationProtocol(
            participants=required_agents,
            task_description=task_description,
            coordination_method=coordination_method,
            communication_channels=[f"channel_{uuid.uuid4().hex[:8]}" for _ in required_agents]
        )
        
        self.collaboration_protocols[protocol.protocol_id] = protocol
        return protocol
    
    def execute_collaborative_action(self, action: Action, protocol: CollaborationProtocol) -> Dict[str, Any]:
        """협업 행동 실행"""
        results = {}
        
        if protocol.coordination_method == "sequential":
            results = self._execute_sequential_collaboration(action, protocol)
        elif protocol.coordination_method == "parallel":
            results = self._execute_parallel_collaboration(action, protocol)
        elif protocol.coordination_method == "hierarchical":
            results = self._execute_hierarchical_collaboration(action, protocol)
        
        # 협업 결과 업데이트
        for agent_id in protocol.participants:
            if agent_id in self.agents:
                self.agents[agent_id].collaboration_history.append(protocol.protocol_id)
                self.agents[agent_id].last_activity = datetime.now()
        
        return results
    
    def _execute_sequential_collaboration(self, action: Action, protocol: CollaborationProtocol) -> Dict[str, Any]:
        """순차적 협업 실행"""
        results = {}
        previous_result = None
        
        for agent_id in protocol.participants:
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            agent.current_task = f"Processing {action.action_type.value}"
            
            # 에이전트별 처리
            agent_result = self._process_with_agent(agent, action, previous_result)
            results[agent_id] = agent_result
            previous_result = agent_result
            
            agent.current_task = None
        
        return results
    
    def _execute_parallel_collaboration(self, action: Action, protocol: CollaborationProtocol) -> Dict[str, Any]:
        """병렬 협업 실행"""
        results = {}
        
        # 모든 에이전트가 동시에 작업
        for agent_id in protocol.participants:
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            agent.current_task = f"Parallel processing {action.action_type.value}"
            
            # 에이전트별 처리
            agent_result = self._process_with_agent(agent, action, None)
            results[agent_id] = agent_result
            
            agent.current_task = None
        
        return results
    
    def _execute_hierarchical_collaboration(self, action: Action, protocol: CollaborationProtocol) -> Dict[str, Any]:
        """계층적 협업 실행"""
        results = {}
        
        # 계층 구조 (첫 번째 에이전트가 코디네이터)
        if protocol.participants:
            coordinator_id = protocol.participants[0]
            worker_ids = protocol.participants[1:]
            
            # 코디네이터가 작업 분배
            if coordinator_id in self.agents:
                coordinator = self.agents[coordinator_id]
                coordinator.current_task = f"Coordinating {action.action_type.value}"
                
                # 작업 분배 로직
                task_distribution = self._distribute_tasks(coordinator, action, worker_ids)
                results[coordinator_id] = {"task_distribution": task_distribution}
                
                # 워커들이 작업 실행
                for worker_id in worker_ids:
                    if worker_id in self.agents:
                        worker = self.agents[worker_id]
                        worker.current_task = f"Executing assigned task for {action.action_type.value}"
                        
                        worker_result = self._process_with_agent(worker, action, task_distribution)
                        results[worker_id] = worker_result
                        
                        worker.current_task = None
                
                coordinator.current_task = None
        
        return results
    
    def _process_with_agent(self, agent: AgentState, action: Action, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """에이전트별 처리"""
        # 에이전트 유형별 처리 로직
        if agent.agent_type == AgentType.REASONING:
            return self._reasoning_agent_process(action, previous_result)
        elif agent.agent_type == AgentType.EMOTIONAL:
            return self._emotional_agent_process(action, previous_result)
        elif agent.agent_type == AgentType.MEMORY:
            return self._memory_agent_process(action, previous_result)
        elif agent.agent_type == AgentType.PLANNING:
            return self._planning_agent_process(action, previous_result)
        elif agent.agent_type == AgentType.EXECUTION:
            return self._execution_agent_process(action, previous_result)
        elif agent.agent_type == AgentType.MONITORING:
            return self._monitoring_agent_process(action, previous_result)
        elif agent.agent_type == AgentType.LEARNING:
            return self._learning_agent_process(action, previous_result)
        
        return {"status": "unknown_agent_type", "agent_id": agent.agent_id}
    
    def _reasoning_agent_process(self, action: Action, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """추론 에이전트 처리"""
        return {
            "agent_type": "reasoning",
            "analysis": f"논리적 분석: {action.action_type.value}",
            "recommendations": ["체계적 접근", "단계별 실행"],
            "confidence": 0.8
        }
    
    def _emotional_agent_process(self, action: Action, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """감정 에이전트 처리"""
        return {
            "agent_type": "emotional",
            "emotional_analysis": f"감정적 영향 분석: {action.action_type.value}",
            "empathy_factors": ["사용자 감정", "상황 민감성"],
            "emotional_balance": 0.7
        }
    
    def _memory_agent_process(self, action: Action, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """메모리 에이전트 처리"""
        return {
            "agent_type": "memory",
            "memory_retrieval": f"관련 기억 검색: {action.action_type.value}",
            "context_analysis": ["과거 경험", "패턴 인식"],
            "relevance_score": 0.6
        }
    
    def _planning_agent_process(self, action: Action, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """계획 에이전트 처리"""
        return {
            "agent_type": "planning",
            "strategy_development": f"전략 수립: {action.action_type.value}",
            "resource_allocation": ["시간", "인지 자원"],
            "plan_quality": 0.75
        }
    
    def _execution_agent_process(self, action: Action, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """실행 에이전트 처리"""
        return {
            "agent_type": "execution",
            "action_execution": f"행동 실행: {action.action_type.value}",
            "performance_metrics": ["정확성", "효율성"],
            "execution_quality": 0.8
        }
    
    def _monitoring_agent_process(self, action: Action, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """모니터링 에이전트 처리"""
        return {
            "agent_type": "monitoring",
            "quality_assessment": f"품질 평가: {action.action_type.value}",
            "progress_tracking": ["진행 상황", "목표 달성도"],
            "monitoring_score": 0.7
        }
    
    def _learning_agent_process(self, action: Action, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """학습 에이전트 처리"""
        return {
            "agent_type": "learning",
            "knowledge_integration": f"지식 통합: {action.action_type.value}",
            "skill_development": ["새로운 기술", "행동 개선"],
            "learning_progress": 0.6
        }
    
    def _distribute_tasks(self, coordinator: AgentState, action: Action, worker_ids: List[str]) -> Dict[str, str]:
        """작업 분배"""
        task_distribution = {}
        
        # 간단한 작업 분배 로직
        tasks = ["분석", "실행", "평가"]
        for i, worker_id in enumerate(worker_ids):
            if i < len(tasks):
                task_distribution[worker_id] = tasks[i]
        
        return task_distribution
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """에이전트 성능 요약"""
        summary = {}
        for agent_id, agent in self.agents.items():
            summary[agent_id] = {
                "agent_type": agent.agent_type.value,
                "is_active": agent.is_active,
                "performance_score": agent.performance_score,
                "collaboration_count": len(agent.collaboration_history),
                "last_activity": agent.last_activity.isoformat(),
                "current_task": agent.current_task
            }
        return summary


class ActionSystem:
    """하린코어 행동 시스템 - PM 시스템의 행동 시스템을 참고하여 구현 + 멀티 에이전트 협업"""
    
    def __init__(self, harin_main_loop: EnhancedHarinMainLoop):
        self.harin = harin_main_loop
        
        # 행동 실행기들
        self.action_executors = self._initialize_action_executors()
        
        # 행동 시뮬레이터
        self.action_simulator = ActionSimulator(harin_main_loop)
        
        # 행동 평가기
        self.action_evaluator = ActionEvaluator(harin_main_loop)
        
        # 멀티 에이전트 시스템 추가
        self.multi_agent_system = MultiAgentSystem()
        
        # 행동 히스토리
        self.action_history: List[Action] = []
        
        # 통계 추적
        self.action_stats = {
            "total_actions": 0,
            "by_type": {action_type.value: 0 for action_type in ActionType},
            "by_status": {status.value: 0 for status in ActionStatus},
            "by_priority": {priority.value: 0 for priority in ActionPriority},
            "avg_execution_time": 0.0,
            "success_rate": 0.0,
            "collaboration_rate": 0.0
        }
    
    def _initialize_action_executors(self) -> Dict[ActionType, callable]:
        """행동 실행기 초기화"""
        return {
            ActionType.Reply: self._execute_reply,
            ActionType.ToolCall: self._execute_tool_call,
            ActionType.Ignore: self._execute_ignore,
            ActionType.Sleep: self._execute_sleep,
            ActionType.InitiateUserConversation: self._execute_initiate_conversation,
            ActionType.InitiateInternalContemplation: self._execute_internal_contemplation,
            ActionType.Think: self._execute_think,
            ActionType.RecallMemory: self._execute_recall_memory,
            ActionType.Plan: self._execute_plan,
            ActionType.ReflectThoughts: self._execute_reflect_thoughts,
            ActionType.AnalyzeEmotion: self._execute_analyze_emotion,
            ActionType.ProcessMemory: self._execute_process_memory,
            ActionType.GenerateInsight: self._execute_generate_insight,
            ActionType.AdaptBehavior: self._execute_adapt_behavior,
            ActionType.LearnFromExperience: self._execute_learn_from_experience,
            ActionType.SystemMaintenance: self._execute_system_maintenance,
            ActionType.DataOptimization: self._execute_data_optimization,
            ActionType.PerformanceMonitoring: self._execute_performance_monitoring,
            # 멀티 에이전트 협업 행동 추가
            ActionType.CoordinateAgents: self._execute_coordinate_agents,
            ActionType.DelegateTask: self._execute_delegate_task,
            ActionType.SynthesizeResults: self._execute_synthesize_results,
            ActionType.ResolveConflict: self._execute_resolve_conflict
        }
    
    def create_action(self, action_type: ActionType, content: str, 
                     context: ActionContext, priority: ActionPriority = ActionPriority.Medium) -> Action:
        """행동 생성"""
        action = Action(
            action_type=action_type,
            priority=priority,
            content=content,
            context=context
        )
        
        print(f"행동 생성: {action_type.value} - {content[:50]}...")
        return action
    
    def simulate_action(self, action: Action, num_simulations: int = 3) -> List[ActionSimulation]:
        """행동 시뮬레이션"""
        return self.action_simulator.simulate_action(action, num_simulations)
    
    def select_best_action(self, actions: List[Action], context: ActionContext) -> Action:
        """최적 행동 선택 - 멀티 에이전트 협업 고려"""
        if not actions:
            raise ValueError("행동 목록이 비어있습니다")
        
        # 각 행동에 대해 시뮬레이션 실행
        action_scores = []
        for action in actions:
            simulations = self.simulate_action(action, num_simulations=2)
            if simulations:
                avg_score = sum(sim.overall_score for sim in simulations) / len(simulations)
                action_scores.append((action, avg_score))
            else:
                action_scores.append((action, 0.0))
        
        # 협업이 필요한 복잡한 행동인지 판단
        complex_actions = [action for action in actions if self._requires_collaboration(action)]
        
        if complex_actions:
            # 협업 기반 선택
            return self._select_collaborative_action(complex_actions, context)
        else:
            # 단순 행동 선택
            return max(action_scores, key=lambda x: x[1])[0]
    
    def _requires_collaboration(self, action: Action) -> bool:
        """협업이 필요한 행동인지 판단"""
        collaboration_required = [
            ActionType.Think,
            ActionType.Plan,
            ActionType.AnalyzeEmotion,
            ActionType.ProcessMemory,
            ActionType.GenerateInsight,
            ActionType.CoordinateAgents,
            ActionType.SynthesizeResults
        ]
        return action.action_type in collaboration_required
    
    def _select_collaborative_action(self, actions: List[Action], context: ActionContext) -> Action:
        """협업 기반 행동 선택"""
        best_action = None
        best_collaboration_score = 0.0
        
        for action in actions:
            # 필요한 전문성 식별
            required_specializations = self._identify_required_specializations(action)
            
            # 사용 가능한 에이전트 조회
            available_agents = self.multi_agent_system.get_available_agents(required_specializations)
            
            if available_agents:
                # 협업 프로토콜 생성
                protocol = self.multi_agent_system.create_collaboration_protocol(
                    task_description=action.content,
                    required_agents=[agent.agent_id for agent in available_agents],
                    coordination_method="hierarchical"
                )
                
                # 협업 시뮬레이션
                collaboration_score = self._simulate_collaboration(action, protocol)
                
                if collaboration_score > best_collaboration_score:
                    best_collaboration_score = collaboration_score
                best_action = action
                    action.collaborating_agents = [agent.agent_id for agent in available_agents]
        
        return best_action or actions[0]
    
    def _identify_required_specializations(self, action: Action) -> List[str]:
        """행동에 필요한 전문성 식별"""
        specialization_mapping = {
            ActionType.Think: ["logical_analysis", "problem_solving"],
            ActionType.Plan: ["goal_setting", "strategy_development"],
            ActionType.AnalyzeEmotion: ["emotion_analysis", "empathy"],
            ActionType.ProcessMemory: ["memory_retrieval", "pattern_recognition"],
            ActionType.GenerateInsight: ["knowledge_integration", "creative_thinking"],
            ActionType.CoordinateAgents: ["leadership", "communication"],
            ActionType.SynthesizeResults: ["analysis", "synthesis"]
        }
        
        return specialization_mapping.get(action.action_type, ["general_processing"])
    
    def _simulate_collaboration(self, action: Action, protocol: CollaborationProtocol) -> float:
        """협업 시뮬레이션"""
        try:
            # 간단한 협업 시뮬레이션
            agent_count = len(protocol.participants)
            coordination_efficiency = 0.8 if protocol.coordination_method == "hierarchical" else 0.6
            
            # 기본 점수 계산
            base_score = 0.5
            agent_bonus = min(agent_count * 0.1, 0.3)  # 최대 30% 보너스
            coordination_bonus = coordination_efficiency * 0.2
            
            total_score = base_score + agent_bonus + coordination_bonus
            return min(total_score, 1.0)
            
        except Exception as e:
            print(f"협업 시뮬레이션 실패: {e}")
            return 0.5
    
    def execute_action(self, action: Action) -> Dict[str, Any]:
        """행동 실행 + 멀티 에이전트 협업 + 모니터링 통합"""
        start_time = time.time()
        
        # === [추가] 모니터링 시작 ===
        self._start_action_monitoring(action)
        # === [기존 기능 유지] ===
        
        try:
            # 1. 행동 검증
            validation_result = self._validate_action(action)
            if not validation_result.is_valid:
            action.status = ActionStatus.Failed
                action.error_message = validation_result.error_message
                return ActionResult(
                    success=False,
                    action=action,
                    execution_time=time.time() - start_time,
                    error_message=validation_result.error_message
                )
            
            # 2. 멀티 에이전트 협업 처리
            if hasattr(action, 'collaborating_agents') and action.collaborating_agents:
                collaboration_result = self._execute_collaborative_action(action)
                if collaboration_result:
                    action = collaboration_result
            
            # 3. 행동 타입별 실행
            if action.action_type == ActionType.COGNITIVE:
                result = self._execute_cognitive_action(action)
            elif action.action_type == ActionType.EMOTIONAL:
                result = self._execute_emotional_action(action)
            elif action.action_type == ActionType.MEMORY:
                result = self._execute_memory_action(action)
            elif action.action_type == ActionType.COLLABORATIVE:
                result = self._execute_collaborative_action(action)
            elif action.action_type == ActionType.SYSTEM:
                result = self._execute_system_action(action)
            else:
                result = self._execute_generic_action(action)
            
            # 4. 실행 시간 기록
            execution_time = time.time() - start_time
            action.execution_time = execution_time
            
            # 5. 결과 처리
            if result.success:
                action.status = ActionStatus.Completed
                action.result_data = result.data
        else:
                action.status = ActionStatus.Failed
                action.error_message = result.error_message
            
            # === [추가] 모니터링 완료 ===
            self._complete_action_monitoring(action, result, execution_time)
            # === [기존 기능 유지] ===
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"행동 실행 중 오류 발생: {str(e)}"
            
            action.status = ActionStatus.Failed
            action.error_message = error_message
            action.execution_time = execution_time
            
            # === [추가] 모니터링 오류 ===
            self._handle_action_monitoring_error(action, error_message, execution_time)
            # === [기존 기능 유지] ===
            
            return ActionResult(
                success=False,
                action=action,
                execution_time=execution_time,
                error_message=error_message
            )
    
    def _start_action_monitoring(self, action: Action):
        """행동 모니터링 시작"""
        try:
            if hasattr(self, 'monitoring_system') and self.monitoring_system:
                # 행동 시작 메트릭 기록
                action_metrics = {
                    "action_type": action.action_type.value,
                    "action_id": action.action_id,
                    "start_time": time.time(),
                    "collaboration_count": len(action.collaborating_agents) if hasattr(action, 'collaborating_agents') else 0,
                    "priority": action.priority.value if hasattr(action, 'priority') else "normal"
                }
                
                # 모니터링 시스템에 행동 시작 기록
                self.monitoring_system.record_action_start(action_metrics)
                
        except Exception as e:
            print(f"행동 모니터링 시작 오류: {e}")
    
    def _complete_action_monitoring(self, action: Action, result: ActionResult, execution_time: float):
        """행동 모니터링 완료"""
        try:
            if hasattr(self, 'monitoring_system') and self.monitoring_system:
                # 행동 완료 메트릭 기록
                action_metrics = {
                    "action_type": action.action_type.value,
                    "action_id": action.action_id,
                    "execution_time": execution_time,
                    "success": result.success,
                    "status": action.status.value,
                    "collaboration_count": len(action.collaborating_agents) if hasattr(action, 'collaborating_agents') else 0,
                    "result_data_size": len(str(result.data)) if result.data else 0
                }
                
                # 모니터링 시스템에 행동 완료 기록
                self.monitoring_system.record_action_completion(action_metrics)
                
                # 협업 메트릭 기록
                if hasattr(action, 'collaborating_agents') and action.collaborating_agents:
                    collaboration_metrics = {
                        "agent_count": len(action.collaborating_agents),
                        "collaboration_efficiency": self._calculate_collaboration_efficiency(action),
                        "task_distribution": self._analyze_task_distribution(action),
                        "communication_overhead": self._calculate_communication_overhead(action)
                    }
                    
                    self.monitoring_system.record_collaboration_metrics(collaboration_metrics)
                
        except Exception as e:
            print(f"행동 모니터링 완료 오류: {e}")
    
    def _handle_action_monitoring_error(self, action: Action, error_message: str, execution_time: float):
        """행동 모니터링 오류 처리"""
        try:
            if hasattr(self, 'monitoring_system') and self.monitoring_system:
                # 오류 메트릭 기록
                error_metrics = {
                    "action_type": action.action_type.value,
                    "action_id": action.action_id,
                    "execution_time": execution_time,
                    "error_message": error_message,
                    "error_type": "execution_error"
                }
                
                # 모니터링 시스템에 오류 기록
                self.monitoring_system.record_action_error(error_metrics)
                
        except Exception as e:
            print(f"행동 모니터링 오류 처리 실패: {e}")
    
    def _calculate_collaboration_efficiency(self, action: Action) -> float:
        """협업 효율성 계산"""
        if not hasattr(action, 'collaborating_agents') or not action.collaborating_agents:
            return 0.0
        
        # 간단한 효율성 계산 (실제로는 더 정교한 로직 필요)
        agent_count = len(action.collaborating_agents)
        execution_time = action.execution_time or 1.0
        
        # 에이전트 수와 실행 시간을 기반으로 효율성 계산
        # 에이전트가 많을수록, 실행 시간이 짧을수록 효율적
        efficiency = min(1.0, (agent_count * 0.2) / max(execution_time, 0.1))
        
        return efficiency
    
    def _analyze_task_distribution(self, action: Action) -> Dict[str, Any]:
        """작업 분배 분석"""
        if not hasattr(action, 'collaborating_agents') or not action.collaborating_agents:
            return {"distribution": "none", "balance": 0.0}
        
        # 에이전트별 작업 분배 분석
        agent_tasks = {}
        for agent in action.collaborating_agents:
            agent_id = agent.get('agent_id', 'unknown')
            task_count = agent.get('task_count', 1)
            agent_tasks[agent_id] = task_count
        
        # 분배 균형 계산
        if agent_tasks:
            task_counts = list(agent_tasks.values())
            avg_tasks = sum(task_counts) / len(task_counts)
            variance = sum((count - avg_tasks) ** 2 for count in task_counts) / len(task_counts)
            balance = max(0.0, 1.0 - (variance / avg_tasks)) if avg_tasks > 0 else 0.0
        else:
            balance = 0.0
        
        return {
            "distribution": "distributed" if len(agent_tasks) > 1 else "single",
            "balance": balance,
            "agent_tasks": agent_tasks
        }
    
    def _calculate_communication_overhead(self, action: Action) -> float:
        """통신 오버헤드 계산"""
        if not hasattr(action, 'collaborating_agents') or not action.collaborating_agents:
            return 0.0
        
        # 에이전트 수에 따른 통신 오버헤드 계산
        agent_count = len(action.collaborating_agents)
        
        # n*(n-1)/2 형태의 통신 복잡도
        communication_pairs = (agent_count * (agent_count - 1)) / 2
        overhead = min(1.0, communication_pairs / 10.0)  # 정규화
        
        return overhead
    
    def set_monitoring_system(self, monitoring_system):
        """모니터링 시스템 설정"""
        self.monitoring_system = monitoring_system
        print("모니터링 시스템이 행동 시스템에 연결되었습니다.")
    
    def get_action_monitoring_summary(self) -> Dict[str, Any]:
        """행동 모니터링 요약 조회"""
        if not hasattr(self, 'monitoring_system') or not self.monitoring_system:
            return {"status": "monitoring_not_available"}
        
        try:
            # 행동 통계 수집
            action_stats = {
                "total_actions": len(self.action_history),
                "successful_actions": len([a for a in self.action_history if a.status == ActionStatus.Completed]),
                "failed_actions": len([a for a in self.action_history if a.status == ActionStatus.Failed]),
                "collaborative_actions": len([a for a in self.action_history if hasattr(a, 'collaborating_agents') and a.collaborating_agents])
            }
            
            # 평균 실행 시간 계산
            completed_actions = [a for a in self.action_history if a.execution_time]
            if completed_actions:
                action_stats["avg_execution_time"] = sum(a.execution_time for a in completed_actions) / len(completed_actions)
            else:
                action_stats["avg_execution_time"] = 0.0
            
            # 행동 타입별 통계
            action_type_stats = {}
            for action_type in ActionType:
                type_actions = [a for a in self.action_history if a.action_type == action_type]
                action_type_stats[action_type.value] = {
                    "count": len(type_actions),
                    "success_rate": len([a for a in type_actions if a.status == ActionStatus.Completed]) / max(len(type_actions), 1)
                }
        
        return {
                "status": "success",
                "action_stats": action_stats,
                "action_type_stats": action_type_stats,
                "monitoring_metrics": self.monitoring_system.get_metrics_summary(
                    metric_type=None, time_window=3600  # 최근 1시간
                )
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_action_dashboard(self, filepath: str):
        """행동 대시보드 생성"""
        if not hasattr(self, 'monitoring_system') or not self.monitoring_system:
            print("모니터링 시스템이 연결되지 않았습니다.")
            return
        
        try:
            # 행동 메트릭 데이터 수집
            performance_metrics = self.monitoring_system.metrics.get('performance', [])
            collaboration_metrics = self.monitoring_system.metrics.get('collaboration', [])
            
            if not performance_metrics and not collaboration_metrics:
                print("행동 메트릭 데이터가 없습니다.")
                return
            
            # Plotly를 사용한 대시보드 생성
            if VISUALIZATION_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('행동 성능', '협업 효율성', '행동 타입별 성공률', '실행 시간 분포'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # 행동 성능
                if performance_metrics:
                    timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in performance_metrics]
                    values = [mp.value for mp in performance_metrics]
                    
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=values, name="행동 성능", line=dict(color='blue')),
                        row=1, col=1
                    )
                
                # 협업 효율성
                if collaboration_metrics:
                    timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in collaboration_metrics]
                    values = [mp.value for mp in collaboration_metrics]
                    
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=values, name="협업 효율성", line=dict(color='green')),
                        row=1, col=2
                    )
                
                # 행동 타입별 성공률
                action_type_stats = self.get_action_monitoring_summary().get('action_type_stats', {})
                if action_type_stats:
                    action_types = list(action_type_stats.keys())
                    success_rates = [stats['success_rate'] for stats in action_type_stats.values()]
                    
                    fig.add_trace(
                        go.Bar(x=action_types, y=success_rates, name="성공률", marker_color='orange'),
                        row=2, col=1
                    )
                
                # 실행 시간 분포
                completed_actions = [a for a in self.action_history if a.execution_time]
                if completed_actions:
                    execution_times = [a.execution_time for a in completed_actions]
                    
                    fig.add_trace(
                        go.Histogram(x=execution_times, name="실행 시간", marker_color='red'),
                        row=2, col=2
                    )
                
                # 레이아웃 설정
                fig.update_layout(
                    title="하린코어 행동 모니터링 대시보드",
                    height=800,
                    showlegend=True
                )
                
                # HTML 파일로 저장
                fig.write_html(filepath)
                print(f"행동 대시보드 생성 완료: {filepath}")
            else:
                print("시각화 라이브러리가 없어 대시보드를 생성할 수 없습니다.")
                
        except Exception as e:
            print(f"행동 대시보드 생성 실패: {e}")
    
    def export_action_data(self, filepath: str):
        """행동 데이터 내보내기"""
        try:
            # 행동 히스토리 데이터 변환
            action_history_data = []
            for action in self.action_history:
                action_data = {
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "status": action.status.value,
                    "execution_time": action.execution_time,
                    "error_message": action.error_message,
                    "timestamp": action.timestamp.isoformat() if hasattr(action, 'timestamp') else None,
                    "collaborating_agents": action.collaborating_agents if hasattr(action, 'collaborating_agents') else []
                }
                action_history_data.append(action_data)
            
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "action_history": action_history_data,
                "action_summary": self.get_action_monitoring_summary()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"행동 데이터 내보내기 완료: {filepath}")
            
        except Exception as e:
            print(f"행동 데이터 내보내기 실패: {e}")


class ActionSimulator:
    """행동 시뮬레이터"""
    
    def __init__(self, harin_main_loop: EnhancedHarinMainLoop):
        self.harin = harin_main_loop
    
    def simulate_action(self, action: Action, num_simulations: int = 3) -> List[ActionSimulation]:
        """행동 시뮬레이션"""
        simulations = []
        
        for i in range(num_simulations):
            sim_id = f"sim_{action.action_id}_{i}"
            
            # 시뮬레이션 실행
            simulation = self._run_single_simulation(action, sim_id)
            simulations.append(simulation)
        
        return simulations
    
    def _run_single_simulation(self, action: Action, sim_id: str) -> ActionSimulation:
        """단일 시뮬레이션 실행"""
        # 예측된 사용자 반응
        predicted_user_reaction = self._predict_user_reaction(action)
        
        # 예측된 감정적 영향
        predicted_emotional_impact = self._predict_emotional_impact(action)
        
        # 예측된 욕구 영향
        predicted_needs_impact = self._predict_needs_impact(action)
        
        # 예측된 인지적 영향
        predicted_cognitive_impact = self._predict_cognitive_impact(action)
        
        # 성공 확률 계산
        success_probability = self._calculate_success_probability(action)
        
        # 예상 이익 계산
        expected_benefit = self._calculate_expected_benefit(action)
        
        # 위험도 계산
        risk_level = self._calculate_risk_level(action)
        
        # 인지 부하 계산
        cognitive_load = self._calculate_cognitive_load(action)
        
        # 예상 실행 시간
        execution_time = self._estimate_execution_time(action)
        
        # 전체 점수 계산
        overall_score = self._calculate_overall_score(
            success_probability, expected_benefit, risk_level, cognitive_load
        )
        
        return ActionSimulation(
            simulation_id=sim_id,
            action_type=action.action_type,
            action_content=action.content,
            predicted_user_reaction=predicted_user_reaction,
            predicted_emotional_impact=predicted_emotional_impact,
            predicted_needs_impact=predicted_needs_impact,
            predicted_cognitive_impact=predicted_cognitive_impact,
            success_probability=success_probability,
            expected_benefit=expected_benefit,
            risk_level=risk_level,
            cognitive_load=cognitive_load,
            execution_time=execution_time,
            overall_score=overall_score
        )
    
    def _predict_user_reaction(self, action: Action) -> str:
        """사용자 반응 예측"""
        # 간단한 예측 로직 (실제로는 더 정교한 모델 사용)
        if action.action_type == ActionType.Reply:
            return "사용자가 응답에 만족할 것으로 예상됩니다."
        elif action.action_type == ActionType.ToolCall:
            return "사용자가 도구 사용 결과에 관심을 보일 것으로 예상됩니다."
        elif action.action_type == ActionType.Ignore:
            return "사용자가 반응이 없을 것으로 예상됩니다."
        else:
            return "사용자의 반응을 예측하기 어렵습니다."
    
    def _predict_emotional_impact(self, action: Action) -> EmotionalAxesModel:
        """감정적 영향 예측"""
        # 기본 감정 상태 복사
        emotional_impact = EmotionalAxesModel()
        
        # 행동 유형에 따른 감정적 영향 예측
        if action.action_type == ActionType.Reply:
            emotional_impact.joy = 0.1
            emotional_impact.contentment = 0.05
        elif action.action_type == ActionType.ToolCall:
            emotional_impact.curiosity = 0.1
        elif action.action_type == ActionType.Ignore:
            emotional_impact.frustration = -0.1
        
        return emotional_impact
    
    def _predict_needs_impact(self, action: Action) -> NeedsAxesModel:
        """욕구 영향 예측"""
        # 기본 욕구 상태 복사
        needs_impact = NeedsAxesModel()
        
        # 행동 유형에 따른 욕구 영향 예측
        if action.action_type == ActionType.Reply:
            needs_impact.connection = 0.1
            needs_impact.relevance = 0.05
        elif action.action_type == ActionType.ToolCall:
            needs_impact.relevance = 0.1
        elif action.action_type == ActionType.LearnFromExperience:
            needs_impact.learning_growth = 0.1
        
        return needs_impact
    
    def _predict_cognitive_impact(self, action: Action) -> CognitionAxesModel:
        """인지적 영향 예측"""
        # 기본 인지 상태 복사
        cognitive_impact = CognitionAxesModel()
        
        # 행동 유형에 따른 인지적 영향 예측
        if action.action_type == ActionType.Think:
            cognitive_impact.analytical_thinking = 0.1
        elif action.action_type == ActionType.Plan:
            cognitive_impact.strategic_planning = 0.1
        elif action.action_type == ActionType.ReflectThoughts:
            cognitive_impact.metacognition = 0.1
        
        return cognitive_impact
    
    def _calculate_success_probability(self, action: Action) -> float:
        """성공 확률 계산"""
        base_probability = 0.8  # 기본 성공 확률
        
        # 행동 유형에 따른 조정
        if action.action_type == ActionType.Reply:
            base_probability = 0.9
        elif action.action_type == ActionType.ToolCall:
            base_probability = 0.7
        elif action.action_type == ActionType.Ignore:
            base_probability = 1.0
        
        return min(1.0, base_probability)
    
    def _calculate_expected_benefit(self, action: Action) -> float:
        """예상 이익 계산"""
        base_benefit = 0.5  # 기본 이익
        
        # 행동 유형에 따른 조정
        if action.action_type == ActionType.Reply:
            base_benefit = 0.8
        elif action.action_type == ActionType.ToolCall:
            base_benefit = 0.7
        elif action.action_type == ActionType.LearnFromExperience:
            base_benefit = 0.6
        
        return min(1.0, base_benefit)
    
    def _calculate_risk_level(self, action: Action) -> float:
        """위험도 계산"""
        base_risk = 0.2  # 기본 위험도
        
        # 행동 유형에 따른 조정
        if action.action_type == ActionType.ToolCall:
            base_risk = 0.4
        elif action.action_type == ActionType.Ignore:
            base_risk = 0.1
        
        return min(1.0, base_risk)
    
    def _calculate_cognitive_load(self, action: Action) -> float:
        """인지 부하 계산"""
        base_load = 0.3  # 기본 인지 부하
        
        # 행동 유형에 따른 조정
        if action.action_type == ActionType.Think:
            base_load = 0.8
        elif action.action_type == ActionType.Plan:
            base_load = 0.7
        elif action.action_type == ActionType.Ignore:
            base_load = 0.1
        
        return min(1.0, base_load)
    
    def _estimate_execution_time(self, action: Action) -> float:
        """실행 시간 예측 (초)"""
        base_time = 1.0  # 기본 실행 시간
        
        # 행동 유형에 따른 조정
        if action.action_type == ActionType.Reply:
            base_time = 2.0
        elif action.action_type == ActionType.ToolCall:
            base_time = 5.0
        elif action.action_type == ActionType.Think:
            base_time = 3.0
        elif action.action_type == ActionType.Ignore:
            base_time = 0.1
        
        return base_time
    
    def _calculate_overall_score(self, success_probability: float, expected_benefit: float,
                                risk_level: float, cognitive_load: float) -> float:
        """전체 점수 계산"""
        # 가중치 설정
        w_success = 0.3
        w_benefit = 0.4
        w_risk = 0.2
        w_load = 0.1
        
        # 점수 계산 (위험도와 인지 부하는 낮을수록 좋음)
        score = (w_success * success_probability + 
                w_benefit * expected_benefit + 
                w_risk * (1.0 - risk_level) + 
                w_load * (1.0 - cognitive_load))
        
        return min(1.0, max(0.0, score))


class ActionEvaluator:
    """행동 평가기"""
    
    def __init__(self, harin_main_loop: EnhancedHarinMainLoop):
        self.harin = harin_main_loop
    
    def evaluate_action(self, action: Action, result: Dict[str, Any]) -> ActionRating:
        """행동 평가"""
        # 실행 시간
        execution_time = action.execution_time or 0.0
        
        # 성공 점수
        success_score = 1.0 if action.status == ActionStatus.Completed else 0.0
        
        # 사용자 만족도 (간단한 추정)
        user_satisfaction = self._estimate_user_satisfaction(action, result)
        
        # 감정적 영향
        emotional_impact = self._calculate_emotional_impact(action, result)
        
        # 욕구 충족도
        needs_fulfillment = self._calculate_needs_fulfillment(action, result)
        
        # 인지 효율성
        cognitive_efficiency = self._calculate_cognitive_efficiency(action, result)
        
        # 학습 가치
        learning_value = self._calculate_learning_value(action, result)
        
        # 전체 평점
        overall_rating = self._calculate_overall_rating(
            success_score, user_satisfaction, emotional_impact, 
            needs_fulfillment, cognitive_efficiency, learning_value
        )
        
        return ActionRating(
            action_id=action.action_id,
            action_type=action.action_type,
            execution_time=execution_time,
            success_score=success_score,
            user_satisfaction=user_satisfaction,
            emotional_impact=emotional_impact,
            needs_fulfillment=needs_fulfillment,
            cognitive_efficiency=cognitive_efficiency,
            learning_value=learning_value,
            overall_rating=overall_rating,
            feedback_notes=f"행동 {action.action_type.value} 평가 완료"
        )
    
    def _estimate_user_satisfaction(self, action: Action, result: Dict[str, Any]) -> float:
        """사용자 만족도 추정"""
        base_satisfaction = 0.7  # 기본 만족도
        
        # 행동 유형에 따른 조정
        if action.action_type == ActionType.Reply:
            base_satisfaction = 0.8
        elif action.action_type == ActionType.ToolCall:
            base_satisfaction = 0.7
        elif action.action_type == ActionType.Ignore:
            base_satisfaction = 0.3
        
        return min(1.0, base_satisfaction)
    
    def _calculate_emotional_impact(self, action: Action, result: Dict[str, Any]) -> float:
        """감정적 영향 계산"""
        base_impact = 0.0  # 중립
        
        # 행동 유형에 따른 조정
        if action.action_type == ActionType.Reply:
            base_impact = 0.1
        elif action.action_type == ActionType.ToolCall:
            base_impact = 0.05
        elif action.action_type == ActionType.Ignore:
            base_impact = -0.1
        
        return max(-1.0, min(1.0, base_impact))
    
    def _calculate_needs_fulfillment(self, action: Action, result: Dict[str, Any]) -> float:
        """욕구 충족도 계산"""
        base_fulfillment = 0.5  # 기본 충족도
        
        # 행동 유형에 따른 조정
        if action.action_type == ActionType.Reply:
            base_fulfillment = 0.7
        elif action.action_type == ActionType.ToolCall:
            base_fulfillment = 0.6
        elif action.action_type == ActionType.LearnFromExperience:
            base_fulfillment = 0.8
        
        return min(1.0, base_fulfillment)
    
    def _calculate_cognitive_efficiency(self, action: Action, result: Dict[str, Any]) -> float:
        """인지 효율성 계산"""
        base_efficiency = 0.7  # 기본 효율성
        
        # 실행 시간에 따른 조정
        if action.execution_time:
            if action.execution_time < 1.0:
                base_efficiency += 0.1
            elif action.execution_time > 5.0:
                base_efficiency -= 0.2
        
        return min(1.0, max(0.0, base_efficiency))
    
    def _calculate_learning_value(self, action: Action, result: Dict[str, Any]) -> float:
        """학습 가치 계산"""
        base_learning = 0.3  # 기본 학습 가치
        
        # 행동 유형에 따른 조정
        if action.action_type == ActionType.LearnFromExperience:
            base_learning = 0.9
        elif action.action_type == ActionType.Think:
            base_learning = 0.7
        elif action.action_type == ActionType.ReflectThoughts:
            base_learning = 0.8
        
        return min(1.0, base_learning)
    
    def _calculate_overall_rating(self, success_score: float, user_satisfaction: float,
                                 emotional_impact: float, needs_fulfillment: float,
                                 cognitive_efficiency: float, learning_value: float) -> float:
        """전체 평점 계산"""
        # 가중치 설정
        w_success = 0.25
        w_satisfaction = 0.25
        w_emotion = 0.15
        w_needs = 0.15
        w_efficiency = 0.1
        w_learning = 0.1
        
        # 감정적 영향은 절댓값으로 변환
        emotional_score = abs(emotional_impact)
        
        # 점수 계산
        overall = (w_success * success_score + 
                  w_satisfaction * user_satisfaction + 
                  w_emotion * emotional_score + 
                  w_needs * needs_fulfillment + 
                  w_efficiency * cognitive_efficiency + 
                  w_learning * learning_value)
        
        return min(1.0, max(0.0, overall))


# 싱글톤 인스턴스 관리
_instance = None

def get_action_system(harin_main_loop: EnhancedHarinMainLoop) -> ActionSystem:
    """행동 시스템 싱글톤 인스턴스 반환"""
    global _instance
    if _instance is None:
        print("--- 싱글톤 행동 시스템 인스턴스 생성 ---")
        _instance = ActionSystem(harin_main_loop)
    return _instance 