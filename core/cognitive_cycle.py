"""
Harin Core Cognitive Cycle - Lida Integration with LangGraph State Management
심리학적으로 타당한 인지 사이클을 통합한 AI 동반자 시스템
LangGraph 기반 상태 관리 및 조건부 엣지 시스템 추가
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pydantic import BaseModel, Field
import math
import random
import numpy as np
import asyncio
import re
try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("sklearn not available, using fallback clustering")
    class AgglomerativeClustering:
        def __init__(self, *args, **kwargs):
            pass
        def fit_predict(self, *args, **kwargs):
            return [0] * len(args[0]) if args else []
from enum import Enum
import uuid

# LangGraph 관련 import 추가
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available, using fallback state management")
    # 폴백을 위한 더미 클래스들
    class StateGraph:
        def __init__(self, *args, **kwargs):
            pass
        def add_node(self, *args, **kwargs):
            pass
        def add_edge(self, *args, **kwargs):
            pass
        def add_conditional_edges(self, *args, **kwargs):
            pass
        def set_checkpointer(self, *args, **kwargs):
            pass
        def compile(self):
            return self
        def invoke(self, *args, **kwargs):
            return {}
    
    class END:
        pass
    
    class ToolNode:
        pass
    
    class MemorySaver:
        pass

from memory.models import (
    EmotionalAxesModel, NeedsAxesModel, CognitionAxesModel,
    StateDeltas, Stimulus, Intention, Action, Feature, Narrative,
    KnoxelList, GhostState, StimulusType, ActionType, FeatureType,
    StimulusTriage, CognitiveEventTriggers
)

# PM Machine 기반 시스템들 import 추가
from core.state_models import MentalStateManager, StateDeltas
from core.graph_memory import CognitiveMemoryManager
from core.gwt_agents import GWTAgentManager
from core.meta_learning import MetaLearningManager
from core.tree_of_thoughts import TreeOfThoughtsManager, ThoughtType

# LIDA 시스템들 import 추가 (지연 import)
from core.perception_system import PerceptionSystem, PatternType, PerceptionResult
from core.consciousness_system import ConsciousnessSystem
from core.decision_system import DecisionSystem

# 기존 import 수정 - 폴백 처리
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # KMeans 대체 클래스
    class KMeans:
        def __init__(self, n_clusters=3):
            self.n_clusters = n_clusters
        def fit(self, X):
            return self
        def predict(self, X):
            return [0] * len(X)

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    LANGRAPH_AVAILABLE = True
except ImportError:
    LANGRAPH_AVAILABLE = False
    # LangGraph 대체 클래스들
    class StateGraph:
        def __init__(self, *args, **kwargs):
            pass
        def add_node(self, *args, **kwargs):
            return self
        def add_edge(self, *args, **kwargs):
            return self
        def compile(self, *args, **kwargs):
            return self
        def invoke(self, *args, **kwargs):
            return {}
    
    class ToolNode:
        def __init__(self, *args, **kwargs):
            pass
    
    class MemorySaver:
        def __init__(self, *args, **kwargs):
            pass
    
    END = "END"

class CognitiveStateType(Enum):
    """인지 상태 유형"""
    ATTENTION = "attention"
    PERCEPTION = "perception"
    REASONING = "reasoning"
    DECISION = "decision"
    ACTION = "action"
    LEARNING = "learning"
    INTEGRATION = "integration"
    CONSCIOUSNESS = "consciousness"


class CognitiveTransition(BaseModel):
    """인지 상태 전환"""
    from_state: CognitiveStateType
    to_state: CognitiveStateType
    condition: str
    probability: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CognitiveCycleConfig(BaseModel):
    """인지 사이클 설정"""
    companion_name: str = "Harin"
    user_name: str = "User"
    default_decay_factor: float = 0.95
    retrieval_limit_episodic: int = 8
    retrieval_limit_facts: int = 16
    retrieval_limit_features_context: int = 16
    retrieval_limit_features_causal: int = 256
    context_events_similarity_max_tokens: int = 512
    short_term_intent_count: int = 3
    retrieval_limit_expectations: int = 5
    expectation_relevance_decay: float = 0.8
    expectation_generation_count: int = 2
    min_simulations_per_reply: int = 1
    mid_simulations_per_reply: int = 2
    max_simulations_per_reply: int = 3
    importance_threshold_more_sims: float = 0.5
    force_assistant: bool = False
    remember_per_category_limit: int = 4
    coalition_aux_rating_factor: float = 0.03
    cognition_delta_factor: float = 0.33
    
    # LangGraph 설정 추가
    use_langgraph: bool = True
    enable_conditional_edges: bool = True
    state_persistence: bool = True
    max_state_history: int = 100


class CognitiveCycle:
    """인지 사이클 메인 클래스 - Lida의 Ghost 시스템 기반 + LangGraph 상태 관리"""
    
    def __init__(self, config: CognitiveCycleConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.tick_id = 0
        self.states: List[GhostState] = []
        
        # 현재 상태
        self.current_emotions = EmotionalAxesModel()
        self.current_needs = NeedsAxesModel()
        self.current_cognition = CognitionAxesModel()
        
        # Knoxel 저장소
        self.knoxels: List[Any] = []
        self.narratives: Dict[str, Narrative] = {}
        
        # 임베딩 모델 (실제로는 sentence-transformers 사용)
        self.embedding_model = None
        
        # 메타인지 통찰 저장소(임시, 추후 확장)
        self.meta_insights: List[Feature] = []
        
        # LangGraph 상태 관리 초기화
        self.langgraph_state = None
        self.state_graph = None
        self.state_transitions: List[CognitiveTransition] = []
        
        # PM Machine 기반 시스템들 초기화
        self.mental_state_manager = MentalStateManager()
        self.cognitive_memory_manager = CognitiveMemoryManager()
        self.gwt_agent_manager = GWTAgentManager()
        self.meta_learning_manager = MetaLearningManager()
        self.tree_of_thoughts_manager = TreeOfThoughtsManager()
        
        # LIDA 시스템들 초기화 (지연 import)
        self.perception_system = None
        self.consciousness_system = None
        self.decision_system = None
        
        # 통합 모니터링 시스템
        self.monitoring_system = None
        
        if self.config.use_langgraph and LANGGRAPH_AVAILABLE:
            self._initialize_langgraph_state_management()
        else:
            self._initialize_fallback_state_management()
    
    def _initialize_langgraph_state_management(self):
        """LangGraph 기반 상태 관리 초기화"""
        try:
            # 상태 그래프 정의
            self.state_graph = StateGraph(CognitiveStateType)
            
            # 상태 노드 추가
            self.state_graph.add_node(CognitiveStateType.ATTENTION, self._attention_node)
            self.state_graph.add_node(CognitiveStateType.PERCEPTION, self._perception_node)
            self.state_graph.add_node(CognitiveStateType.REASONING, self._reasoning_node)
            self.state_graph.add_node(CognitiveStateType.DECISION, self._decision_node)
            self.state_graph.add_node(CognitiveStateType.ACTION, self._action_node)
            self.state_graph.add_node(CognitiveStateType.LEARNING, self._learning_node)
            self.state_graph.add_node(CognitiveStateType.INTEGRATION, self._integration_node)
            self.state_graph.add_node(CognitiveStateType.CONSCIOUSNESS, self._consciousness_node)
            
            # 조건부 엣지 설정
            if self.config.enable_conditional_edges:
                self._setup_conditional_edges()
            else:
                self._setup_linear_edges()
            
            # 상태 저장 설정
            if self.config.state_persistence:
                memory = MemorySaver()
                self.state_graph.set_checkpointer(memory)
            
            # 컴파일
            self.state_graph = self.state_graph.compile()
            
        except Exception as e:
            print(f"LangGraph 초기화 실패: {e}")
            self.state_graph = None
            self._initialize_fallback_state_management()
    
    def _initialize_fallback_state_management(self):
        """LangGraph 없을 때 폴백 상태 관리"""
        self.current_cognitive_state = CognitiveStateType.ATTENTION
        self.state_history: List[Dict[str, Any]] = []
    
    def _setup_conditional_edges(self):
        """조건부 엣지 설정"""
        if not self.state_graph:
            return
            
        # 주의 → 지각 (항상)
        self.state_graph.add_edge(CognitiveStateType.ATTENTION, CognitiveStateType.PERCEPTION)
        
        # 지각 → 추론 (중요도에 따라)
        self.state_graph.add_conditional_edges(
            CognitiveStateType.PERCEPTION,
            self._should_reason,
            {
                CognitiveStateType.REASONING: "important_stimulus",
                CognitiveStateType.DECISION: "simple_stimulus"
            }
        )
        
        # 추론 → 결정 (복잡도에 따라)
        self.state_graph.add_conditional_edges(
            CognitiveStateType.REASONING,
            self._should_deliberate,
            {
                CognitiveStateType.DECISION: "complex_reasoning_complete",
                CognitiveStateType.REASONING: "continue_reasoning"
            }
        )
        
        # 결정 → 행동 (행동 유형에 따라)
        self.state_graph.add_conditional_edges(
            CognitiveStateType.DECISION,
            self._action_type_router,
            {
                CognitiveStateType.ACTION: "execute_action",
                CognitiveStateType.LEARNING: "learning_action",
                CognitiveStateType.INTEGRATION: "integration_action"
            }
        )
        
        # 행동 → 학습 (결과에 따라)
        self.state_graph.add_conditional_edges(
            CognitiveStateType.ACTION,
            self._should_learn,
            {
                CognitiveStateType.LEARNING: "action_completed",
                CognitiveStateType.INTEGRATION: "action_failed"
            }
        )
        
        # 학습/통합 → 주의 (사이클 완료)
        self.state_graph.add_edge(CognitiveStateType.LEARNING, CognitiveStateType.INTEGRATION)
        self.state_graph.add_edge(CognitiveStateType.INTEGRATION, END)
    
    def _setup_linear_edges(self):
        """선형 엣지 설정"""
        if not self.state_graph:
            return
            
        self.state_graph.add_edge(CognitiveStateType.ATTENTION, CognitiveStateType.PERCEPTION)
        self.state_graph.add_edge(CognitiveStateType.PERCEPTION, CognitiveStateType.REASONING)
        self.state_graph.add_edge(CognitiveStateType.REASONING, CognitiveStateType.DECISION)
        self.state_graph.add_edge(CognitiveStateType.DECISION, CognitiveStateType.ACTION)
        self.state_graph.add_edge(CognitiveStateType.ACTION, CognitiveStateType.LEARNING)
        self.state_graph.add_edge(CognitiveStateType.LEARNING, CognitiveStateType.INTEGRATION)
        self.state_graph.add_edge(CognitiveStateType.INTEGRATION, END)
    
    # 상태 노드 함수들
    def _attention_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """주의 노드"""
        stimulus = state.get("stimulus")
        if not stimulus:
            return {**state, "cognitive_state": CognitiveStateType.ATTENTION}
            
        attention_candidates = self._gather_attention_candidates(stimulus)
        
        return {
            **state,
            "attention_candidates": attention_candidates,
            "attention_focus": self._select_attention_focus(attention_candidates),
            "cognitive_state": CognitiveStateType.ATTENTION
        }
    
    def _perception_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """지각 노드 - LIDA PerceptionSystem 사용 및 결과 연동"""
        stimulus = state.get("stimulus")
        attention_focus = state.get("attention_focus")
        
        if not stimulus:
            return {**state, "cognitive_state": CognitiveStateType.PERCEPTION}
        
        # LIDA PerceptionSystem 사용
        perception_result = self.perception_system.process_stimulus(
            stimulus=stimulus,
            attention_focus=attention_focus,
            context=state.get("context", {})
        )
        
        # GhostState에 perception_result 기록
        if "ghost_state" in state and hasattr(state["ghost_state"], "perception_result"):
            state["ghost_state"].perception_result = perception_result
        
        # CognitiveMemoryManager에 perception_result 저장 (중요 정보만)
        if hasattr(self, "cognitive_memory_manager"):
            self.cognitive_memory_manager.integrate_experience(
                text_from_interaction=str(stimulus.content),
                ai_internal_state_text=f"perception: {perception_result.pattern_type}, importance: {perception_result.importance_score}"
            )
        
        return {
            **state,
            "perception_result": perception_result,
            "stimulus_importance": perception_result.importance_score,
            "cognitive_state": CognitiveStateType.PERCEPTION
        }
    
    def _reasoning_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """추론 노드"""
        stimulus = state.get("stimulus")
        perception_result = state.get("perception_result")
        
        if not stimulus or not perception_result:
            return {**state, "cognitive_state": CognitiveStateType.REASONING}
            
        # 기존 reasoning 로직
        result = {}
        # AdaptiveReasoningLoop 전략 결정
        try:
            from reasoning.adaptive_loop import AdaptiveReasoningLoop
            adaptive = AdaptiveReasoningLoop()
            # 예시: 질문/응답/리듬 점수 등은 state에서 추출
            question = state.get('input', '')
            answer = state.get('last_response', '')
            rhythm_score = state.get('rhythm_score', 0.5)
            adaptive.register_turn(question, answer, rhythm_score)
            strategy_info = adaptive.decide_strategy()
            result['reasoning_strategy'] = strategy_info
        except Exception as e:
            result['reasoning_strategy'] = {'error': str(e)}
        # AutoResearcher로 knowledge gap 보완
        try:
            if state.get('need_evidence') or state.get('knowledge_gap'):
                from reasoning.auto_researcher import AutoResearcher
                # 실제 상황에 맞게 memory, search_client, persona, llm 등 전달 필요
                memory = state.get('memory_engine')
                search_client = state.get('web_search_client')
                persona = state.get('identity_mgr')
                llm = state.get('llm_client')
                flow = state.get('thought_flow')
                if memory and search_client and persona and llm and flow:
                    researcher = AutoResearcher(memory, search_client, persona, llm)
                    enriched = researcher.enrich(flow)
                    result['auto_research'] = {
                        'enriched_flow': enriched.flow,
                        'evidence_summary': enriched.summary
                    }
        except Exception as e:
            result['auto_research'] = {'error': str(e)}
        # 기존 GWT, scar/meta 등 내부 호출 유지
        try:
            from core.gwt_agents import GWTAgentManager
            gwt_manager = GWTAgentManager(v8_mode=True)
            gwt_result = gwt_manager.create_agent_group_conversation(
                context_data=state.get('context_data', ''),
                task=state.get('task', ''),
                v8_mode=True
            )
            result['gwt_reasoning'] = gwt_result
        except Exception as e:
            result['gwt_reasoning'] = {'error': str(e)}
        try:
            from core.meta_learning import ScarTriggerEngine
            scar_checker = ScarTriggerEngine()
            scars = scar_checker.detect(str(result), meta=state)
            result['scar_analysis'] = scars
        except Exception as e:
            result['scar_analysis'] = {'error': str(e)}
        state.update(result)
        return state
    
    def _decision_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """결정 노드 - LIDA DecisionSystem 사용 및 결과 연동"""
        reasoning_result = state.get("reasoning_result")
        stimulus_importance = state.get("stimulus_importance")
        perception_result = state.get("perception_result")
        ghost_state = state.get("ghost_state", None)
        
        if not reasoning_result:
            return {**state, "cognitive_state": CognitiveStateType.DECISION}
        
        # 프롬프트 생성
        result = {}
        try:
            from prompt.prompt_architect import ContextualPromptArchitect
            architect = ContextualPromptArchitect()
            prompt = architect.build_prompt(
                memory_path=state.get('memory_path', []),
                agent_thoughts=state.get('agent_thoughts', []),
                context=state.get('context', {})
            )
            result['decision_prompt'] = prompt
        except Exception as e:
            result['decision_prompt'] = {'error': str(e)}
        # 메모리 요약
        try:
            from memory.models import PalantirMemoryGraph
            graph = PalantirMemoryGraph()
            summary = graph.summarize_subgraph(state.get('memory_start_id', ''))
            result['memory_summary'] = summary
        except Exception as e:
            result['memory_summary'] = {'error': str(e)}
        state.update(result)
        
        # LIDA DecisionSystem 사용
        decision_result = self.decision_system.make_decision(
            context=reasoning_result,
            importance=stimulus_importance,
            perception=perception_result,
            emotional_state=state.get("emotional_state", {}),
            memory_context=state.get("memory_context", {})
        )
        
        # 기존 Action, Intention, Narrative 생성에 반영
        if ghost_state is not None:
            from memory.models import Action, Intention, Narrative, ActionType
            # Action 생성
            action = Action(
                content=decision_result.selected_option or decision_result.reasoning,
                action_type=decision_result.decision_type.value,
                generated_expectation_ids=[]
            )
            ghost_state.selected_action_knoxel = action
            # Intention/Narrative 등은 필요시 추가
        
        return {
            **state,
            "decision_result": decision_result,
            "cognitive_state": CognitiveStateType.DECISION
        }
    
    def _action_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """행동 노드"""
        action_plan = state.get("action_plan")
        
        action_result = self._execute_action_plan(action_plan)
        
        result = {}
        # 행동 시뮬레이션 예시 (추후 확장)
        result['action_simulation'] = 'simulated_action_result'
        state.update(result)
        
        return {
            **state,
            "action_result": action_result,
            "action_success": self._assess_action_success(action_result),
            "cognitive_state": CognitiveStateType.ACTION
        }
    
    def _learning_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """학습 노드"""
        action_result = state.get("action_result")
        action_success = state.get("action_success")
        
        result = {}
        # meta learning/scar 감지 예시
        try:
            from core.meta_learning import MetaLearningLoop
            meta_loop = MetaLearningLoop(scars=state.get('scar_analysis', []), previous_response=str(state), context=state)
            reflection = meta_loop.reflect()
            result['meta_reflection'] = reflection
        except Exception as e:
            result['meta_reflection'] = {'error': str(e)}
        state.update(result)
        
        learning_outcome = self._perform_learning(action_result, action_success)
        
        return {
            **state,
            "learning_outcome": learning_outcome,
            "cognitive_state": CognitiveStateType.LEARNING
        }
    
    def _integration_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """통합 노드"""
        learning_outcome = state.get("learning_outcome")
        
        integration_result = self._perform_integration(learning_outcome)
        
        return {
            **state,
            "integration_result": integration_result,
            "cognitive_state": CognitiveStateType.INTEGRATION
        }
    
    # 조건부 엣지 함수들
    def _should_reason(self, state: Dict[str, Any]) -> str:
        """추론이 필요한지 판단"""
        stimulus_importance = state.get("stimulus_importance", 0.0)
        return "important_stimulus" if stimulus_importance > 0.5 else "simple_stimulus"
    
    def _should_deliberate(self, state: Dict[str, Any]) -> str:
        """추가 추론이 필요한지 판단"""
        reasoning_complexity = state.get("reasoning_complexity", 0.0)
        return "continue_reasoning" if reasoning_complexity > 0.7 else "complex_reasoning_complete"
    
    def _action_type_router(self, state: Dict[str, Any]) -> str:
        """행동 유형 라우팅"""
        decision = state.get("decision", {})
        action_type = decision.get("action_type", "Reply")
        
        if action_type in ["LearnFromExperience", "ProcessMemory"]:
            return "learning_action"
        elif action_type in ["GenerateInsight", "AdaptBehavior"]:
            return "integration_action"
        else:
            return "execute_action"
    
    def _should_learn(self, state: Dict[str, Any]) -> str:
        """학습이 필요한지 판단"""
        action_success = state.get("action_success", False)
        return "action_completed" if action_success else "action_failed"
    
    # 헬퍼 함수들
    def _gather_attention_candidates(self, stimulus: Stimulus) -> List[Any]:
        """주의 후보 수집"""
        candidates = []
        candidates.append(stimulus)
        
        # 최근 기억
        recent_memories = [k for k in self.knoxels[-10:] if isinstance(k, Feature) or isinstance(k, Action)]
        candidates.extend(recent_memories)
        
        # 현재 의도
        intentions = [k for k in self.knoxels if isinstance(k, Intention) and k.tick_id == self.tick_id]
        candidates.extend(intentions)
        
        # 최근 내러티브
        recent_narratives = list(self.narratives.values())[-3:] if self.narratives else []
        candidates.extend(recent_narratives)
        
        # 메타인지 통찰
        if hasattr(self, 'meta_insights'):
            candidates.extend(self.meta_insights[-2:])
        
        return candidates
    
    def _select_attention_focus(self, candidates: List[Any]) -> Any:
        """주의 집중 대상 선택"""
        if not candidates:
            return None
        
        # 간단한 우선순위 기반 선택
        priorities = []
        for candidate in candidates:
            if isinstance(candidate, Stimulus):
                priority = 1.0
            elif isinstance(candidate, Intention):
                priority = 0.8
            elif isinstance(candidate, Feature):
                priority = 0.6
            else:
                priority = 0.4
            priorities.append(priority)
        
        # 가중치 기반 선택
        total_priority = sum(priorities)
        if total_priority == 0:
            return candidates[0]
        
        weights = [p / total_priority for p in priorities]
        return random.choices(candidates, weights=weights)[0]
    
    def _process_perception(self, stimulus: Stimulus, attention_focus: Any) -> Dict[str, Any]:
        """지각 처리"""
        return {
            "stimulus_type": stimulus.stimulus_type,
            "content": stimulus.content,
            "attention_focus": attention_focus,
            "perception_confidence": 0.8
        }
    
    def _assess_importance(self, stimulus: Stimulus) -> float:
        """중요도 평가"""
        if stimulus.stimulus_type == StimulusType.UserMessage:
            content = stimulus.content.lower()
            if any(word in content for word in ["도움", "어떻게", "문제", "고민"]):
                return 0.9
            elif any(word in content for word in ["감정", "기분", "스트레스", "불안"]):
                return 0.8
            elif len(content) > 50:
                return 0.6
            else:
                return 0.3
        return 0.5
    
    def _perform_reasoning(self, stimulus: Stimulus, perception_result: Dict[str, Any]) -> Dict[str, Any]:
        """추론 수행"""
        return {
            "reasoning_type": "analytical",
            "conclusions": [f"자극 {stimulus.id}에 대한 분석 완료"],
            "confidence": 0.7,
            "complexity": 0.5
        }
    
    def _assess_reasoning_complexity(self, reasoning_result: Dict[str, Any]) -> float:
        """추론 복잡도 평가"""
        return reasoning_result.get("complexity", 0.5)
    
    def _make_decision(self, context: Dict[str, Any], stimulus_importance: float = 0.5) -> Dict[str, Any]:
        """의사결정 - 타입 오류 수정"""
        try:
            # stimulus_importance가 None인 경우 기본값 사용
            if stimulus_importance is None:
                stimulus_importance = 0.5
            
            # 기존 로직 유지
            decision = {
                "action_type": "respond",
                "confidence": 0.8,
                "reasoning": "사용자 입력에 대한 응답 필요"
            }
            
            return decision
        except Exception as e:
            logger.error(f"의사결정 중 오류: {e}")
            return {
                "action_type": "error",
                "confidence": 0.0,
                "reasoning": f"의사결정 오류: {e}"
            }
    
    def _create_action_plan(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """행동 계획 생성"""
        return {
            "action_type": decision.get("action_type"),
            "content": f"결정된 행동: {decision.get('action_type')}",
            "priority": decision.get("priority"),
            "estimated_time": 1.0
        }
    
    def _execute_action_plan(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """행동 계획 실행 - 타입 오류 수정"""
        try:
            # action_plan이 None인 경우 기본값 사용
            if action_plan is None:
                action_plan = {"action_type": "respond"}
            
            # 기존 로직 유지
            result = {
                "success": True,
                "action_type": action_plan.get("action_type", "unknown"),
                "output": "행동 실행 완료"
            }
            
            return result
        except Exception as e:
            logger.error(f"행동 계획 실행 중 오류: {e}")
            return {
                "success": False,
                "action_type": "error",
                "output": f"실행 오류: {e}"
            }
    
    def _assess_action_success(self, action_result: Dict[str, Any]) -> bool:
        """행동 성공 여부 평가"""
        return action_result.get("success", False)
    
    def _perform_learning(self, action_result: Dict[str, Any], action_success: bool = True) -> Dict[str, Any]:
        """학습 수행 - 타입 오류 수정"""
        try:
            # None 값 처리
            if action_result is None:
                action_result = {"action_type": "unknown"}
            if action_success is None:
                action_success = True
            
            # 기존 로직 유지
            learning_outcome = {
                "learned": True,
                "insights": ["행동 결과로부터 학습 완료"],
                "improvements": ["향후 개선 방향"]
            }
            
            return learning_outcome
        except Exception as e:
            logger.error(f"학습 수행 중 오류: {e}")
            return {
                "learned": False,
                "insights": [],
                "improvements": [f"학습 오류: {e}"]
            }
    
    def _perform_integration(self, learning_outcome: Dict[str, Any]) -> Dict[str, Any]:
        """통합 수행 - 타입 오류 수정"""
        try:
            # learning_outcome이 None인 경우 기본값 사용
            if learning_outcome is None:
                learning_outcome = {"learned": False}
            
            # 기존 로직 유지
            integration_result = {
                "integrated": True,
                "memory_updated": True,
                "knowledge_enhanced": learning_outcome.get("learned", False)
            }
            
            return integration_result
        except Exception as e:
            logger.error(f"통합 수행 중 오류: {e}")
            return {
                "integrated": False,
                "memory_updated": False,
                "knowledge_enhanced": False
            }
    
    def gather_attention_candidates(self, state: GhostState, stimulus: Stimulus):
        """
        자극, 기억, 의도, 특성, 내러티브, 메타인사이트 등 모든 인지 요소를 attention_candidates pool에 통합
        """
        candidates = []
        # 1. 자극
        candidates.append(stimulus)
        # 2. 최근 기억(예: 최근 10개)
        recent_memories = [k for k in self.knoxels[-10:] if isinstance(k, Feature) or isinstance(k, Action)]
        candidates.extend(recent_memories)
        # 3. 현재 tick에서 생성된 의도
        intentions = [k for k in self.knoxels if isinstance(k, Intention) and k.tick_id == self.tick_id]
        candidates.extend(intentions)
        # 4. 최근 내러티브(예: 최근 3개)
        recent_narratives = list(self.narratives.values())[-3:] if self.narratives else []
        candidates.extend(recent_narratives)
        # 5. 메타인지 통찰(임시)
        if hasattr(self, 'meta_insights'):
            candidates.extend(self.meta_insights[-2:])
        # 6. 기타 특성(예: 감정, 주관적 경험 등)
        features = [k for k in self.knoxels[-5:] if isinstance(k, Feature)]
        candidates.extend(features)
        # 중복 제거(간단히 id 기준)
        unique_candidates = {id(c): c for c in candidates}.values()
        state.attention_candidates = KnoxelList(list(unique_candidates))
    
    def tick(self, stimulus: Stimulus) -> Action:
        """인지 사이클의 한 틱 실행 - LangGraph 기반 상태 관리 + 모니터링 통합 + 웹 검색 통합"""
        self.tick_id += 1
        
        # 1. 틱 상태 초기화
        state = self._initialize_tick_state(stimulus)
        
        # === [추가] 모니터링 시작 ===
        self._start_tick_monitoring(state)
        # === [기존 기능 유지] ===
        
        # === [추가] 웹 검색 통합 ===
        self._integrate_web_search(stimulus, state)
        # === [기존 기능 유지] ===
        
        # LangGraph 기반 처리 또는 폴백 처리
        if self.config.use_langgraph and LANGGRAPH_AVAILABLE and self.state_graph:
            action = self._tick_with_langgraph(stimulus, state)
        else:
            action = self._tick_with_fallback(stimulus, state)
        
        # === [추가] 모니터링 완료 ===
        self._complete_tick_monitoring(state, action)
        # === [기존 기능 유지] ===
        
        return action
    
    def _integrate_web_search(self, stimulus: Stimulus, state: GhostState):
        """웹 검색 통합"""
        try:
            # 웹 검색 시스템이 있는지 확인
            if hasattr(self, 'web_search_system') and self.web_search_system:
                # 자극 내용에서 검색 키워드 추출
                search_keywords = self._extract_search_keywords(stimulus.content)
                
                if search_keywords:
                    # 비동기 웹 검색 실행
                    asyncio.create_task(self._perform_web_search(search_keywords, stimulus))
                
                # API 호출이 필요한지 확인
                api_calls = self._identify_api_calls(stimulus.content)
                if api_calls:
                    asyncio.create_task(self._perform_api_calls(api_calls, stimulus))
                
        except Exception as e:
            print(f"웹 검색 통합 오류: {e}")
    
    def _extract_search_keywords(self, content: str) -> List[str]:
        """검색 키워드 추출"""
        keywords = []
        
        # 검색 의도를 나타내는 패턴들
        search_patterns = [
            r'검색해줘[:\s]*(.+)',
            r'찾아줘[:\s]*(.+)',
            r'알려줘[:\s]*(.+)',
            r'무엇인가요[:\s]*(.+)',
            r'뭐야[:\s]*(.+)',
            r'정보[:\s]*(.+)',
            r'최신[:\s]*(.+)',
            r'뉴스[:\s]*(.+)',
            r'날씨[:\s]*(.+)',
            r'계산[:\s]*(.+)',
            r'번역[:\s]*(.+)'
        ]
        
        for pattern in search_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            keywords.extend(matches)
        
        # 일반적인 질문에서도 키워드 추출
        question_keywords = self._extract_question_keywords(content)
        keywords.extend(question_keywords)
        
        return list(set(keywords))  # 중복 제거
    
    def _extract_question_keywords(self, content: str) -> List[str]:
        """질문에서 키워드 추출"""
        keywords = []
        
        # 질문 패턴
        question_patterns = [
            r'(\w+)\s*는\s*무엇인가요',
            r'(\w+)\s*에\s*대해\s*알려주세요',
            r'(\w+)\s*에\s*대한\s*정보',
            r'(\w+)\s*어떻게\s*되나요',
            r'(\w+)\s*언제\s*인가요',
            r'(\w+)\s*어디서\s*인가요',
            r'(\w+)\s*누가\s*인가요',
            r'(\w+)\s*왜\s*인가요'
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            keywords.extend(matches)
        
        return keywords
    
    def _identify_api_calls(self, content: str) -> List[Dict[str, Any]]:
        """API 호출 식별"""
        api_calls = []
        
        # 날씨 API 호출
        weather_patterns = [
            r'(\w+)\s*날씨',
            r'(\w+)\s*기온',
            r'(\w+)\s*날씨\s*어떤가요',
            r'(\w+)\s*기온\s*어떤가요'
        ]
        
        for pattern in weather_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                api_calls.append({
                    'type': 'weather',
                    'params': {'city': match},
                    'priority': 'high'
                })
        
        # 뉴스 API 호출
        news_patterns = [
            r'(\w+)\s*뉴스',
            r'(\w+)\s*소식',
            r'(\w+)\s*최신\s*정보',
            r'(\w+)\s*관련\s*뉴스'
        ]
        
        for pattern in news_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                api_calls.append({
                    'type': 'news',
                    'params': {'query': match},
                    'priority': 'medium'
                })
        
        # 계산 API 호출
        calc_patterns = [
            r'(\d+[\+\-\*\/\^\(\)\d\s]+)',
            r'계산[:\s]*(\d+[\+\-\*\/\^\(\)\d\s]+)',
            r'(\d+[\+\-\*\/\^\(\)\d\s]+)\s*계산'
        ]
        
        for pattern in calc_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if any(op in match for op in ['+', '-', '*', '/', '^']):
                    api_calls.append({
                        'type': 'calculation',
                        'params': {'expression': match},
                        'priority': 'high'
                    })
        
        # 번역 API 호출
        translation_patterns = [
            r'번역[:\s]*(.+)',
            r'(\w+)\s*번역',
            r'(\w+)\s*영어로',
            r'(\w+)\s*한국어로'
        ]
        
        for pattern in translation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                api_calls.append({
                    'type': 'translation',
                    'params': {'text': match},
                    'priority': 'medium'
                })
        
        return api_calls
    
    async def _perform_web_search(self, keywords: List[str], stimulus: Stimulus):
        """웹 검색 수행"""
        try:
            from core.web_search_system import SearchQuery, SearchEngine
            
            for keyword in keywords[:3]:  # 최대 3개 키워드만 검색
                search_query = SearchQuery(
                    query=keyword,
                    search_engines=[SearchEngine.GOOGLE, SearchEngine.WIKIPEDIA],
                    max_results=5,
                    language="ko"
                )
                
                results = await self.web_search_system.search(search_query)
                
                # 검색 결과를 메모리에 저장
                self._store_search_results(keyword, results, stimulus)
                
                # 검색 결과를 인지 상태에 반영
                self._integrate_search_results_to_cognition(results, stimulus)
                
        except Exception as e:
            print(f"웹 검색 수행 오류: {e}")
    
    async def _perform_api_calls(self, api_calls: List[Dict[str, Any]], stimulus: Stimulus):
        """API 호출 수행"""
        try:
            for api_call in api_calls:
                api_type = api_call['type']
                params = api_call['params']
                
                if api_type == 'weather':
                    response = await self.web_search_system.get_weather(
                        params.get('city', 'Seoul')
                    )
                elif api_type == 'news':
                    response = await self.web_search_system.get_news(
                        query=params.get('query')
                    )
                elif api_type == 'calculation':
                    response = await self.web_search_system.calculate_expression(
                        params.get('expression')
                    )
                elif api_type == 'translation':
                    response = await self.web_search_system.translate_text(
                        params.get('text')
                    )
                else:
                    continue
                
                # API 응답을 메모리에 저장
                self._store_api_response(api_type, response, stimulus)
                
                # API 응답을 인지 상태에 반영
                self._integrate_api_response_to_cognition(response, stimulus)
                
        except Exception as e:
            print(f"API 호출 수행 오류: {e}")
    
    def _store_search_results(self, keyword: str, results: List, stimulus: Stimulus):
        """검색 결과를 메모리에 저장"""
        try:
            # 검색 결과를 knoxel로 변환하여 저장
            for result in results:
                knoxel_data = {
                    'type': 'web_search_result',
                    'keyword': keyword,
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'source': result.source.value,
                    'relevance_score': result.relevance_score,
                    'timestamp': result.timestamp.isoformat(),
                    'related_stimulus': stimulus.content
                }
                
                # 기존 knoxel 시스템에 저장
                self.knoxels.append(knoxel_data)
                
        except Exception as e:
            print(f"검색 결과 저장 오류: {e}")
    
    def _store_api_response(self, api_type: str, response, stimulus: Stimulus):
        """API 응답을 메모리에 저장"""
        try:
            if response.success:
                api_data = {
                    'type': 'api_response',
                    'api_type': api_type,
                    'data': response.data,
                    'response_time': response.response_time,
                    'timestamp': response.timestamp.isoformat(),
                    'related_stimulus': stimulus.content
                }
                
                # 기존 knoxel 시스템에 저장
                self.knoxels.append(api_data)
                
        except Exception as e:
            print(f"API 응답 저장 오류: {e}")
    
    def _integrate_search_results_to_cognition(self, results: List, stimulus: Stimulus):
        """검색 결과를 인지 상태에 통합"""
        try:
            # 검색 결과의 관련성 점수를 기반으로 인지 상태 업데이트
            avg_relevance = sum(r.relevance_score for r in results) / len(results) if results else 0.0
            
            # 주의 수준 증가
            self.current_cognition.attention += avg_relevance * 0.1
            
            # 명확성 증가 (관련 정보 획득)
            self.current_cognition.clarity += avg_relevance * 0.05
            
            # 검색 결과를 내러티브에 추가
            if results:
                narrative = {
                    'type': 'web_search_integration',
                    'results_count': len(results),
                    'avg_relevance': avg_relevance,
                    'top_result': results[0].title if results else '',
                    'stimulus': stimulus.content,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.narratives.append(narrative)
                
        except Exception as e:
            print(f"검색 결과 인지 통합 오류: {e}")
    
    def _integrate_api_response_to_cognition(self, response, stimulus: Stimulus):
        """API 응답을 인지 상태에 통합"""
        try:
            if response.success:
                # API 응답 성공 시 인지 상태 개선
                self.current_cognition.confidence += 0.05
                self.current_cognition.clarity += 0.03
                
                # API 응답을 내러티브에 추가
                narrative = {
                    'type': 'api_response_integration',
                    'api_type': response.api_type.value,
                    'response_time': response.response_time,
                    'success': response.success,
                    'stimulus': stimulus.content,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.narratives.append(narrative)
            else:
                # API 응답 실패 시 인지 상태 약간 저하
                self.current_cognition.confidence -= 0.02
                
        except Exception as e:
            print(f"API 응답 인지 통합 오류: {e}")
    
    def set_web_search_system(self, web_search_system):
        """웹 검색 시스템 설정"""
        self.web_search_system = web_search_system
        print("웹 검색 시스템이 인지 사이클에 연결되었습니다.")
    
    def get_web_search_summary(self) -> Dict[str, Any]:
        """웹 검색 요약 조회"""
        if not hasattr(self, 'web_search_system') or not self.web_search_system:
            return {"status": "web_search_not_available"}
        
        try:
            # 웹 검색 관련 knoxel 조회
            web_search_knoxels = [
                k for k in self.knoxels 
                if isinstance(k, dict) and k.get('type') == 'web_search_result'
            ]
            
            api_response_knoxels = [
                k for k in self.knoxels 
                if isinstance(k, dict) and k.get('type') == 'api_response'
            ]
            
            # 웹 검색 통계
            search_statistics = self.web_search_system.get_search_statistics()
            
            return {
                "status": "success",
                "web_search_results": len(web_search_knoxels),
                "api_responses": len(api_response_knoxels),
                "search_statistics": search_statistics,
                "recent_searches": [
                    {
                        'keyword': k.get('keyword', ''),
                        'title': k.get('title', ''),
                        'source': k.get('source', ''),
                        'relevance_score': k.get('relevance_score', 0.0)
                    }
                    for k in web_search_knoxels[-5:]  # 최근 5개
                ],
                "recent_api_calls": [
                    {
                        'api_type': k.get('api_type', ''),
                        'response_time': k.get('response_time', 0.0),
                        'success': k.get('data') is not None
                    }
                    for k in api_response_knoxels[-5:]  # 최근 5개
                ]
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def search_external_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """외부 지식 검색"""
        if not hasattr(self, 'web_search_system') or not self.web_search_system:
            return []
        
        try:
            # 동기적 검색 실행 (간단한 쿼리의 경우)
            from core.web_search_system import SearchQuery, SearchEngine
            
            search_query = SearchQuery(
                query=query,
                search_engines=[SearchEngine.GOOGLE, SearchEngine.WIKIPEDIA],
                max_results=max_results,
                language="ko"
            )
            
            # 비동기 검색을 동기적으로 실행
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                self.web_search_system.search(search_query)
            )
            
            # 결과를 딕셔너리 형태로 변환
            return [
                {
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'source': result.source.value,
                    'relevance_score': result.relevance_score
                }
                for result in results
            ]
            
        except Exception as e:
            print(f"외부 지식 검색 오류: {e}")
            return []
    
    async def get_real_time_information(self, info_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """실시간 정보 조회"""
        if not hasattr(self, 'web_search_system') or not self.web_search_system:
            return {"error": "웹 검색 시스템이 연결되지 않았습니다."}
        
        try:
            params = params or {}
            
            if info_type == "weather":
                response = await self.web_search_system.get_weather(
                    params.get('city', 'Seoul'),
                    params.get('country_code', 'KR')
                )
            elif info_type == "news":
                response = await self.web_search_system.get_news(
                    query=params.get('query'),
                    category=params.get('category')
                )
            elif info_type == "translation":
                response = await self.web_search_system.translate_text(
                    params.get('text', ''),
                    params.get('target_lang', 'en'),
                    params.get('source_lang', 'ko')
                )
            elif info_type == "calculation":
                response = await self.web_search_system.calculate_expression(
                    params.get('expression', '')
                )
            else:
                return {"error": f"지원하지 않는 정보 유형: {info_type}"}
            
            if response.success:
                return {
                    "success": True,
                    "data": response.data,
                    "response_time": response.response_time,
                    "api_type": response.api_type.value
                }
            else:
                return {
                    "success": False,
                    "error": response.error_message,
                    "response_time": response.response_time
                }
                
        except Exception as e:
            return {"error": f"실시간 정보 조회 오류: {str(e)}"}

    def _start_tick_monitoring(self, state: GhostState):
        """틱 모니터링 시작"""
        try:
            # 모니터링 시스템이 있는지 확인
            if hasattr(self, 'monitoring_system') and self.monitoring_system:
                # 인지 상태 메트릭 기록
                cognitive_metrics = {
                    "attention_level": self._calculate_attention_level(state),
                    "reasoning_complexity": self._calculate_reasoning_complexity(state),
                    "decision_confidence": self._calculate_decision_confidence(state),
                    "emotional_balance": self._calculate_emotional_balance(state),
                    "memory_efficiency": self._calculate_memory_efficiency(state)
                }
                
                # 모니터링 시스템에 메트릭 전송
                self.monitoring_system.record_cognitive_metrics(cognitive_metrics, self.tick_id)
                
        except Exception as e:
            print(f"틱 모니터링 시작 오류: {e}")
    
    def _complete_tick_monitoring(self, state: GhostState, action: Action):
        """틱 모니터링 완료"""
        try:
            if hasattr(self, 'monitoring_system') and self.monitoring_system:
                # 행동 결과 메트릭 기록
                action_metrics = {
                    "action_type": action.action_type.value,
                    "execution_time": action.execution_time or 0.0,
                    "success": action.status.value == "Completed",
                    "collaboration_count": len(action.collaborating_agents) if hasattr(action, 'collaborating_agents') else 0
                }
                
                # 모니터링 시스템에 행동 메트릭 전송
                self.monitoring_system.record_action_metrics(action_metrics, self.tick_id)
                
                # 인지 그래프 생성
                if self.config.enable_graph_visualization:
                    self._generate_cognitive_graph(state, action)
                
        except Exception as e:
            print(f"틱 모니터링 완료 오류: {e}")
    
    def _calculate_attention_level(self, state: GhostState) -> float:
        """주의 수준 계산"""
        if hasattr(state, 'attention_candidates') and state.attention_candidates:
            return min(1.0, len(state.attention_candidates) / 10.0)
        return 0.5
    
    def _calculate_reasoning_complexity(self, state: GhostState) -> float:
        """추론 복잡도 계산"""
        # 간단한 복잡도 계산 (실제로는 더 정교한 로직 필요)
        complexity_factors = [
            len(self.knoxels) / 100.0,  # 지식 베이스 크기
            len(self.narratives) / 10.0,  # 내러티브 수
            len(self.meta_insights) / 5.0  # 메타인지 통찰 수
        ]
        return min(1.0, sum(complexity_factors) / len(complexity_factors))
    
    def _calculate_decision_confidence(self, state: GhostState) -> float:
        """결정 신뢰도 계산"""
        # 감정 상태와 인지 상태의 균형을 기반으로 계산
        emotional_stability = abs(self.current_emotions.joy - self.current_emotions.sadness)
        cognitive_clarity = self.current_cognition.clarity
        
        return (emotional_stability + cognitive_clarity) / 2.0
    
    def _calculate_emotional_balance(self, state: GhostState) -> float:
        """감정 균형 계산"""
        emotions = [
            self.current_emotions.joy,
            self.current_emotions.sadness,
            self.current_emotions.anger,
            self.current_emotions.fear,
            self.current_emotions.surprise,
            self.current_emotions.disgust
        ]
        
        # 감정의 표준편차를 기반으로 균형 계산 (낮을수록 균형적)
        mean_emotion = sum(emotions) / len(emotions)
        variance = sum((e - mean_emotion) ** 2 for e in emotions) / len(emotions)
        std_dev = variance ** 0.5
        
        # 표준편차를 0-1 범위로 정규화 (낮을수록 균형적)
        return max(0.0, 1.0 - std_dev)
    
    def _calculate_memory_efficiency(self, state: GhostState) -> float:
        """메모리 효율성 계산"""
        if not self.knoxels:
            return 0.5
        
        # 메모리 접근 패턴과 관련성 기반 계산
        recent_accesses = len([k for k in self.knoxels[-20:] if hasattr(k, 'access_count')])
        total_memories = len(self.knoxels)
        
        efficiency = recent_accesses / max(total_memories, 1)
        return min(1.0, efficiency)
    
    def _generate_cognitive_graph(self, state: GhostState, action: Action):
        """인지 그래프 생성"""
        try:
            if hasattr(self, 'monitoring_system') and self.monitoring_system:
                # 시스템 스냅샷 생성
                snapshot = self._create_system_snapshot(state, action)
                
                # 인지 그래프 생성
                graph = self.monitoring_system.create_cognitive_graph(snapshot)
                
                # 그래프 저장 (주기적으로)
                if self.tick_id % 10 == 0:  # 10틱마다 저장
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    graph_path = f"data/monitoring/cognitive_graph_{timestamp}.png"
                    self.monitoring_system.save_graph_visualization(graph, graph_path)
                
        except Exception as e:
            print(f"인지 그래프 생성 오류: {e}")
    
    def _create_system_snapshot(self, state: GhostState, action: Action):
        """시스템 스냅샷 생성"""
        from core.monitoring_system import SystemSnapshot
        
        snapshot = SystemSnapshot(
            cognitive_state={
                "attention_level": self._calculate_attention_level(state),
                "reasoning_complexity": self._calculate_reasoning_complexity(state),
                "decision_confidence": self._calculate_decision_confidence(state),
                "tick_id": self.tick_id
            },
            emotional_state={
                "emotional_balance": self._calculate_emotional_balance(state),
                "joy": self.current_emotions.joy,
                "sadness": self.current_emotions.sadness,
                "anger": self.current_emotions.anger,
                "fear": self.current_emotions.fear
            },
            memory_state={
                "memory_efficiency": self._calculate_memory_efficiency(state),
                "knoxel_count": len(self.knoxels),
                "narrative_count": len(self.narratives),
                "meta_insight_count": len(self.meta_insights)
            },
            agent_states={
                "active_agents": len(action.collaborating_agents) if hasattr(action, 'collaborating_agents') else 0,
                "action_type": action.action_type.value,
                "action_status": action.status.value
            },
            performance_metrics={
                "response_time": action.execution_time or 0.0,
                "success_rate": 1.0 if action.status.value == "Completed" else 0.0,
                "collaboration_efficiency": len(action.collaborating_agents) / 10.0 if hasattr(action, 'collaborating_agents') else 0.0
            }
        )
        
        return snapshot
    
    def set_monitoring_system(self, monitoring_system):
        """모니터링 시스템 설정"""
        self.monitoring_system = monitoring_system
        print("모니터링 시스템이 인지 사이클에 연결되었습니다.")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """모니터링 요약 조회"""
        if not hasattr(self, 'monitoring_system') or not self.monitoring_system:
            return {"status": "monitoring_not_available"}
        
        try:
            return {
                "tick_id": self.tick_id,
                "metrics_summary": self.monitoring_system.get_metrics_summary(),
                "active_alerts": len(self.monitoring_system.get_active_alerts()),
                "system_health": self._calculate_system_health(),
                "cognitive_state": {
                    "attention_level": self._calculate_attention_level(self.states[-1]) if self.states else 0.5,
                    "reasoning_complexity": self._calculate_reasoning_complexity(self.states[-1]) if self.states else 0.5,
                    "decision_confidence": self._calculate_decision_confidence(self.states[-1]) if self.states else 0.5,
                    "emotional_balance": self._calculate_emotional_balance(self.states[-1]) if self.states else 0.5,
                    "memory_efficiency": self._calculate_memory_efficiency(self.states[-1]) if self.states else 0.5
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _calculate_system_health(self) -> float:
        """시스템 건강도 계산"""
        if not self.states:
            return 0.5
        
        latest_state = self.states[-1]
        health_factors = [
            self._calculate_attention_level(latest_state),
            self._calculate_reasoning_complexity(latest_state),
            self._calculate_decision_confidence(latest_state),
            self._calculate_emotional_balance(latest_state),
            self._calculate_memory_efficiency(latest_state)
        ]
        
        return sum(health_factors) / len(health_factors)

    def _tick_with_langgraph(self, stimulus: Stimulus, state: GhostState) -> Action:
        """LangGraph 기반 틱 처리 - LIDA 시스템 통합"""
        # 1. 자극 분류 및 평가
        triage_result = self._triage_stimulus(stimulus)
        state.primary_stimulus = stimulus
        
        # 2. 주의 후보 수집
        self.gather_attention_candidates(state, stimulus)
        
        # === LIDA 시스템 통합 시작 ===
        
        # 3. LIDA PerceptionSystem - 지각 처리
        attention_focus = self._select_attention_focus(state.attention_candidates.to_list())
        perception_result = self.perception_system.process_stimulus(
            stimulus=stimulus,
            attention_focus=attention_focus,
            context={
                "emotional_state": {
                    "valence": state.state_emotions.valence,
                    "arousal": state.state_emotions.anxiety
                },
                "memory": [k.content for k in state.attention_candidates.knoxels[-5:]]
            }
        )
        
        # GhostState에 perception_result 저장
        state.perception_result = perception_result
        
        # 4. LIDA ConsciousnessSystem - 의식 처리
        workspace_state = self.consciousness_system.update_workspace(
            perception_result=perception_result,
            current_workspace=state.conscious_workspace,
            attention_focus=attention_focus
        )
        
        # GhostState에 workspace_state 동기화
        state.conscious_workspace = workspace_state
        
        # 5. 기존 추론 로직 (LIDA와 병행)
        reasoning_result = self._perform_reasoning(stimulus, perception_result)
        
        # 6. LIDA DecisionSystem - 결정 처리
        decision_result = self.decision_system.make_decision(
            context=reasoning_result,
            importance=perception_result.importance_score,
            perception=perception_result,
            emotional_state={
                "valence": state.state_emotions.valence,
                "arousal": state.state_emotions.anxiety
            },
            memory_context={
                "memories": [k.content for k in state.attention_candidates.knoxels],
                "narratives": self.narratives
            }
        )
        
        # === LIDA 시스템 통합 완료 ===
        
        # LangGraph 상태 초기화 (LIDA 결과 포함)
        graph_state = {
            "stimulus": stimulus,
            "ghost_state": state,
            "tick_id": self.tick_id,
            "current_emotions": self.current_emotions,
            "current_needs": self.current_needs,
            "current_cognition": self.current_cognition,
            "knoxels": self.knoxels,
            "narratives": self.narratives,
            # LIDA 결과 추가
            "lida_perception": perception_result,
            "lida_workspace": workspace_state,
            "lida_decision": decision_result,
            "reasoning_result": reasoning_result
        }
        
        # LangGraph 실행
        try:
            result = self.state_graph.invoke(graph_state)
            
            # LIDA 결정 결과를 LangGraph 결과와 통합
            action_type = self._determine_best_action_type(state)
            
            # LIDA 결정 결과를 기존 행동 결정에 반영
            if decision_result.decision_type == DecisionType.REPLY:
                action_type = ActionType.Reply
            elif decision_result.decision_type == DecisionType.INQUIRE:
                action_type = ActionType.ToolCallAndReply
            elif decision_result.decision_type == DecisionType.LEARN:
                action_type = ActionType.Think
            
            # 결과에서 행동 추출
            action_result = result.get("action_result", {})
            if action_result and action_result.get("executed"):
                # LIDA 결과를 반영한 Action 객체 생성
                action = Action(
                    action_id=str(uuid.uuid4()),
                    action_type=action_type,
                    content=action_result.get("result", "LangGraph + LIDA 기반 응답"),
                    tick_id=self.tick_id
                )
            else:
                # LIDA 기반 기본 응답 생성
                action_details = self._deliberate_and_select_action(state, action_type)
                action_details["lida_decision"] = decision_result
                action_details["lida_confidence"] = decision_result.confidence
                
                action = self._execute_action(action_type, action_details, state)
            
            # 상태 업데이트
            self._update_states_from_langgraph_result(result, state)
            
        except Exception as e:
            print(f"LangGraph 실행 오류: {e}")
            # 폴백으로 처리 (이미 LIDA가 통합된 폴백 사용)
            action = self._tick_with_fallback(stimulus, state)
        
        # 기대 생성 및 학습
        expectations = self._generate_expectations_for_action(action, {"lida_decision": decision_result})
        self._perform_learning_and_consolidation(state)
        
        # 상태 저장
        self.states.append(state)
        return action
    
    def _tick_with_fallback(self, stimulus: Stimulus, state: GhostState) -> Action:
        """폴백 처리 - LIDA 시스템 통합"""
        try:
            # LIDA 시스템 지연 초기화
            if self.perception_system is None:
                from core.perception_system import PerceptionSystem
                self.perception_system = PerceptionSystem()
            
            if self.consciousness_system is None:
                from core.consciousness_system import ConsciousnessSystem
                self.consciousness_system = ConsciousnessSystem()
            
            if self.decision_system is None:
                from core.decision_system import DecisionSystem
                self.decision_system = DecisionSystem()
            
            # 1. 자극 분류 및 평가
            triage_result = self._triage_stimulus(stimulus)
            state.primary_stimulus = stimulus
            
            # 2. 주의 후보 수집
            self.gather_attention_candidates(state, stimulus)
            
            # === LIDA 시스템 통합 시작 ===
            
            # 3. LIDA PerceptionSystem - 지각 처리
            attention_focus = self._select_attention_focus(state.attention_candidates.to_list())
            perception_result = self.perception_system.process_stimulus(
                stimulus=stimulus,
                attention_focus=attention_focus,
                context={
                    "emotional_state": {
                        "valence": state.state_emotions.valence,
                        "arousal": state.state_emotions.anxiety
                    },
                    "memory": [k.content for k in state.attention_candidates.knoxels[-5:]]
                }
            )
            
            # GhostState에 perception_result 저장
            state.perception_result = perception_result
            
            # 4. LIDA ConsciousnessSystem - 의식 처리
            workspace_state = self.consciousness_system.update_workspace(
                perception_result=perception_result,
                current_workspace=state.conscious_workspace,
                attention_focus=attention_focus
            )
            
            # GhostState에 workspace_state 동기화
            state.conscious_workspace = workspace_state
            
            # 5. 기존 추론 로직 (LIDA와 병행)
            reasoning_result = self._perform_reasoning(stimulus, perception_result)
            
            # 6. LIDA DecisionSystem - 결정 처리
            decision_result = self.decision_system.make_decision(
                context=reasoning_result,
                importance=perception_result.importance_score,
                perception=perception_result,
                emotional_state={
                    "valence": state.state_emotions.valence,
                    "arousal": state.state_emotions.anxiety
                },
                memory_context={
                    "memories": [k.content for k in state.attention_candidates.knoxels],
                    "narratives": self.narratives
                }
            )
            
            # === LIDA 시스템 통합 완료 ===
            
            # 7. 기존 행동 결정 로직 (LIDA 결과 반영)
            action_type = self._determine_best_action_type(state)
            
            # LIDA 결정 결과를 기존 행동 결정에 반영
            if decision_result.decision_type == DecisionType.REPLY:
                action_type = ActionType.Reply
            elif decision_result.decision_type == DecisionType.INQUIRE:
                action_type = ActionType.ToolCallAndReply
            elif decision_result.decision_type == DecisionType.LEARN:
                action_type = ActionType.Think
            
            # 8. 행동 심의 및 선택
            action_details = self._deliberate_and_select_action(state, action_type)
            
            # LIDA 결정 결과를 action_details에 반영
            action_details["lida_decision"] = decision_result
            action_details["lida_confidence"] = decision_result.confidence
            
            # 9. 행동 실행
            action = self._execute_action(action_type, action_details, state)
            
            # 10. 기대 생성 및 학습
            expectations = self._generate_expectations_for_action(action, action_details)
            self._perform_learning_and_consolidation(state)
            
            return action
            
        except Exception as e:
            logger.error(f"폴백 처리 중 오류: {e}")
            # 오류 시 기본 응답 생성
            return Action(
                content=f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}",
                action_type=ActionType.Reply,
                generated_expectation_ids=[]
            )
    
    def _initialize_tick_state(self, stimulus: Stimulus) -> GhostState:
        """틱 상태 초기화"""
        state = GhostState(
            tick_id=self.tick_id,
            previous_tick_id=self.tick_id - 1 if self.tick_id > 0 else -1,
            timestamp=datetime.now(),
            primary_stimulus=stimulus,
            state_emotions=self.current_emotions,
            state_needs=self.current_needs,
            state_cognition=self.current_cognition
        )
        
        # 자극을 Knoxel로 추가
        stimulus.tick_id = self.tick_id
        stimulus.id = len(self.knoxels)
        self.knoxels.append(stimulus)
        
        return state
    
    def _triage_stimulus(self, stimulus: Stimulus) -> Dict[str, Any]:
        """자극 분류"""
        # 간단한 분류 로직 (실제로는 LLM 사용)
        if stimulus.stimulus_type == StimulusType.UserMessage:
            content = stimulus.content.lower()
            
            # 중요도 판단
            if any(word in content for word in ["도움", "어떻게", "문제", "고민"]):
                return {"category": StimulusTriage.Significant, "reasoning": "사용자가 도움을 요청하는 중요한 메시지"}
            elif any(word in content for word in ["감정", "기분", "스트레스", "불안"]):
                return {"category": StimulusTriage.Significant, "reasoning": "감정적 주제로 중요한 상호작용"}
            elif len(content) > 50:
                return {"category": StimulusTriage.Moderate, "reasoning": "중간 길이의 메시지"}
            else:
                return {"category": StimulusTriage.Insignificant, "reasoning": "짧은 일상적 메시지"}
        else:
            return {"category": StimulusTriage.Moderate, "reasoning": "시스템 자극"}
    
    def _appraise_stimulus(self, state: GhostState):
        """자극 평가 및 상태 변화"""
        stimulus = state.primary_stimulus
        
        # 간단한 상태 변화 계산 (실제로는 LLM 사용)
        emotion_delta = self._calculate_emotion_delta(stimulus)
        needs_delta = self._calculate_needs_delta(stimulus)
        cognition_delta = self._calculate_cognition_delta(stimulus)
        
        # 상태 업데이트
        self.current_emotions = self.current_emotions + emotion_delta
        self.current_needs = self.current_needs + needs_delta
        self.current_cognition = self.current_cognition + cognition_delta
        
        # 상태 스냅샷 업데이트
        state.state_emotions = self.current_emotions
        state.state_needs = self.current_needs
        state.state_cognition = self.current_cognition
        
        # 감정 특성 생성
        feeling = Feature(
            id=len(self.knoxels),
            tick_id=self.tick_id,
            content=f"감정 변화: {emotion_delta.get_overall_valence():.2f}",
            feature_type=FeatureType.Feeling,
            source="stimulus_appraisal",
            affective_valence=emotion_delta.get_overall_valence(),
            causal=True
        )
        self.knoxels.append(feeling)
        state.attention_candidates.add(feeling)
    
    def _calculate_emotion_delta(self, stimulus: Stimulus) -> EmotionalAxesModel:
        """감정 변화 계산"""
        content = stimulus.content.lower()
        delta = EmotionalAxesModel()
        
        # 간단한 키워드 기반 감정 변화
        if any(word in content for word in ["좋아", "행복", "기쁘", "감사"]):
            delta.valence = 0.3
            delta.affection = 0.2
        elif any(word in content for word in ["나쁘", "슬프", "화나", "짜증"]):
            delta.valence = -0.3
            delta.anxiety = 0.2
        elif any(word in content for word in ["도움", "감사", "고마워"]):
            delta.self_worth = 0.2
            delta.relevance = 0.3
        
        return delta
    
    def _calculate_needs_delta(self, stimulus: Stimulus) -> NeedsAxesModel:
        """욕구 변화 계산"""
        content = stimulus.content.lower()
        delta = NeedsAxesModel()
        
        # 상호작용으로 인한 욕구 충족
        if stimulus.stimulus_type == StimulusType.UserMessage:
            delta.connection = 0.2
            delta.relevance = 0.1
            delta.learning_growth = 0.1
        
        return delta
    
    def _calculate_cognition_delta(self, stimulus: Stimulus) -> CognitionAxesModel:
        """인지 변화 계산"""
        content = stimulus.content.lower()
        delta = CognitionAxesModel()
        
        # 복잡한 요청에 대한 인지 변화
        if len(content) > 100 or any(word in content for word in ["복잡", "어려운", "상세"]):
            delta.mental_aperture = 0.2
            delta.willpower = 0.1
        
        return delta
    
    def _generate_short_term_intentions(self, state: GhostState):
        """단기 의도 생성"""
        intentions = []
        
        # 현재 상태에 기반한 의도 생성
        if self.current_needs.connection < 0.3:
            intentions.append(Intention(
                id=len(self.knoxels),
                tick_id=self.tick_id,
                content="사용자와의 연결 강화",
                urgency=0.8,
                affective_valence=0.5,
                internal=True
            ))
        
        if self.current_needs.relevance < 0.3:
            intentions.append(Intention(
                id=len(self.knoxels) + 1,
                tick_id=self.tick_id,
                content="유용한 정보나 도움 제공",
                urgency=0.7,
                affective_valence=0.4,
                internal=True
            ))
        
        for intention in intentions:
            self.knoxels.append(intention)
            state.attention_candidates.add(intention)
    
    def _gather_memories_for_attention(self, state: GhostState):
        """주의를 위한 기억 수집"""
        # 간단한 기억 검색 (실제로는 임베딩 기반 유사도 검색)
        relevant_memories = []
        
        # 최근 대화 기억
        recent_dialogue = [k for k in self.knoxels[-10:] if hasattr(k, 'feature_type') and k.feature_type == FeatureType.Dialogue]
        relevant_memories.extend(recent_dialogue[:3])
        
        # 감정적으로 관련된 기억
        current_valence = self.current_emotions.get_overall_valence()
        emotional_memories = [k for k in self.knoxels if hasattr(k, 'affective_valence') and abs(k.affective_valence - current_valence) < 0.3]
        relevant_memories.extend(emotional_memories[:2])
        
        for memory in relevant_memories:
            state.attention_candidates.add(memory)
    
    def _build_structures_get_coalitions(self, state: GhostState):
        """구조 구축 및 연합 형성 (의미론적 클러스터링 기반)"""
        candidates = state.attention_candidates.to_list()
        # 임베딩이 있는 후보만 추출
        emb_candidates = [(i, c) for i, c in enumerate(candidates) if hasattr(c, 'embedding') and c.embedding and len(c.embedding) > 0]
        if len(emb_candidates) > 2:
            idxs, objs = zip(*emb_candidates)
            X = np.array([c.embedding for c in objs])
            # 클러스터 개수는 후보 수/2로 제한(최소 2, 최대 5)
            n_clusters = min(max(2, len(objs)//2), 5)
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(X)
            for cluster_id in range(n_clusters):
                cluster_objs = [objs[i] for i, label in enumerate(labels) if label == cluster_id]
                state.coalitions_balanced[cluster_id] = cluster_objs
        else:
            # 기존 감정/인지 분류 fallback
            if len(candidates) <= 3:
                state.coalitions_balanced[0] = candidates
            else:
                emotional_coalition = [c for c in candidates if hasattr(c, 'affective_valence') and c.affective_valence is not None]
                cognitive_coalition = [c for c in candidates if c not in emotional_coalition]
                state.coalitions_balanced[0] = emotional_coalition[:3]
                state.coalitions_balanced[1] = cognitive_coalition[:3]
    
    def _simulate_attention_on_coalitions(self, state: GhostState):
        """연합에 대한 주의 시뮬레이션 (LLM/절차적 평가 결합)"""
        best_coalition = None
        best_score = -1
        coalition_scores = {}
        # 1차: 기존 절차적 평가
        for coalition_id, coalition in state.coalitions_balanced.items():
            score = self._rate_coalition(coalition, state)
            coalition_scores[coalition_id] = score
        # 2차: LLM 평가(옵션)
        if self.llm_client is not None and len(state.coalitions_balanced) > 1:
            summaries = {cid: KnoxelList(coal).get_story() for cid, coal in state.coalitions_balanced.items()}
            prompt = "다음은 AI의 주의집중 후보군(의미 집합) 요약이다. 각 집합이 현재 상황에서 얼마나 중요한지 0~1 사이 점수로 평가하라.\n"
            for cid, summary in summaries.items():
                prompt += f"[집합 {cid}] {summary}\n"
            prompt += "형식: 집합ID:점수 (예: 0:0.8, 1:0.3 ...)\n"
            llm_result = self.llm_client(prompt)
            # 간단 파싱
            llm_scores = {}
            for part in llm_result.split(','):
                if ':' in part:
                    k, v = part.split(':')
                    try:
                        llm_scores[int(k.strip())] = float(v.strip())
                    except:
                        continue
            # 가중 평균(절차적 0.5, LLM 0.5)
            for cid in coalition_scores:
                if cid in llm_scores:
                    coalition_scores[cid] = 0.5 * coalition_scores[cid] + 0.5 * llm_scores[cid]
        # 최종 선택
        for coalition_id, score in coalition_scores.items():
            if score > best_score:
                best_score = score
                best_coalition = state.coalitions_balanced[coalition_id]
        if best_coalition:
            state.attention_focus = KnoxelList(best_coalition)
            state.conscious_workspace = KnoxelList(best_coalition)
            # === [추가] conscious_workspace의 감정/욕구/인지 상태를 실시간 반영 ===
            emotions = [getattr(k, 'affective_valence', None) for k in best_coalition if hasattr(k, 'affective_valence') and k.affective_valence is not None]
            needs = [getattr(k, 'relevance', None) for k in best_coalition if hasattr(k, 'relevance') and k.relevance is not None]
            cognition = [getattr(k, 'mental_aperture', None) for k in best_coalition if hasattr(k, 'mental_aperture') and k.mental_aperture is not None]
            # 평균값 계산 및 상태에 누적 반영
            if emotions:
                avg_emotion = sum(emotions) / len(emotions)
                self.current_emotions.valence += avg_emotion * 0.2  # 영향력 가중치
            if needs:
                avg_need = sum(needs) / len(needs)
                self.current_needs.relevance += avg_need * 0.2
            if cognition:
                avg_cog = sum(cognition) / len(cognition)
                self.current_cognition.mental_aperture += avg_cog * 0.2
            # 상태 스냅샷도 즉시 반영
            state.state_emotions = self.current_emotions
            state.state_needs = self.current_needs
            state.state_cognition = self.current_cognition
    
    def _rate_coalition(self, coalition: List[Any], state: GhostState) -> float:
        """연합 평가"""
        if not coalition:
            return 0.0
        
        # 감정적 가치
        emotional_value = sum(getattr(k, 'affective_valence', 0) or 0 for k in coalition) / len(coalition)
        
        # 긴급성
        urgency_value = sum(getattr(k, 'urgency', 0) for k in coalition) / len(coalition)
        
        # 최신성
        recency_value = sum(1 for k in coalition if k.tick_id == self.tick_id) / len(coalition)
        
        # 종합 점수
        total_score = (emotional_value + urgency_value + recency_value) / 3
        return max(0.0, min(1.0, total_score))
    
    def _generate_subjective_experience(self, state: GhostState):
        """주관적 경험 생성"""
        if not state.conscious_workspace.knoxels:
            return
        
        # 의식적 작업공간의 내용을 종합
        workspace_content = state.conscious_workspace.get_story()
        
        # 현재 감정 상태
        emotion_summary = self._verbalize_emotional_state()
        
        # 주관적 경험 생성
        subjective_content = f"현재 감정: {emotion_summary}. 의식적 내용: {workspace_content[:200]}..."
        
        subjective_experience = Feature(
            id=len(self.knoxels),
            tick_id=self.tick_id,
            content=subjective_content,
            feature_type=FeatureType.SubjectiveExperience,
            source="conscious_workspace",
            causal=True
        )
        
        self.knoxels.append(subjective_experience)
        state.subjective_experience = subjective_experience
    
    def _verbalize_emotional_state(self) -> str:
        """감정 상태를 언어로 표현"""
        valence = self.current_emotions.get_overall_valence()
        
        if valence > 0.5:
            return "매우 긍정적이고 기쁜 상태"
        elif valence > 0:
            return "약간 긍정적인 상태"
        elif valence > -0.5:
            return "중립적이고 차분한 상태"
        else:
            return "부정적이고 우울한 상태"
    
    def _determine_best_action_type(self, state: GhostState) -> ActionType:
        """최적 행동 유형 결정"""
        stimulus = state.primary_stimulus
        
        if stimulus.stimulus_type == StimulusType.UserMessage:
            # 사용자 메시지에 대한 응답
            return ActionType.Reply
        elif stimulus.stimulus_type == StimulusType.UserInactivity:
            # 사용자 비활성에 대한 대응
            return ActionType.InitiateUserConversation
        else:
            # 기본적으로 응답
            return ActionType.Reply
    
    def _deliberate_and_select_action(self, state: GhostState, action_type: ActionType) -> Dict[str, Any]:
        """행동 심의 및 선택"""
        # 간단한 행동 생성 (실제로는 LLM 사용)
        if action_type == ActionType.Reply:
            return self._deliberate_reply(state)
        elif action_type == ActionType.InitiateUserConversation:
            return self._deliberate_initiate_conversation(state)
        else:
            return {"content": "기본 응답", "confidence": 0.5}
    
    def _deliberate_reply(self, state: GhostState) -> Dict[str, Any]:
        """응답 심의"""
        stimulus = state.primary_stimulus
        emotion_summary = self._verbalize_emotional_state()
        
        # 감정 상태에 따른 응답 스타일 결정
        valence = self.current_emotions.get_overall_valence()
        
        if valence > 0.3:
            response_style = "따뜻하고 긍정적인"
        elif valence < -0.3:
            response_style = "공감적이고 지원적인"
        else:
            response_style = "중립적이고 도움이 되는"
        
        content = f"[{response_style} 응답] {stimulus.content}에 대한 {emotion_summary} 상태에서의 응답"
        
        return {
            "content": content,
            "confidence": 0.8,
            "response_style": response_style,
            "emotional_context": emotion_summary
        }
    
    def _deliberate_initiate_conversation(self, state: GhostState) -> Dict[str, Any]:
        """대화 시작 심의"""
        # 사용자 관심사나 현재 상황에 기반한 대화 시작
        content = "사용자와의 연결을 유지하기 위한 관심 있는 질문이나 주제 제안"
        
        return {
            "content": content,
            "confidence": 0.6,
            "conversation_type": "engagement"
        }
    
    def _execute_action(self, action_type: ActionType, action_details: Dict[str, Any], state: GhostState) -> Action:
        """행동 실행"""
        action = Action(
            id=len(self.knoxels),
            tick_id=self.tick_id,
            content=action_details.get("content", "기본 행동"),
            action_type=action_type
        )
        
        self.knoxels.append(action)
        state.selected_action_knoxel = action
        
        # 기대 생성
        expectations = self._generate_expectations_for_action(action, action_details)
        for expectation in expectations:
            self.knoxels.append(expectation)
            action.generated_expectation_ids.append(expectation.id)
        
        return action
    
    def _generate_expectations_for_action(self, action: Action, action_details: Dict[str, Any]) -> List[Intention]:
        """행동에 대한 기대 생성"""
        expectations = []
        
        if action.action_type == ActionType.Reply:
            # 사용자가 응답에 만족할 것이라는 기대
            expectation = Intention(
                id=len(self.knoxels),
                tick_id=self.tick_id,
                content="사용자가 응답에 만족하고 추가 상호작용을 원할 것",
                urgency=0.5,
                affective_valence=0.3,
                internal=False,
                originating_action_id=action.id
            )
            expectations.append(expectation)
        
        return expectations
    
    def _perform_learning_and_consolidation(self, state: GhostState):
        """학습 및 통합 수행"""
        # 간단한 학습 (실제로는 더 복잡한 메모리 통합)
        
        # 에피소드 요약 생성
        episode_summary = f"틱 {self.tick_id}: {state.primary_stimulus.content[:50]}... -> {state.selected_action_knoxel.content[:50]}..."
        
        # 에피소드 특성 생성
        episode_feature = Feature(
            id=len(self.knoxels),
            tick_id=self.tick_id,
            content=episode_summary,
            feature_type=FeatureType.WorldEvent,
            source="episode_consolidation",
            causal=True
        )
        
        self.knoxels.append(episode_feature)
        
        # 상태 감쇠
        self.current_emotions.decay_to_baseline(0.1)
        self.current_needs.decay_to_zero(0.05)
        self.current_cognition.decay_to_baseline(0.1)
    
    def get_current_state_summary(self) -> str:
        """현재 상태 요약"""
        summary = f"=== 인지 사이클 상태 (틱 {self.tick_id}) ===\n"
        summary += f"감정: {self._verbalize_emotional_state()}\n"
        summary += f"연결 욕구: {self.current_needs.connection:.2f}\n"
        summary += f"관련성 욕구: {self.current_needs.relevance:.2f}\n"
        summary += f"자아 강도: {self.current_cognition.ego_strength:.2f}\n"
        summary += f"총 Knoxel 수: {len(self.knoxels)}\n"
        
        return summary
    
    def get_recent_interactions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """최근 상호작용 조회"""
        interactions = []
        
        for i in range(max(0, len(self.states) - limit), len(self.states)):
            state = self.states[i]
            interaction = {
                "tick_id": state.tick_id,
                "stimulus": state.primary_stimulus.content if state.primary_stimulus else "",
                "action": state.selected_action_knoxel.content if state.selected_action_knoxel else "",
                "emotion": state.state_emotions.get_overall_valence(),
                "timestamp": state.timestamp.isoformat()
            }
            interactions.append(interaction)
        
        return interactions

    def _evaluate_expectations_and_update_narratives(self, state: GhostState):
        """기대치 위반/충족 평가 및 내러티브 진화, 메타인지 인사이트 자동화 (placeholder)"""
        # 1. 기대치 위반/충족 평가 (예시: 최근 행동과 기대치 비교)
        recent_expectations = [k for k in self.knoxels[-10:] if isinstance(k, Intention) and not k.internal]
        for exp in recent_expectations:
            # 간단히: 최근 행동 내용이 기대치 내용에 포함되면 충족
            if state.selected_action_knoxel and exp.content in state.selected_action_knoxel.content:
                exp.fulfilment = 1.0
            else:
                exp.fulfilment = max(0.0, exp.fulfilment - 0.1)
        # 2. 내러티브 진화 (예시: 최근 감정/행동 요약을 내러티브에 추가)
        if self.narratives is not None:
            summary = f"틱 {self.tick_id} 감정:{self.current_emotions.get_overall_valence():.2f} 행동:{state.selected_action_knoxel.content if state.selected_action_knoxel else ''}"
            self.narratives[f"tick_{self.tick_id}"] = Narrative(
                id=len(self.narratives),
                tick_id=self.tick_id,
                narrative_type=None,
                target_name=self.config.companion_name,
                content=summary
            )
        # 3. 메타인지 인사이트 자동화 (예시: 최근 tick의 주요 변화 기록)
        if hasattr(self, 'meta_insights'):
            insight = Feature(
                id=len(self.knoxels),
                tick_id=self.tick_id,
                content=f"틱 {self.tick_id} 주요 변화 기록",
                feature_type=FeatureType.MetaInsight,
                source="meta_automation",
                causal=False
            )
            self.meta_insights.append(insight)
            self.knoxels.append(insight)

    def _update_states_from_langgraph_result(self, result: Dict[str, Any], state: GhostState):
        """LangGraph 결과에서 상태 업데이트"""
        # 감정 상태 업데이트
        if "current_emotions" in result:
            self.current_emotions = result["current_emotions"]
            state.state_emotions = self.current_emotions
        
        # 욕구 상태 업데이트
        if "current_needs" in result:
            self.current_needs = result["current_needs"]
            state.state_needs = self.current_needs
        
        # 인지 상태 업데이트
        if "current_cognition" in result:
            self.current_cognition = result["current_cognition"]
            state.state_cognition = self.current_cognition
        
        # Knoxel 업데이트
        if "knoxels" in result:
            self.knoxels = result["knoxels"]
        
        # 내러티브 업데이트
        if "narratives" in result:
            self.narratives = result["narratives"]
        
        # 학습 결과 처리
        if "learning_outcome" in result:
            learning_outcome = result["learning_outcome"]
            if learning_outcome.get("learned"):
                # 메타인지 통찰 추가
                for insight in learning_outcome.get("insights", []):
                    feature = Feature(
                        id=len(self.knoxels),
                        tick_id=self.tick_id,
                        feature_type=FeatureType.MetaInsight,
                        content=insight,
                        confidence=0.8
                    )
                    self.knoxels.append(feature)
                    self.meta_insights.append(feature)
        
        # 통합 결과 처리
        if "integration_result" in result:
            integration_result = result["integration_result"]
            if integration_result.get("integrated"):
                # 행동 변화 적용
                for behavior_change in integration_result.get("behavior_changes", []):
                    # 향후 행동 적응 로직에 반영
                    pass 

    def _create_cognition_delta(self, quality_score: float):
        """인지 상태 델타 생성"""
        from core.state_models import CognitionAxesModel
        
        if quality_score > 0.7:
            # 높은 품질 - 자신감 증가
            return CognitionAxesModel(
                ego_strength=0.05,
                willpower=0.05
            )
        else:
            # 개선 필요 - 더 집중
            return CognitionAxesModel(
                mental_aperture=-0.1,
                willpower=0.1
            )

    def tick_with_lida(self, stimulus: Stimulus) -> Action:
        """LIDA 시스템을 사용한 완전한 인지 사이클 실행"""
        self.tick_id += 1
        
        # 초기 상태 생성
        state = self._initialize_tick_state(stimulus)
        
        # 1. 주의 단계 (Attention)
        attention_candidates = self.gather_attention_candidates(state, stimulus)
        attention_focus = self._select_attention_focus(attention_candidates)
        state.attention_focus = attention_focus
        
        # 2. 지각 단계 (Perception) - LIDA PerceptionSystem
        perception_result = self.perception_system.process_stimulus(
            stimulus=stimulus,
            attention_focus=attention_focus,
            context={"emotional_state": state.emotional_state, "memory": state.memories}
        )
        state.perception_result = perception_result
        
        # 3. 의식 단계 (Consciousness) - LIDA ConsciousnessSystem
        workspace_state = self.consciousness_system.update_workspace(
            perception_result=perception_result,
            current_workspace=state.get("workspace_state", WorkspaceState()),
            attention_focus=attention_focus
        )
        state.workspace_state = workspace_state
        
        # 4. 추론 단계 (Reasoning)
        reasoning_result = self._perform_reasoning(stimulus, perception_result)
        state.reasoning_result = reasoning_result
        
        # 5. 결정 단계 (Decision) - LIDA DecisionSystem
        decision_result = self.decision_system.make_decision(
            context=reasoning_result,
            importance=perception_result.importance_score,
            perception=perception_result,
            emotional_state=state.emotional_state,
            memory_context={"memories": state.memories, "narratives": self.narratives}
        )
        state.decision_result = decision_result
        
        # 6. 행동 계획 생성
        action_plan = self._create_action_plan_from_decision(decision_result)
        state.action_plan = action_plan
        
        # 7. 행동 실행
        action = self._execute_action_plan(action_plan)
        
        # 8. 학습 및 통합
        self._perform_learning_and_consolidation(state)
        
        # 모니터링 업데이트
        if self.monitoring_system:
            self._complete_tick_monitoring(state, action)
        
        return action
    
    def _create_action_plan_from_decision(self, decision_result: DecisionResult) -> Dict[str, Any]:
        """DecisionResult에서 행동 계획 생성"""
        action_plan = {
            "action_type": decision_result.decision_type.value,
            "confidence": decision_result.confidence,
            "reasoning": decision_result.reasoning,
            "options_considered": decision_result.options_considered,
            "selected_option": decision_result.selected_option,
            "execution_steps": []
        }
        
        # 결정 유형에 따른 실행 단계 생성
        if decision_result.decision_type == DecisionType.REPLY:
            action_plan["execution_steps"] = [
                {"step": "generate_response", "priority": 1},
                {"step": "format_response", "priority": 2},
                {"step": "validate_response", "priority": 3}
            ]
        elif decision_result.decision_type == DecisionType.INQUIRE:
            action_plan["execution_steps"] = [
                {"step": "formulate_question", "priority": 1},
                {"step": "select_question_type", "priority": 2}
            ]
        elif decision_result.decision_type == DecisionType.LEARN:
            action_plan["execution_steps"] = [
                {"step": "identify_learning_target", "priority": 1},
                {"step": "create_learning_plan", "priority": 2},
                {"step": "execute_learning", "priority": 3}
            ]
        
        return action_plan
