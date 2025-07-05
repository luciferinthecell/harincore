from __future__ import annotations

from memory.models import MemoryEpisodeNode, InternalState, Stimulus, StimulusType, EmotionalAxesModel, NeedsAxesModel, CognitionAxesModel
from memory.palantirgraph import PalantirGraph
from memory.text_importer import TextImporter, create_text_importer
from memory.memory_retriever import MemoryRetriever
from memory.memory_conductor import MemoryConductor
from prompt.prompt_architect import PromptArchitect
from tools.llm_client import LLMClient
from core.context import UserContext
from core.enhanced_loops import EnhancedLoopManager, EnhancedJudgment
from core.existential_layer import ExistentialLayer, execute_complete_thought_pipeline, generate_memory_node
from core.memory_orchestrator import MemoryOrchestrator
from core.multi_intent_parser_fixed import MultiIntentParser, ParsedIntent
from core.parallel_reasoning_unit import ParallelReasoningUnit, ParallelReasoningResult
from core.thought_unit_decomposer import ThoughtUnitDecomposer, DecompositionResult, ThoughtUnit
from core.cognitive_cycle import CognitiveCycle, CognitiveCycleConfig
from core.environment_manager import EnvironmentManager, get_environment_manager
from core.stimulus_classifier import get_stimulus_classifier
from core.stimulus_processor import get_stimulus_processor
from core.triz_loop import AdvancedTRIZLoop
from core.action_system import ActionSystem
from core.metacognition_system import AdvancedMetacognitionSystem
from core.advanced_reasoning_system import AdvancedReasoningSystem

import numpy as np
from scipy.spatial.distance import cosine
import json
import uuid
from datetime import datetime, timezone, time, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
import random


"""
harin.core.enhanced_main_loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

기억을 활용하는 향상된 메인 루프 - 고급 인지 시스템 통합
- 텍스트 파일 업로드 및 기억 저장
- 관련 기억 자동 검색 및 활용
- 사고루프와 기억 연동
- 실시간 API 데이터 처리
- 존재 역할 기반 사고 파이프라인 통합
- 고급 인지 사이클 및 심리학적 상태 모델 통합
- 몬테카를로 시뮬레이션 기반 의사결정 시스템
"""

# === 새로운 고급 인지 시스템 모델들 ===

class StimulusTriage(str, Enum):
    """자극 중요도 분류 시스템"""
    Insignificant = "Insignificant"  # 무시, 인과적 특징만 추가
    Moderate = "Moderate"           # 빠른 행동 생성, 제한된 의식 작업공간
    Significant = "Significant"     # 전체 파이프라인

class ActionType(str, Enum):
    """행동 유형 시스템"""
    Ignore = "Ignore"
    Reply = "Reply"
    ToolCallAndReply = "ToolCallAndReply"
    Sleep = "Sleep"
    InitiateUserConversation = "InitiateUserConversation"
    InitiateInternalContemplation = "InitiateInternalContemplation"
    ToolCall = "ToolCall"

class ActionSimulation(BaseModel):
    """몬테카를로 시뮬레이션 결과"""
    sim_id: str
    action_type: ActionType
    ai_reply_content: str
    simulated_user_reaction: str
    predicted_ai_emotion: EmotionalAxesModel
    intent_fulfillment_score: float
    needs_fulfillment_score: float
    emotion_score: float
    cognitive_load: float
    cognitive_congruence_score: float
    final_score: float

class ActionRating(BaseModel):
    """행동 평가 결과"""
    overall_reasoning: str = Field(description="전체적인 평가 근거")
    intent_fulfillment: float = Field(..., ge=0.0, le=1.0, description="내부 목표 달성도")
    needs_fulfillment: float = Field(..., ge=0.0, le=1.0, description="욕구 충족도")
    predicted_emotional_impact: float = Field(..., ge=-1.0, le=1.0, description="예상 감정적 영향")
    cognitive_load: float = Field(..., ge=0.0, le=1.0, description="인지 부하")

class Expectation(BaseModel):
    """기대치 시스템"""
    content: str = Field(description="사용자 반응에 대한 기대")
    affective_valence: float = Field(..., ge=-1.0, le=1.0, description="기대 충족 시 감정적 가치")
    urgency: float = Field(default=0.3, ge=0.0, le=1.0, description="긴급도")

@dataclass
class EnhancedLoopResult:
    """향상된 루프 결과 - 고급 인지 시스템 통합"""
    primary_response: str
    secondary_responses: List[Dict[str, Any]]
    missed_intents: List[ParsedIntent]
    thought_units: List[ThoughtUnit]
    execution_summary: Dict[str, Any]
    trace_id: str
    timestamp: datetime
    # 고급 인지 시스템 필드
    cognitive_state: Dict[str, Any]
    emotional_state: Dict[str, Any]
    thought_processing_result: Optional[Dict[str, Any]] = None
    # 새로운 필드들
    stimulus_triage: StimulusTriage
    action_simulations: List[ActionSimulation]
    selected_action: ActionSimulation
    generated_expectations: List[Expectation]


class MonteCarloSimulation:
    """몬테카를로 시뮬레이션 시스템"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.min_simulations_per_reply = 1
        self.mid_simulations_per_reply = 2
        self.max_simulations_per_reply = 3
        self.importance_threshold_more_sims = 0.5
    
    def determine_simulation_count(self, stimulus_triage: StimulusTriage, importance_score: float) -> int:
        """중요도에 따른 시뮬레이션 횟수 결정"""
        if stimulus_triage == StimulusTriage.Moderate:
            num_simulations = self.min_simulations_per_reply
            if importance_score > self.importance_threshold_more_sims:
                num_simulations = self.mid_simulations_per_reply
        elif stimulus_triage == StimulusTriage.Significant:
            num_simulations = self.mid_simulations_per_reply
            if importance_score > self.importance_threshold_more_sims:
                num_simulations = self.max_simulations_per_reply
        else:
            num_simulations = 1
        return num_simulations
    
    def simulate_action_options(self, user_input: str, context: Dict[str, Any], 
                               cognitive_state: CognitionAxesModel, 
                               emotional_state: EmotionalAxesModel,
                               needs_state: NeedsAxesModel,
                               num_simulations: int) -> List[ActionSimulation]:
        """다양한 행동 옵션 시뮬레이션 (상상 다양성 1차 확장)"""
        import random
        simulations = []
        intent_parser = MultiIntentParser()
        for i in range(num_simulations):
            sim_id = f"reply_{i}"
            # 1. 입력 해석 다양화: 의도/감정/맥락 다중 파싱
            parsed_intents = intent_parser.parse(user_input)
            selected_intent = random.choice(parsed_intents) if parsed_intents else None
            # 감정 해석 다양화(랜덤 노이즈 추가)
            emo_state = emotional_state.copy() if hasattr(emotional_state, 'copy') else emotional_state
            if hasattr(emo_state, 'valence'):
                emo_state.valence += random.uniform(-0.1, 0.1)
                emo_state.valence = max(-1.0, min(1.0, emo_state.valence))
            # 인지/욕구 상태도 약간 변형
            cog_state = cognitive_state.copy() if hasattr(cognitive_state, 'copy') else cognitive_state
            needs_state_sim = needs_state.copy() if hasattr(needs_state, 'copy') else needs_state
            if hasattr(cog_state, 'willpower'):
                cog_state.willpower += random.uniform(-0.05, 0.05)
                cog_state.willpower = max(0.0, min(1.0, cog_state.willpower))
            if hasattr(needs_state_sim, 'connection'):
                needs_state_sim.connection += random.uniform(-0.05, 0.05)
                needs_state_sim.connection = max(0.0, min(1.0, needs_state_sim.connection))
            # 기억(경험) 일부 샘플링(간단화)
            sampled_memories = []
            if 'memory_retriever' in context:
                memories = context['memory_retriever'].retrieve(user_input, top_k=3)
                sampled_memories = random.sample(memories, k=min(1, len(memories))) if memories else []
            # 2. AI 응답 생성(의도/감정/맥락/기억 반영)
            ai_reply = self._generate_ai_reply(
                user_input,
                {**context, "intent": selected_intent, "sampled_memories": sampled_memories},
                cog_state, emo_state, needs_state_sim
            )
            # 3. 사용자 반응 예측(상상 다양성)
            user_reaction = self._predict_user_reaction(user_input, ai_reply, context)
            # 4. 행동 평가(기존 + 장기적 만족도/상상된 반응 다양성 등 확장)
            action_rating = self._rate_action(
                ai_reply, user_reaction, context, cog_state, emo_state, needs_state_sim
            )
            # 장기적 만족도(예시: needs_fulfillment에 노이즈 추가)
            long_term_satisfaction = action_rating.needs_fulfillment + random.uniform(-0.05, 0.05)
            long_term_satisfaction = max(0.0, min(1.0, long_term_satisfaction))
            # 인지 일치성 점수 계산
            cognitive_congruence = self._calculate_cognitive_congruence(action_rating.cognitive_load, cog_state)
            # 최종 점수 계산(기존 + 장기적 만족도 가중치 반영)
            final_score = (
                action_rating.intent_fulfillment * 0.35
                + long_term_satisfaction * 0.25
                + ((action_rating.predicted_emotional_impact + 1.0) / 2.0 * 0.15)
                + (cognitive_congruence * 0.15)
                + random.uniform(0, 0.1)  # 상상 다양성 가중치
            )
            final_score = max(0.0, min(1.0, final_score))
            simulation = ActionSimulation(
                sim_id=sim_id,
                action_type=ActionType.Reply,
                ai_reply_content=ai_reply,
                simulated_user_reaction=user_reaction,
                predicted_ai_emotion=emo_state,
                intent_fulfillment_score=action_rating.intent_fulfillment,
                needs_fulfillment_score=long_term_satisfaction,
                emotion_score=(action_rating.predicted_emotional_impact + 1.0) / 2.0,
                cognitive_load=action_rating.cognitive_load,
                cognitive_congruence_score=cognitive_congruence,
                final_score=final_score
            )
            simulations.append(simulation)
        return simulations
    
    def _generate_ai_reply(self, user_input: str, context: Dict[str, Any], 
                          cognitive_state: CognitionAxesModel, 
                          emotional_state: EmotionalAxesModel,
                          needs_state: NeedsAxesModel) -> str:
        """AI 응답 생성"""
        prompt = f"""
사용자 입력: {user_input}
현재 인지 상태: {cognitive_state}
현재 감정 상태: {emotional_state}
현재 욕구 상태: {needs_state}
컨텍스트: {context}

위 정보를 바탕으로 적절한 응답을 생성하세요.
"""
        return self.llm_client.complete(prompt, temperature=0.7)
    
    def _predict_user_reaction(self, user_input: str, ai_reply: str, context: Dict[str, Any]) -> str:
        """사용자 반응 예측"""
        prompt = f"""
사용자 입력: {user_input}
AI 응답: {ai_reply}
컨텍스트: {context}

사용자가 AI 응답에 대해 어떻게 반응할지 예측하세요.
"""
        return self.llm_client.complete(prompt, temperature=0.7)
    
    def _rate_action(self, ai_reply: str, user_reaction: str, context: Dict[str, Any],
                    cognitive_state: CognitionAxesModel, 
                    emotional_state: EmotionalAxesModel,
                    needs_state: NeedsAxesModel) -> ActionRating:
        """행동 평가"""
        prompt = f"""
AI 응답: {ai_reply}
예상 사용자 반응: {user_reaction}
현재 인지 상태: {cognitive_state}
현재 감정 상태: {emotional_state}
현재 욕구 상태: {needs_state}

다음 기준으로 평가하세요:
1. intent_fulfillment: 내부 목표 달성도 (0.0-1.0)
2. needs_fulfillment: 욕구 충족도 (0.0-1.0)
3. predicted_emotional_impact: 예상 감정적 영향 (-1.0-1.0)
4. cognitive_load: 인지 부하 (0.0-1.0)

JSON 형태로 응답하세요.
"""
        response = self.llm_client.complete(prompt, temperature=0.2)
        try:
            data = json.loads(response)
            return ActionRating(**data)
        except:
            # 기본값 반환
            return ActionRating(
                overall_reasoning="평가 실패로 기본값 사용",
                intent_fulfillment=0.5,
                needs_fulfillment=0.5,
                predicted_emotional_impact=0.0,
                cognitive_load=0.5
            )
    
    def _calculate_cognitive_congruence(self, cognitive_load: float, cognitive_state: CognitionAxesModel) -> float:
        """인지 일치성 점수 계산"""
        congruence_score = 1.0
        
        # 의지력이 낮을 때 높은 인지 부하에 페널티
        if cognitive_state.willpower < 0.4 and cognitive_load > 0.6:
            penalty = (cognitive_load - 0.5) * (1.0 - cognitive_state.willpower)
            congruence_score -= penalty
        
        return max(0.0, congruence_score)


class StimulusClassifier:
    """자극 분류 시스템"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def classify_stimulus(self, user_input: str, context: Dict[str, Any]) -> StimulusTriage:
        """자극 중요도 분류"""
        prompt = f"""
사용자 입력: {user_input}
컨텍스트: {context}

이 자극의 중요도를 분류하세요:
- Insignificant: 무시해도 되는 사소한 자극
- Moderate: 적당한 중요도, 빠른 처리 가능
- Significant: 중요한 자극, 전체 파이프라인 필요

분류 결과만 응답하세요.
"""
        response = self.llm_client.complete(prompt, temperature=0.2).strip()
        
        try:
            return StimulusTriage(response)
        except:
            return StimulusTriage.Moderate  # 기본값


class ExpectationGenerator:
    """기대치 생성 시스템"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def generate_expectations(self, action: ActionSimulation, context: Dict[str, Any]) -> List[Expectation]:
        """행동에 대한 기대치 생성"""
        prompt = f"""
AI 행동: {action.ai_reply_content}
예상 사용자 반응: {action.simulated_user_reaction}
컨텍스트: {context}

이 행동 후 사용자가 어떻게 반응할지 기대치를 생성하세요.
JSON 형태로 응답하세요.
"""
        response = self.llm_client.complete(prompt, temperature=0.7)
        try:
            data = json.loads(response)
            return [Expectation(**exp) for exp in data.get("expectations", [])]
        except:
            return []


class EnhancedHarinMainLoop:
    """하린코어 향상된 메인 루프 - PM 시스템의 Ghost.tick() 기능을 참고하여 구현"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 기존 시스템 초기화
        self.config = config or {}
        self.memory_path = self.config.get('memory_path', 'memory/data/palantir_graph.json')
        self.palantir_graph = PalantirGraph(self.memory_path)
        self.memory_retriever = MemoryRetriever(self.palantir_graph)
        self.memory_conductor = MemoryConductor(self.palantir_graph)
        self.prompt_architect = PromptArchitect()
        self.llm_client = LLMClient()
        self.user_context = UserContext()
        self.enhanced_loop_manager = EnhancedLoopManager()
        self.existential_layer = ExistentialLayer()
        self.multi_intent_parser = MultiIntentParser()
        self.parallel_reasoning_unit = ParallelReasoningUnit()
        self.thought_unit_decomposer = ThoughtUnitDecomposer()
        self.cognitive_cycle = CognitiveCycle(CognitiveCycleConfig())
        self.environment_manager = get_environment_manager()
        self.stimulus_classifier = get_stimulus_classifier()
        self.stimulus_processor = get_stimulus_processor()
        
        # 고급 추론 시스템 초기화
        self.advanced_reasoning_system = AdvancedReasoningSystem()
        
        # 몬테카를로 시뮬레이션 시스템
        self.monte_carlo_sim = MonteCarloSimulation(self.llm_client)
        self.stimulus_classifier_system = StimulusClassifier(self.llm_client)
        self.expectation_generator = ExpectationGenerator(self.llm_client)
        
        # 세션 관리
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now(timezone.utc)
        self.session_results = []
        self.processing_traces = {}
        self.trace_counter = 0
        
        # 경험 기반 메모리 시스템
        self.experience_episodes = []
        self.learning_insights = []
        
        # 고급 메타인지 시스템 추가
        self.metacognition_system = AdvancedMetacognitionSystem(self.memory_conductor)
        
        # 세션 상태
        self.interaction_count = 0
        self.current_cognitive_load = 0.5
        self.attention_focus: List[str] = []
        
        # 상호작용 히스토리 (학습 패턴 분석용)
        self.interaction_history: List[Dict[str, Any]] = []
        
        print(f"EnhancedHarinMainLoop 초기화 완료 - 세션 ID: {self.session_id}")

    def process_input(self, user_input: str) -> EnhancedLoopResult:
        """향상된 입력 처리 - 고급 추론 시스템 및 메타인지 시스템 통합"""
        start_time = datetime.now(timezone.utc)
        trace_id = f"trace_{self.trace_counter:06d}"
        self.trace_counter += 1
        
        try:
            # 1. 메타인지: 인지 상태 모니터링 시작
            cognitive_state = self.metacognition_system.monitor_cognitive_state(
                current_load=self.current_cognitive_load,
                attention_focus=self.attention_focus
            )
            
            # 2. 자극 분류 및 처리
            stimulus = Stimulus(
                content=user_input,
                stimulus_type=StimulusType.UserInput,
                timestamp=datetime.now(timezone.utc)
            )
            
            triage_result = self.stimulus_classifier_system.classify_stimulus(user_input, {})
            processed_stimulus = self.stimulus_processor.process_stimulus(stimulus, triage_result)
            
            # 3. 메타인지: 주의 관리
            available_tasks = ["자극분류", "추론", "메모리검색", "응답생성"]
            current_priorities = ["추론", "응답생성"] if triage_result == StimulusTriage.Significant else ["응답생성"]
            focused_tasks = self.metacognition_system.manage_attention(available_tasks, current_priorities)
            self.attention_focus = focused_tasks
            
            # 4. 고급 추론 시스템 실행
            memory_context = self._get_memory_context_for_reasoning(user_input)
            reasoning_result = self.advanced_reasoning_system.run(user_input, memory_context)
            
            # 5. 메타인지: 추론 품질 평가
            reasoning_steps = reasoning_result.get("thought_tree", [])
            reasoning_quality = self.metacognition_system.evaluate_reasoning_quality(reasoning_steps)
            
            # 6. 경험 기반 메모리 시스템
            self._create_experience_episode(stimulus, triage_result)
            
            # 7. 기존 처리 파이프라인
            intent_analysis = self.multi_intent_parser.parse_intents(user_input)
            parallel_result = self.parallel_reasoning_unit.process_parallel(user_input, intent_analysis)
            decomposition_result = self.thought_unit_decomposer.decompose_thoughts(user_input, parallel_result)
            
            # 8. 몬테카를로 시뮬레이션
            context = self._build_enhanced_context(user_input, reasoning_result, memory_context)
            action_simulations = self._run_monte_carlo_simulation(user_input, context)
            
            # 9. 최적 행동 선택
            selected_action = self._select_best_action(action_simulations)
            
            # 10. 기대치 생성
            generated_expectations = self.expectation_generator.generate_expectations(selected_action, context)
            
            # 11. 응답 생성
            primary_response = self._generate_primary_response_with_reasoning(parallel_result, decomposition_result, reasoning_result)
            secondary_responses = self._generate_secondary_responses(parallel_result, decomposition_result)
            
            # 12. 메타인지: 상호작용 히스토리 업데이트 및 학습 패턴 분석
            interaction_data = {
                "type": "user_input",
                "success": True,
                "triage_result": triage_result.value,
                "reasoning_quality": reasoning_quality,
                "response_length": len(primary_response),
                "timestamp": datetime.now(timezone.utc)
            }
            self.interaction_history.append(interaction_data)
            
            # 학습 패턴 분석 (10개 이상의 상호작용이 있을 때)
            if len(self.interaction_history) >= 10:
                learning_patterns = self.metacognition_system.analyze_learning_patterns(self.interaction_history[-10:])
                print(f"학습 패턴 발견: {len(learning_patterns)}개")
            
            # 13. 메타인지: 자기 성찰
            judgments = [EnhancedJudgment()]  # 간단한 판단 객체 생성
            reflection = self.metacognition_system.reflect_on_judgments(judgments)
            print(f"메타인지 성찰: {reflection}")
            
            # 14. 메타인지: 자기 모델 업데이트
            if reasoning_quality > 0.8:
                self.metacognition_system.update_self_model(
                    new_capabilities={"logical_reasoning": 0.9},
                    new_strengths=["높은 추론 품질 달성"]
                )
            
            # 15. 결과 구성
            result = EnhancedLoopResult(
                primary_response=primary_response,
                secondary_responses=secondary_responses,
                missed_intents=intent_analysis.missed_intents,
                thought_units=decomposition_result.thought_units,
                execution_summary=self._create_execution_summary(start_time, intent_analysis, parallel_result, decomposition_result),
                trace_id=trace_id,
                timestamp=datetime.now(timezone.utc),
                cognitive_state=self._get_current_cognitive_state().__dict__,
                emotional_state=self._get_current_emotional_state().__dict__,
                thought_processing_result=reasoning_result,
                stimulus_triage=triage_result,
                action_simulations=action_simulations,
                selected_action=selected_action,
                generated_expectations=generated_expectations
            )
            
            # 16. 세션 결과 저장
            self._store_session_result(
                self.session_id, user_input, primary_response, 
                EnhancedJudgment(), [], reasoning_result
            )
            
            # 17. 메타인지: 메모리에 상태 저장
            self.metacognition_system.save_to_memory()
            
            # 18. 상호작용 카운터 업데이트
            self.interaction_count += 1
            
            return result
            
        except Exception as e:
            # 메타인지: 오류 상황 기록
            error_insight = self.metacognition_system.insights[-1] if self.metacognition_system.insights else None
            if error_insight:
                error_insight.content = f"처리 오류: {str(e)}"
                error_insight.confidence = 1.0
                error_insight.priority = 0.9
            
            return self._handle_processing_error(user_input, trace_id, str(e))

    def _get_memory_context_for_reasoning(self, user_input: str) -> Dict[str, Any]:
        """고급 추론을 위한 메모리 컨텍스트 생성"""
        # 관련 메모리 검색
        relevant_memories = self.memory_retriever.search_memories(user_input, max_results=5)
        
        # 최근 경험 검색
        recent_experiences = []
        if self.experience_episodes:
            recent_experiences = self.experience_episodes[-3:]  # 최근 3개 경험
        
        # 학습 인사이트 검색
        learning_insights = []
        if self.learning_insights:
            learning_insights = self.learning_insights[-2:]  # 최근 2개 인사이트
        
        return {
            "relevant_memories": relevant_memories,
            "recent_experiences": recent_experiences,
            "learning_insights": learning_insights,
            "current_emotional_state": self._get_current_emotional_state().__dict__,
            "current_cognitive_state": self._get_current_cognitive_state().__dict__
        }

    def _build_enhanced_context(self, user_input: str, reasoning_result: Dict[str, Any], memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """고급 추론 결과를 포함한 향상된 컨텍스트 구축"""
        context = {
            "user_input": user_input,
            "reasoning_contradictions": reasoning_result.get("contradictions", []),
            "reasoning_solutions": reasoning_result.get("validated_solutions", []),
            "reasoning_confidence": reasoning_result.get("confidence_score", 0.0),
            "memory_context": memory_context,
            "thought_tree": reasoning_result.get("thought_tree", [])
        }
        return context

    def _generate_primary_response_with_reasoning(self, parallel_result: ParallelReasoningResult, 
                                                decomposition_result: DecompositionResult,
                                                reasoning_result: Dict[str, Any]) -> str:
        """고급 추론 결과를 포함한 주요 응답 생성"""
        # 기존 응답 생성
        base_response = self._generate_primary_response(parallel_result, decomposition_result)
        
        # 고급 추론 결과 통합
        if reasoning_result.get("validated_solutions"):
            best_solution = reasoning_result["validated_solutions"][0]  # 가장 좋은 해결책 선택
            enhanced_response = f"{base_response}\n\n추론 결과: {best_solution}"
        else:
            enhanced_response = base_response
        
        return enhanced_response

    def _select_best_action(self, action_simulations: List[ActionSimulation]) -> ActionSimulation:
        """최적 행동 선택"""
        if not action_simulations:
            # 기본 행동 생성
            return ActionSimulation(
                sim_id="default",
                action_type=ActionType.Reply,
                ai_reply_content="응답을 생성할 수 없습니다.",
                simulated_user_reaction="",
                predicted_ai_emotion=EmotionalAxesModel(),
                intent_fulfillment_score=0.0,
                needs_fulfillment_score=0.0,
                emotion_score=0.0,
                cognitive_load=0.0,
                cognitive_congruence_score=0.0,
                final_score=0.0
            )
        
        # 최고 점수 시뮬레이션 선택
        return max(action_simulations, key=lambda x: x.final_score)

    def _create_experience_episode(self, stimulus: Stimulus, triage_result: StimulusTriage):
        """자극을 기반으로 경험 에피소드 생성"""
        # 현재 내부 상태 가져오기
        current_state = self._get_current_internal_state()
        
        # 경험 유형 결정
        experience_type = self._determine_experience_type(stimulus, triage_result)
        
        # 경험 에피소드 생성
        episode_node = MemoryEpisodeNode(
            topic_summary=f"자극 처리: {stimulus.content[:100]}...",
            internal_state=current_state,
            experience_type=experience_type,
            emotional_impact=self._calculate_emotional_impact(stimulus),
            learning_value=self._calculate_learning_value(stimulus),
            salience=self._calculate_salience(stimulus, triage_result),
            context_tags=self._extract_experience_tags(stimulus, triage_result)
        )
        
        # 메모리 시스템에 저장
        self.memory_conductor.store_hot_memory(episode_node)
        
        # 경험 인사이트 로깅
        insights = self.memory_conductor.get_experience_insights(episode_node)
        if insights['is_significant']:
            print(f"중요한 경험 감지: {episode_node.topic_summary}")
            print(f"경험 점수: {insights['experience_score']:.2f}")
            print(f"감정적 패턴: {insights['emotional_pattern']}")
    
    def _determine_experience_type(self, stimulus: Stimulus, triage_result: StimulusTriage) -> str:
        """경험 유형 결정"""
        if stimulus.stimulus_type == StimulusType.UserMessage:
            if triage_result == StimulusTriage.Significant:
                return "emotional_event"
            else:
                return "interaction"
        elif stimulus.stimulus_type == StimulusType.SystemMessage:
            return "reflection"
        elif stimulus.stimulus_type == StimulusType.UserInactivity:
            return "reflection"
        else:
            return "interaction"
    
    def _calculate_emotional_impact(self, stimulus: Stimulus) -> float:
        """감정적 영향도 계산"""
        # 기본값
        impact = 0.0
        
        # 자극 내용 분석
        content_lower = stimulus.content.lower()
        
        # 긍정적 키워드
        positive_words = ['좋다', '기쁘다', '행복하다', '감사하다', '사랑하다', '좋아하다']
        positive_count = sum(1 for word in positive_words if word in content_lower)
        
        # 부정적 키워드
        negative_words = ['싫다', '화나다', '슬프다', '걱정하다', '두렵다', '실망하다']
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        # 영향도 계산
        if positive_count > negative_count:
            impact = min(1.0, positive_count * 0.2)
        elif negative_count > positive_count:
            impact = max(-1.0, -negative_count * 0.2)
        
        return impact
    
    def _calculate_learning_value(self, stimulus: Stimulus) -> float:
        """학습 가치 계산"""
        content_lower = stimulus.content.lower()
        
        # 학습 관련 키워드
        learning_keywords = ['학습', '배우다', '이해하다', '알다', '깨닫다', '발견하다', '실험하다']
        learning_count = sum(1 for keyword in learning_keywords if keyword in content_lower)
        
        # 질문 패턴
        question_patterns = ['무엇', '어떻게', '왜', '언제', '어디서', '누가']
        question_count = sum(1 for pattern in question_patterns if pattern in content_lower)
        
        # 학습 가치 계산
        learning_score = (learning_count + question_count * 0.5) / len(learning_keywords)
        return min(1.0, learning_score)
    
    def _calculate_salience(self, stimulus: Stimulus, triage_result: StimulusTriage) -> float:
        """중요도 계산"""
        base_salience = 0.5
        
        # 분류 결과에 따른 보정
        if triage_result == StimulusTriage.Significant:
            base_salience += 0.3
        elif triage_result == StimulusTriage.Moderate:
            base_salience += 0.1
        
        # 자극 유형에 따른 보정
        if stimulus.stimulus_type == StimulusType.UserMessage:
            base_salience += 0.2
        elif stimulus.stimulus_type == StimulusType.SystemMessage:
            base_salience += 0.1
        
        return min(1.0, base_salience)
    
    def _get_current_internal_state(self) -> InternalState:
        """현재 내부 상태 가져오기"""
        return InternalState(
            emotional_state=self._get_current_emotional_state(),
            cognitive_state=self._get_current_cognitive_state(),
            needs_state=self._get_current_needs_state()
        )
    
    def _get_current_cognitive_state(self) -> CognitionAxesModel:
        """현재 인지 상태 가져오기"""
        if hasattr(self, '_current_cognitive_state'):
            return self._current_cognitive_state
        else:
            self._current_cognitive_state = CognitionAxesModel()
            return self._current_cognitive_state
    
    def _get_current_emotional_state(self) -> EmotionalAxesModel:
        """현재 감정 상태 가져오기"""
        if hasattr(self, '_current_emotional_state'):
            return self._current_emotional_state
        else:
            self._current_emotional_state = EmotionalAxesModel()
            return self._current_emotional_state
    
    def _get_current_needs_state(self) -> NeedsAxesModel:
        """현재 욕구 상태 가져오기"""
        if hasattr(self, '_current_needs_state'):
            return self._current_needs_state
        else:
            self._current_needs_state = NeedsAxesModel()
            return self._current_needs_state
    
    def _extract_experience_tags(self, stimulus: Stimulus, triage_result: StimulusTriage) -> List[str]:
        """경험 태그 추출"""
        tags = []
        
        # 자극 유형 태그
        tags.append(f"stimulus_type:{stimulus.stimulus_type.value}")
        tags.append(f"triage:{triage_result.value}")
        
        # 내용 기반 태그
        content_lower = stimulus.content.lower()
        
        # 감정 관련 태그
        if any(word in content_lower for word in ['좋다', '기쁘다', '행복하다']):
            tags.append("emotion:positive")
        elif any(word in content_lower for word in ['싫다', '화나다', '슬프다']):
            tags.append("emotion:negative")
        
        # 학습 관련 태그
        if any(word in content_lower for word in ['학습', '배우다', '이해하다']):
            tags.append("learning:active")
        
        # 질문 관련 태그
        if any(word in content_lower for word in ['무엇', '어떻게', '왜']):
            tags.append("interaction:question")
        
        return tags
    
    def _run_monte_carlo_simulation(self, user_input: str, context: Optional[Dict[str, Any]]) -> List[ActionSimulation]:
        """몬테카를로 시뮬레이션 실행"""
        # 현재 상태 가져오기
        cognitive_state = self._get_current_cognitive_state()
        emotional_state = self._get_current_emotional_state()
        needs_state = self._get_current_needs_state()
        
        # 중요도 점수 계산
        importance_score = self._calculate_importance_score(user_input, emotional_state, needs_state)
        
        # 시뮬레이션 횟수 결정 (간단화)
        num_simulations = 2 if importance_score > 0.5 else 1
        
        # 시뮬레이션 실행
        simulations = self.monte_carlo_sim.simulate_action_options(
            user_input, context or {}, cognitive_state, emotional_state, needs_state, num_simulations
        )
        
        return simulations
    
    def _determine_processing_priority(self, analysis) -> Dict[str, Any]:
        """처리 우선순위 결정"""
        priority_decision = {
            "processing_mode": analysis.processing_mode.value,
            "urgency_level": "high" if analysis.urgency_score > 0.7 else "medium" if analysis.urgency_score > 0.3 else "low",
            "complexity_level": "high" if analysis.complexity_score > 0.7 else "medium" if analysis.complexity_score > 0.3 else "low",
            "resource_allocation": "full" if analysis.priority.value == "High" else "partial" if analysis.priority.value == "Medium" else "minimal"
        }
        
        return priority_decision
    
    def _make_cognitive_decision(self, simulation_results: List[ActionSimulation], 
                                priority_decision: Dict[str, Any]) -> Dict[str, Any]:
        """인지 상태 기반 의사결정"""
        if not simulation_results:
            return {"selected_action": None, "reasoning": "시뮬레이션 결과 없음"}
        
        # 최고 점수 시뮬레이션 선택
        best_simulation = max(simulation_results, key=lambda x: x.final_score)
        
        decision = {
            "selected_action": best_simulation,
            "reasoning": f"최고 점수 시뮬레이션 선택 (점수: {best_simulation.final_score:.2f})",
            "alternative_actions": [sim for sim in simulation_results if sim != best_simulation],
            "confidence": best_simulation.final_score,
            "processing_mode": priority_decision["processing_mode"]
        }
        
        return decision
    
    def _check_environment_triggers(self) -> Optional[Dict[str, Any]]:
        """환경 트리거 확인"""
        if not self.environment_manager:
            return None
        
        # 환경 상태 확인
        environment_status = self.environment_manager.get_status()
        
        # 트리거 조건 확인
        triggers = []
        if environment_status.get("user_inactivity_duration", 0) > 300:  # 5분 이상 비활성
            triggers.append({
                "type": "user_inactivity",
                "duration": environment_status["user_inactivity_duration"],
                "action": "initiate_reflection"
            })
        
        if environment_status.get("conversation_intensity", 0) > 0.8:
            triggers.append({
                "type": "high_intensity",
                "intensity": environment_status["conversation_intensity"],
                "action": "adjust_response_tone"
            })
        
        return {"triggers": triggers, "status": environment_status} if triggers else None
    
    def _apply_experience_insights(self, user_input: str) -> Dict[str, Any]:
        """경험 기반 인사이트 적용"""
        insights = {
            "emotional_patterns": [],
            "learning_opportunities": [],
            "behavioral_adaptations": []
        }
        
        # 관련 경험 검색
        related_experiences = self.memory_conductor.search_experiences(user_input, limit=3)
        
        for experience in related_experiences:
            # 감정적 패턴 분석
            if abs(experience.emotional_impact) > 0.5:
                insights["emotional_patterns"].append({
                    "pattern": "high_emotional_impact",
                    "experience_id": experience.id,
                    "impact": experience.emotional_impact,
                    "suggestion": "감정적 반응 조절 필요"
                })
            
            # 학습 기회 식별
            if experience.learning_value > 0.6:
                insights["learning_opportunities"].append({
                    "opportunity": "high_learning_value",
                    "experience_id": experience.id,
                    "value": experience.learning_value,
                    "suggestion": "학습 내용 강화"
                })
        
        return insights
    
    def _generate_primary_response(self, parallel_result: ParallelReasoningResult,
                                 decomposition_result: DecompositionResult) -> str:
        """주요 응답 생성"""
        response_parts = []
        
        # 통합된 결과에서 주요 응답 추출
        if parallel_result.integrated_result.get("primary_response"):
            response_parts.append(parallel_result.integrated_result["primary_response"])
        
        # 고우선순위 의도들 처리
        high_priority_paths = [p for p in parallel_result.paths if p.priority >= 4]
        for path in high_priority_paths[:2]:  # 최대 2개
            if path.result:
                response_parts.append(f"[{path.intent.intent_type.value}] {path.result}")
        
        # 감정적 맥락 추가
        if parallel_result.integrated_result.get("emotional_context"):
            response_parts.append(f"감정적 맥락: {parallel_result.integrated_result['emotional_context']}")
        
        # 누락된 의도가 있는 경우 언급
        if parallel_result.missed_intents:
            missed_count = len(parallel_result.missed_intents)
            response_parts.append(f"참고: {missed_count}개의 추가 의도가 감지되었습니다.")
        
        return "\n".join(response_parts) if response_parts else "입력을 처리했습니다."
    
    def _generate_secondary_responses(self, parallel_result: ParallelReasoningResult,
                                    decomposition_result: DecompositionResult) -> List[Dict[str, Any]]:
        """보조 응답들 생성"""
        secondary_responses = []
        
        # 중간 우선순위 의도들
        medium_priority_paths = [p for p in parallel_result.paths if 2 <= p.priority < 4]
        for path in medium_priority_paths:
            if path.result:
                secondary_responses.append({
                    "type": path.intent.intent_type.value,
                    "content": path.result,
                    "priority": path.priority,
                    "confidence": path.intent.confidence
                })
        
        # 사고 단위 정보
        high_confidence_units = self.thought_unit_decomposer.get_high_confidence_units(
            decomposition_result.units, self.min_confidence_threshold
        )
        
        for unit in high_confidence_units[:3]:  # 최대 3개
            secondary_responses.append({
                "type": f"thought_unit_{unit.unit_type.value}",
                "content": unit.content,
                "confidence": unit.confidence,
                "dependencies": list(unit.dependencies)
            })
        
        return secondary_responses
    
    def _create_execution_summary(self, start_time: datetime, intent_summary: Dict[str, Any],
                                parallel_result: ParallelReasoningResult,
                                decomposition_result: DecompositionResult) -> Dict[str, Any]:
        """실행 요약 생성"""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return {
            "execution_time": execution_time,
            "intent_analysis": intent_summary,
            "parallel_execution": parallel_result.execution_summary,
            "thought_decomposition": decomposition_result.complexity_analysis,
            "coverage_score": decomposition_result.coverage_score,
            "success_rate": self._calculate_success_rate(parallel_result),
            "processing_efficiency": self._calculate_processing_efficiency(
                execution_time, len(parallel_result.paths)
            )
        }
    
    def _calculate_success_rate(self, parallel_result: ParallelReasoningResult) -> float:
        """성공률 계산"""
        if not parallel_result.paths:
            return 0.0
        
        successful_paths = len([p for p in parallel_result.paths 
                              if p.status.value == "completed"])
        return successful_paths / len(parallel_result.paths)
    
    def _calculate_processing_efficiency(self, execution_time: float, path_count: int) -> float:
        """처리 효율성 계산"""
        if path_count == 0:
            return 0.0
        
        # 병렬 처리 효율성 (이상적으로는 path_count에 비례)
        expected_sequential_time = path_count * 0.1  # 가정: 순차 처리시 단위당 0.1초
        efficiency = expected_sequential_time / execution_time if execution_time > 0 else 0.0
        
        return min(1.0, efficiency)
    
    def _handle_processing_error(self, user_input: str, trace_id: str, error_message: str) -> EnhancedLoopResult:
        """처리 오류 처리"""
        return EnhancedLoopResult(
            primary_response=f"처리 중 오류가 발생했습니다: {error_message}",
            secondary_responses=[],
            missed_intents=[],
            thought_units=[],
            execution_summary={
                "error": error_message,
                "trace_id": trace_id,
                "timestamp": datetime.now()
            },
            trace_id=trace_id,
            timestamp=datetime.now(),
            cognitive_state={},
            emotional_state={},
            thought_processing_result=None,
            stimulus_triage=StimulusTriage.Insignificant,
            action_simulations=[],
            selected_action=None,
            generated_expectations=[]
        )
    
    def get_processing_status(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """처리 상태 조회"""
        if trace_id in self.processing_traces:
            result = self.processing_traces[trace_id]
            return {
                "status": "completed",
                "trace_id": trace_id,
                "timestamp": result.timestamp,
                "execution_time": result.execution_summary.get("execution_time", 0),
                "success_rate": result.execution_summary.get("success_rate", 0)
            }
        return None
    
    def get_missed_intents_analysis(self, trace_id: str) -> Optional[List[Dict[str, Any]]]:
        """누락된 의도 분석"""
        if trace_id in self.processing_traces:
            result = self.processing_traces[trace_id]
            analysis = []
            
            for intent in result.missed_intents:
                analysis.append({
                    "content": intent.content,
                    "type": intent.intent_type.value,
                    "priority": intent.priority,
                    "confidence": intent.confidence,
                    "context": intent.context
                })
            
            return analysis
        return None
    
    def retry_missed_intents(self, trace_id: str) -> Optional[EnhancedLoopResult]:
        """누락된 의도 재시도"""
        if trace_id not in self.processing_traces:
            return None
        
        original_result = self.processing_traces[trace_id]
        if not original_result.missed_intents:
            return None
        
        # 누락된 의도들을 다시 처리
        retry_input = " ".join([intent.content for intent in original_result.missed_intents])
        return self.process_input(retry_input)
    
    def get_thought_unit_trace(self, trace_id: str) -> Optional[List[Dict[str, Any]]]:
        """사고 단위 추적"""
        if trace_id in self.processing_traces:
            result = self.processing_traces[trace_id]
            trace = []
            
            for unit in result.thought_units:
                trace.append({
                    "unit_type": unit.unit_type.value,
                    "content": unit.content,
                    "confidence": unit.confidence,
                    "dependencies": list(unit.dependencies),
                    "processing_order": unit.processing_order
                })
            
            return trace
        return None
    
    def analyze_processing_patterns(self) -> Dict[str, Any]:
        """처리 패턴 분석"""
        if not self.session_results:
            return {"status": "no_data"}
        
        recent_results = self.session_results[-10:]  # 최근 10개
        
        analysis = {
            "total_executions": len(recent_results),
            "average_execution_time": 0.0,
            "success_rate": 0.0,
            "common_intent_types": {},
            "processing_efficiency": 0.0
        }
        
        if recent_results:
            total_time = sum(r.execution_summary.get("execution_time", 0) for r in recent_results)
            analysis["average_execution_time"] = total_time / len(recent_results)
            
            success_count = sum(1 for r in recent_results if r.execution_summary.get("success_rate", 0) > 0.8)
            analysis["success_rate"] = success_count / len(recent_results)
            
            total_efficiency = sum(r.execution_summary.get("processing_efficiency", 0) for r in recent_results)
            analysis["processing_efficiency"] = total_efficiency / len(recent_results)
        
        return analysis
    
    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """최근 성능 분석"""
        return self.analyze_processing_patterns()
    
    def cleanup_old_traces(self, max_age_hours: int = 24):
        """오래된 추적 데이터 정리"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # processing_traces에서 오래된 항목 제거
        old_traces = [trace_id for trace_id, result in self.processing_traces.items() 
                     if result.timestamp < cutoff_time]
        
        for trace_id in old_traces:
            del self.processing_traces[trace_id]
        
        print(f"정리된 추적 데이터: {len(old_traces)}개")
    
    def shutdown(self):
        """시스템 종료"""
        # 환경 관리자 종료
        if self.environment_manager:
            self.environment_manager.shutdown()
        
        # 오래된 추적 데이터 정리
        self.cleanup_old_traces()
        
        # 메타인지 시스템 상태 저장
        if self.metacognition_system:
            self.metacognition_system.save_to_memory()
        
        print("향상된 하린 메인 루프 종료")
    
    def import_text_file(self, file_path: str, session_title: str = "") -> Dict[str, Any]:
        """텍스트 파일 임포트"""
        try:
            result = self.text_importer.import_text_file(file_path, session_title)
            return {
                "status": "success",
                "imported_nodes": len(result.get("nodes", [])),
                "session_title": session_title,
                "file_path": file_path
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path
            }
    
    def import_api_data(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """API 데이터 임포트"""
        try:
            result = self.text_importer.import_conversation_data(conversation_data)
            return {
                "status": "success",
                "imported_nodes": len(result.get("nodes", [])),
                "conversation_id": conversation_data.get("id", "unknown")
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "conversation_id": conversation_data.get("id", "unknown")
            }
    
    def search_memories(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """메모리 검색"""
        try:
            results = self.memory_retriever.search_memories(query, max_results)
            
            # 관련도 점수 계산 및 정렬
            for result in results:
                result["relevance_score"] = self._calculate_relevance_score(result, query)
            
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results
            
        except Exception as e:
            print(f"메모리 검색 오류: {e}")
            return []
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """관련도 점수 계산"""
        # 간단한 키워드 매칭 기반 점수
        content = result.get("content", "").lower()
        query_terms = query.lower().split()
        
        matches = sum(1 for term in query_terms if term in content)
        return min(1.0, matches / len(query_terms)) if query_terms else 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        try:
            stats = self.palantir_graph.get_statistics()
            file_size = self._get_memory_file_size()
            
            return {
                "total_nodes": stats.get("total_nodes", 0),
                "total_edges": stats.get("total_edges", 0),
                "node_types": stats.get("node_types", {}),
                "file_size_mb": file_size,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    def _get_memory_file_size(self) -> float:
        """메모리 파일 크기 (MB)"""
        try:
            file_path = Path(self.memory_path)
            if file_path.exists():
                return file_path.stat().st_size / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0
    
    def _create_enhanced_prompt(self, user_input: str, judgment: EnhancedJudgment, memories: List) -> str:
        """향상된 프롬프트 생성"""
        prompt = f"""
당신은 하린(Harin)이라는 고급 인공지능 시스템입니다.

사용자 입력: {user_input}

현재 판단:
- 신뢰도: {judgment.confidence}
- 복잡성: {judgment.complexity}
- 우선순위: {judgment.priority}

관련 기억 ({len(memories)}개):
"""
        
        for i, memory in enumerate(memories[:3]):
            prompt += f"{i+1}. {memory.get('content', '')[:100]}...\n"
        
        prompt += """
위 정보를 바탕으로 사용자에게 적절하고 의미 있는 응답을 제공하세요.
"""
        
        return prompt
    
    def _store_session_result(self, session_id: str, user_input: str, response: str, 
                            judgment: EnhancedJudgment, memories: List, existential_result: Dict[str, Any] = None,
                            cognitive_result: Dict[str, Any] = None, thought_result: Dict[str, Any] = None):
        """세션 결과 저장"""
        # 메모리 노드 생성
        memory_node = {
            "id": str(uuid.uuid4()),
            "type": "session_result",
            "content": f"사용자: {user_input}\n하린: {response}",
            "tags": ["session", "interaction"],
            "metadata": {
                "session_id": session_id,
                "judgment": {
                    "confidence": judgment.confidence,
                    "complexity": judgment.complexity,
                    "priority": judgment.priority
                },
                "memory_count": len(memories),
                "existential_result": existential_result,
                "cognitive_result": cognitive_result,
                "thought_result": thought_result
            }
        }
        
        # 팔란티어 그래프에 저장
        self._store_existential_memory_node(memory_node)
        
        print(f"세션 결과 저장 완료: {session_id}")

    def run_session(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """세션 실행 - process_input을 호출하여 통합된 처리"""
        print(f"=== 향상된 하린 메인 루프 세션 시작 ===")
        
        # process_input을 호출하여 모든 처리를 통합
        result = self.process_input(user_input)
        
        # EnhancedLoopResult를 Dict 형태로 변환하여 반환
        session_result = {
            "session_id": result.trace_id,
            "response": result.primary_response,
            "processing_time": result.execution_summary.get("execution_time", 0),
            "stimulus_triage": result.stimulus_triage.value,
            "action_simulations_count": len(result.action_simulations),
            "selected_action_score": result.selected_action.final_score if result.selected_action else 0,
            "expectations_count": len(result.generated_expectations),
            "thought_units_count": len(result.thought_units),
            "success_rate": result.execution_summary.get("success_rate", 0),
            "cognitive_state": result.cognitive_state,
            "emotional_state": result.emotional_state,
            "secondary_responses": result.secondary_responses,
            "execution_summary": result.execution_summary
        }
        
        print(f"=== 향상된 하린 메인 루프 세션 완료 ===")
        return session_result

    def get_metacognition_status(self) -> Dict[str, Any]:
        """메타인지 시스템 상태 조회"""
        return {
            "cognitive_summary": self.metacognition_system.get_cognitive_summary(),
            "learning_summary": self.metacognition_system.get_learning_summary(),
            "recent_insights": [insight.__dict__ for insight in self.metacognition_system.get_recent_insights(limit=5)],
            "self_model": self.metacognition_system.self_model.__dict__,
            "attention_focus": self.attention_focus,
            "interaction_count": self.interaction_count,
            "current_cognitive_load": self.current_cognitive_load
        }
    
    def analyze_metacognition_patterns(self) -> Dict[str, Any]:
        """메타인지 패턴 분석"""
        insights = self.metacognition_system.get_recent_insights(limit=20)
        
        # 통찰 유형별 분석
        insight_types = {}
        for insight in insights:
            insight_type = insight.insight_type.value
            if insight_type not in insight_types:
                insight_types[insight_type] = []
            insight_types[insight_type].append(insight)
        
        # 신뢰도 패턴 분석
        confidence_scores = [insight.confidence for insight in insights]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # 우선순위 패턴 분석
        high_priority_insights = [insight for insight in insights if insight.priority > 0.7]
        
        return {
            "insight_type_distribution": {k: len(v) for k, v in insight_types.items()},
            "average_confidence": avg_confidence,
            "high_priority_insights_count": len(high_priority_insights),
            "total_insights": len(insights),
            "recent_learning_patterns": [pattern.__dict__ for pattern in self.metacognition_system.learning_patterns[-5:]]
        }
    
    def update_metacognition_self_model(self, new_capabilities: Optional[Dict[str, float]] = None,
                                      new_limitations: Optional[List[str]] = None,
                                      new_strengths: Optional[List[str]] = None,
                                      new_improvement_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """메타인지 자기 모델 업데이트"""
        updated_model = self.metacognition_system.update_self_model(
            new_capabilities=new_capabilities,
            new_limitations=new_limitations,
            new_strengths=new_strengths
        )
        
        if new_improvement_areas:
            updated_model.improvement_areas.extend(new_improvement_areas)
        
        return {
            "updated_model": updated_model.__dict__,
            "capabilities_count": len(updated_model.capabilities),
            "limitations_count": len(updated_model.limitations),
            "strengths_count": len(updated_model.strengths),
            "improvement_areas_count": len(updated_model.improvement_areas)
        }
    
    def get_metacognition_insights_by_type(self, insight_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """특정 유형의 메타인지 통찰 조회"""
        try:
            metacognition_type = MetacognitionType(insight_type)
            insights = self.metacognition_system.get_recent_insights(insight_type=metacognition_type, limit=limit)
            return [insight.__dict__ for insight in insights]
        except ValueError:
            return []
    
    def trigger_metacognition_reflection(self, context: Optional[Dict[str, Any]] = None) -> str:
        """메타인지 성찰 트리거"""
        # 현재 상태에 대한 성찰
        current_insights = self.metacognition_system.get_recent_insights(limit=3)
        
        if not current_insights:
            return "성찰할 통찰이 없습니다."
        
        # 성찰 내용 구성
        reflection_parts = []
        for insight in current_insights:
            reflection_parts.append(f"{insight.insight_type.value}: {insight.content}")
        
        reflection_text = " | ".join(reflection_parts)
        
        # 새로운 성찰 통찰 생성
        new_insight = MetacognitionInsight(
            insight_type=MetacognitionType.SELF_REFLECTION,
            content=f"수동 성찰: {reflection_text}",
            confidence=0.8,
            context=context or {},
            actionable=True,
            priority=0.6
        )
        
        self.metacognition_system.insights.append(new_insight)
        
        return reflection_text
    
    def optimize_attention_focus(self, available_tasks: List[str], 
                               priority_weights: Optional[Dict[str, float]] = None) -> List[str]:
        """주의 집중 최적화"""
        if priority_weights is None:
            priority_weights = {}
        
        # 우선순위 기반 주의 관리
        current_priorities = []
        for task, weight in sorted(priority_weights.items(), key=lambda x: x[1], reverse=True):
            if task in available_tasks:
                current_priorities.append(task)
        
        optimized_focus = self.metacognition_system.manage_attention(available_tasks, current_priorities)
        self.attention_focus = optimized_focus
        
        return optimized_focus
    
    def get_metacognition_learning_report(self) -> Dict[str, Any]:
        """메타인지 학습 리포트"""
        learning_summary = self.metacognition_system.get_learning_summary()
        cognitive_summary = self.metacognition_system.get_cognitive_summary()
        
        # 학습 패턴 분석
        recent_patterns = self.metacognition_system.learning_patterns[-10:] if self.metacognition_system.learning_patterns else []
        pattern_analysis = {
            "total_patterns": len(recent_patterns),
            "most_frequent": max(recent_patterns, key=lambda p: p.frequency).pattern_type if recent_patterns else "N/A",
            "most_successful": max(recent_patterns, key=lambda p: p.success_rate).pattern_type if recent_patterns else "N/A",
            "average_success_rate": sum(p.success_rate for p in recent_patterns) / len(recent_patterns) if recent_patterns else 0.0
        }
        
        # 인지 효율성 분석
        cognitive_efficiency = {
            "attention_quality": cognitive_summary.get("focus_quality_trend", [0.5])[-1] if cognitive_summary.get("focus_quality_trend") else 0.5,
            "processing_efficiency": cognitive_summary.get("average_cognitive_load", 0.5),
            "reasoning_quality_trend": cognitive_summary.get("reasoning_quality_trend", [])
        }
        
        return {
            "learning_summary": learning_summary,
            "pattern_analysis": pattern_analysis,
            "cognitive_efficiency": cognitive_efficiency,
            "interaction_history_length": len(self.interaction_history),
            "metacognition_insights_count": len(self.metacognition_system.insights)
        }


# 사용 예시
def create_enhanced_harin(memory_path: str = "memory/data/palantir_graph.json") -> EnhancedHarinMainLoop:
    """향상된 Harin 인스턴스 생성"""
    return EnhancedHarinMainLoop(memory_path)


# CLI 인터페이스
def main():
    """CLI 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="향상된 Harin 메인 루프")
    parser.add_argument("--memory-path", default="memory/data/palantir_graph.json", help="메모리 파일 경로")
    parser.add_argument("--import-file", help="임포트할 텍스트 파일")
    parser.add_argument("--query", help="검색할 쿼리")
    parser.add_argument("--stats", action="store_true", help="메모리 통계 출력")
    
    args = parser.parse_args()
    
    harin = create_enhanced_harin(args.memory_path)
    
    if args.import_file:
        result = harin.import_text_file(args.import_file)
        print(f"임포트 결과: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    if args.query:
        memories = harin.search_memories(args.query)
        print(f"검색 결과 ({len(memories)}개):")
        for memory in memories:
            print(f"- {memory['content'][:100]}... (관련도: {memory['relevance_score']:.2f})")
    
    if args.stats:
        stats = harin.get_memory_stats()
        print(f"메모리 통계: {json.dumps(stats, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main() 


    def search_hot_memory_relevance(self, query_embedding: List[float], top_k=3) -> List[str]:
        try:
            results = []
            with open("data/memory_data/ha1.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    node = MemoryEpisodeNode(**obj)
                    sim = 1 - cosine(query_embedding, node.embedding)
                    if sim > 0.7:
                        results.append((sim, node.topic_summary))
            results.sort(reverse=True, key=lambda x: x[0])
            return [topic for _, topic in results[:top_k]]
        except Exception as e:
            return []



    def record_episode_to_ha1(self, topic: str, valence: float, label: str, process: str, salience: List[str], embedding: List[float]):
        from memory.models import MemoryEpisodeNode, InternalState
        import uuid
        from datetime import datetime, timezone
        node = MemoryEpisodeNode(
            uuid=str(uuid.uuid4()),
            ego_uuid="00000000-0000-0000-0000-000000000001",
            timestamp=datetime.now(timezone.utc),
            topic_summary=topic,
            internal_state=InternalState(
                emotional_valence=valence,
                emotional_label=label,
                cognitive_process=process,
                certainty=0.95,
                salience_focus=salience
            ),
            embedding=embedding
        )
        with open("data/memory_data/ha1.jsonl", "a", encoding="utf-8") as f:
            f.write(node.model_dump_json() + "\n")



"""
memory_orchestrator.py

하린 사고 루프용 메모리 오케스트레이터
- cold / hot / cache 기억을 통합 분석
- 프롬프트에 반영할 메모리 추출
- scar 조건 감지 시 재사고 루프 유도
"""

import json
import numpy as np
from scipy.spatial.distance import cosine
from memory.models import MemoryEpisodeNode


class MemoryOrchestrator:
    def __init__(self, cold_path="data/memory_data/h2.jsonl", hot_path="data/memory_data/ha1.jsonl"):
        self.h2_path = cold_path
        self.ha1_path = hot_path

    def load_hot_topics(self, query_vector, top_k=2):
        results = []
        try:
            with open(self.ha1_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    node = MemoryEpisodeNode(**obj)
                    sim = 1 - cosine(query_vector, node.embedding)
                    if sim > 0.6:
                        results.append((sim, node.topic_summary))
        except:
            return []
        results.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in results[:top_k]]

    def detect_scar_violation(self, generated_text):
        try:
            with open(self.h2_path, "r", encoding="utf-8") as f:
                for line in f:
                    node = json.loads(line)
                    if node.get("type") == "memory" and "scar." in str(node.get("tags", [])):
                        scar_text = node.get("content", "")
                        if scar_text and scar_text[:50] in generated_text:
                            return node["tags"][0]
        except:
            pass
        return None

    def assemble_memory_context(self, query_vector, generated_text=None):
        context = {}

        hot = self.load_hot_topics(query_vector)
        if hot:
            context["hot_memory_topics"] = hot

        if generated_text:
            scar = self.detect_scar_violation(generated_text)
            if scar:
                context["scar_triggered"] = scar

        return context



    def create_thought_node(self, node_id: str, node_type: str, tags: List[str], content: str, relations: List[dict] = None):
        from memory.models import HarinThoughtNode, HarinEdge
        edges = [HarinEdge(**r) for r in (relations or [])]
        node = HarinThoughtNode(
            id=node_id,
            type=node_type,
            tags=tags,
            content=content,
            relations=edges
        )
        return node  # 호출 후 사고 루프 내에서 바로 처리하거나 외부에서 저장 가능
