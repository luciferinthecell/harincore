"""
Harin Core Decision System - LIDA Integration
의사결정(Decision) 시스템: 옵션 생성, 평가, 최적 선택 시스템
PMM의 choose_action과 plan_action을 참고하여 구현
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import random
import math

from memory.models import Stimulus, Action, ActionType, StimulusType
from core.perception_system import PerceptionResult
from core.consciousness_system import WorkspaceState


class DecisionType(Enum):
    """의사결정 유형"""
    ROUTINE = "routine"           # 일상적 결정
    STRATEGIC = "strategic"       # 전략적 결정
    CREATIVE = "creative"         # 창의적 결정
    EMERGENCY = "emergency"       # 긴급 결정
    LEARNING = "learning"         # 학습 결정


class OptionCategory(Enum):
    """옵션 카테고리"""
    KNOWLEDGE_BASED = "knowledge_based"    # 지식 기반
    CREATIVE = "creative"                  # 창의적
    LEARNING_BASED = "learning_based"     # 학습 기반
    EXPERIENCE_BASED = "experience_based" # 경험 기반
    INTUITIVE = "intuitive"               # 직관적


class DecisionOption(BaseModel):
    """의사결정 옵션"""
    option_id: str = Field(default_factory=lambda: f"opt_{random.randint(1000, 9999)}")
    content: str
    category: OptionCategory
    action_type: ActionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_benefit: float = Field(..., ge=0.0, le=1.0)
    risk_level: float = Field(..., ge=0.0, le=1.0)
    feasibility: float = Field(..., ge=0.0, le=1.0)
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class DecisionEvaluation(BaseModel):
    """의사결정 평가"""
    option_id: str
    feasibility_score: float = Field(..., ge=0.0, le=1.0)
    effectiveness_score: float = Field(..., ge=0.0, le=1.0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    alignment_score: float = Field(..., ge=0.0, le=1.0)
    total_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evaluation_notes: str = ""


class DecisionResult(BaseModel):
    """의사결정 결과"""
    selected_option: DecisionOption
    evaluation: DecisionEvaluation
    decision_type: DecisionType
    decision_confidence: float = Field(..., ge=0.0, le=1.0)
    alternative_options: List[DecisionOption] = Field(default_factory=list)
    decision_reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class DecisionSystem:
    """의사결정 시스템 - LIDA의 Decision Making 단계 구현"""
    
    def __init__(self, knowledge_base=None, experience_base=None):
        self.knowledge_base = knowledge_base
        self.experience_base = experience_base
        self.decision_history: List[DecisionResult] = []
        
        # 의사결정 시스템 설정
        self.max_options_per_category = 5
        self.min_confidence_threshold = 0.3
        self.risk_tolerance = 0.5
        
        # 의사결정 규칙 초기화
        self._initialize_decision_rules()
    
    def _initialize_decision_rules(self):
        """의사결정 규칙 초기화"""
        self.action_type_mapping = {
            StimulusType.UserMessage: [
                ActionType.Reply,
                ActionType.ToolCall,
                ActionType.Ignore,
                ActionType.InitiateUserConversation
            ],
            StimulusType.SystemMessage: [
                ActionType.SystemMaintenance,
                ActionType.ToolCall,
                ActionType.Sleep
            ],
            StimulusType.ToolResult: [
                ActionType.Reply,
                ActionType.ToolCall,
                ActionType.ProcessMemory
            ]
        }
        
        self.decision_weights = {
            "feasibility": 0.25,
            "effectiveness": 0.30,
            "efficiency": 0.20,
            "risk": 0.15,
            "alignment": 0.10
        }
    
    def generate_options(self, processed_info: Dict[str, Any]) -> List[DecisionOption]:
        """행동 옵션 생성"""
        options = []
        
        # 기존 지식 기반 옵션
        knowledge_options = self._generate_knowledge_based_options(processed_info)
        options.extend(knowledge_options)
        
        # 창의적 옵션
        creative_options = self._generate_creative_options(processed_info)
        options.extend(creative_options)
        
        # 학습 기반 옵션
        learning_options = self._generate_learning_based_options(processed_info)
        options.extend(learning_options)
        
        # 경험 기반 옵션
        experience_options = self._generate_experience_based_options(processed_info)
        options.extend(experience_options)
        
        # 직관적 옵션
        intuitive_options = self._generate_intuitive_options(processed_info)
        options.extend(intuitive_options)
        
        return options
    
    def _generate_knowledge_based_options(self, processed_info: Dict[str, Any]) -> List[DecisionOption]:
        """기존 지식 기반 옵션 생성"""
        options = []
        
        # 처리된 정보에서 자극 타입 추출
        stimulus_type = processed_info.get("stimulus_type", StimulusType.UserMessage)
        
        # 자극 타입에 따른 기본 행동 옵션
        if stimulus_type in self.action_type_mapping:
            for action_type in self.action_type_mapping[stimulus_type]:
                option = DecisionOption(
                    content=f"기존 지식에 기반한 {action_type.value} 행동",
                    category=OptionCategory.KNOWLEDGE_BASED,
                    action_type=action_type,
                    confidence=0.7,
                    expected_benefit=0.6,
                    risk_level=0.3,
                    feasibility=0.8,
                    reasoning=f"자극 타입 {stimulus_type.value}에 대한 표준 응답"
                )
                options.append(option)
        
        return options[:self.max_options_per_category]
    
    def _generate_creative_options(self, processed_info: Dict[str, Any]) -> List[DecisionOption]:
        """창의적 옵션 생성"""
        options = []
        
        # 창의적 행동 옵션들
        creative_actions = [
            ActionType.GenerateInsight,
            ActionType.AdaptBehavior,
            ActionType.InitiateUserConversation,
            ActionType.ReflectThoughts
        ]
        
        for action_type in creative_actions:
            option = DecisionOption(
                content=f"창의적인 {action_type.value} 접근",
                category=OptionCategory.CREATIVE,
                action_type=action_type,
                confidence=0.5,
                expected_benefit=0.8,
                risk_level=0.6,
                feasibility=0.6,
                reasoning="새로운 관점과 접근 방식 시도"
            )
            options.append(option)
        
        return options[:self.max_options_per_category]
    
    def _generate_learning_based_options(self, processed_info: Dict[str, Any]) -> List[DecisionOption]:
        """학습 기반 옵션 생성"""
        options = []
        
        # 학습 관련 행동 옵션들
        learning_actions = [
            ActionType.LearnFromExperience,
            ActionType.ProcessMemory,
            ActionType.GenerateInsight
        ]
        
        for action_type in learning_actions:
            option = DecisionOption(
                content=f"학습 기반 {action_type.value}",
                category=OptionCategory.LEARNING_BASED,
                action_type=action_type,
                confidence=0.6,
                expected_benefit=0.7,
                risk_level=0.4,
                feasibility=0.7,
                reasoning="경험과 학습을 통한 개선"
            )
            options.append(option)
        
        return options[:self.max_options_per_category]
    
    def _generate_experience_based_options(self, processed_info: Dict[str, Any]) -> List[DecisionOption]:
        """경험 기반 옵션 생성"""
        options = []
        
        # 과거 경험에서 유사한 상황 찾기
        if self.experience_base:
            # 실제로는 경험 데이터베이스에서 유사한 상황 검색
            pass
        
        # 기본 경험 기반 옵션
        experience_actions = [
            ActionType.Reply,
            ActionType.ProcessMemory,
            ActionType.AdaptBehavior
        ]
        
        for action_type in experience_actions:
            option = DecisionOption(
                content=f"경험 기반 {action_type.value}",
                category=OptionCategory.EXPERIENCE_BASED,
                action_type=action_type,
                confidence=0.8,
                expected_benefit=0.6,
                risk_level=0.2,
                feasibility=0.9,
                reasoning="과거 유사한 상황에서의 성공 경험"
            )
            options.append(option)
        
        return options[:self.max_options_per_category]
    
    def _generate_intuitive_options(self, processed_info: Dict[str, Any]) -> List[DecisionOption]:
        """직관적 옵션 생성"""
        options = []
        
        # 직관적 행동 옵션들
        intuitive_actions = [
            ActionType.Reply,
            ActionType.InitiateUserConversation,
            ActionType.ReflectThoughts
        ]
        
        for action_type in intuitive_actions:
            option = DecisionOption(
                content=f"직관적 {action_type.value}",
                category=OptionCategory.INTUITIVE,
                action_type=action_type,
                confidence=0.4,
                expected_benefit=0.5,
                risk_level=0.7,
                feasibility=0.5,
                reasoning="직관과 감에 기반한 결정"
            )
            options.append(option)
        
        return options[:self.max_options_per_category]
    
    def evaluate_options(self, options: List[DecisionOption], context: Dict[str, Any]) -> List[Tuple[DecisionOption, DecisionEvaluation]]:
        """옵션 평가"""
        evaluated_options = []
        
        for option in options:
            evaluation = DecisionEvaluation(
                option_id=option.option_id,
                feasibility_score=self._evaluate_feasibility(option, context),
                effectiveness_score=self._evaluate_effectiveness(option, context),
                efficiency_score=self._evaluate_efficiency(option, context),
                risk_score=self._evaluate_risk(option, context),
                alignment_score=self._evaluate_alignment(option, context)
            )
            
            # 종합 점수 계산
            evaluation.total_score = self._calculate_total_score(evaluation)
            evaluation.confidence = self._calculate_evaluation_confidence(evaluation)
            
            evaluated_options.append((option, evaluation))
        
        return evaluated_options
    
    def _evaluate_feasibility(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """실행 가능성 평가"""
        base_feasibility = option.feasibility
        
        # 자원 요구사항 평가
        resource_availability = self._assess_resource_availability(option.resource_requirements, context)
        
        # 제약 조건 평가
        constraint_impact = self._assess_constraint_impact(option.constraints, context)
        
        # 기술적 가능성
        technical_feasibility = self._assess_technical_feasibility(option.action_type, context)
        
        # 종합 실행 가능성
        feasibility = (base_feasibility * 0.4 + 
                      resource_availability * 0.3 + 
                      constraint_impact * 0.2 + 
                      technical_feasibility * 0.1)
        
        return min(1.0, feasibility)
    
    def _evaluate_effectiveness(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """효과성 평가"""
        base_effectiveness = option.expected_benefit
        
        # 목표 달성 가능성
        goal_alignment = self._assess_goal_alignment(option, context)
        
        # 사용자 만족도 예측
        user_satisfaction = self._predict_user_satisfaction(option, context)
        
        # 장기적 효과
        long_term_effect = self._assess_long_term_effect(option, context)
        
        # 종합 효과성
        effectiveness = (base_effectiveness * 0.4 + 
                        goal_alignment * 0.3 + 
                        user_satisfaction * 0.2 + 
                        long_term_effect * 0.1)
        
        return min(1.0, effectiveness)
    
    def _evaluate_efficiency(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """효율성 평가"""
        # 자원 대비 성과
        resource_efficiency = self._calculate_resource_efficiency(option, context)
        
        # 시간 효율성
        time_efficiency = self._calculate_time_efficiency(option, context)
        
        # 처리 효율성
        processing_efficiency = self._calculate_processing_efficiency(option, context)
        
        # 종합 효율성
        efficiency = (resource_efficiency * 0.4 + 
                     time_efficiency * 0.3 + 
                     processing_efficiency * 0.3)
        
        return min(1.0, efficiency)
    
    def _evaluate_risk(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """위험도 평가"""
        base_risk = option.risk_level
        
        # 실행 위험
        execution_risk = self._assess_execution_risk(option, context)
        
        # 결과 위험
        outcome_risk = self._assess_outcome_risk(option, context)
        
        # 시스템 위험
        system_risk = self._assess_system_risk(option, context)
        
        # 종합 위험도 (낮을수록 좋음)
        risk = (base_risk * 0.4 + 
                execution_risk * 0.3 + 
                outcome_risk * 0.2 + 
                system_risk * 0.1)
        
        return min(1.0, risk)
    
    def _evaluate_alignment(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """목표 정렬도 평가"""
        # 가치 정렬
        value_alignment = self._assess_value_alignment(option, context)
        
        # 목표 정렬
        goal_alignment = self._assess_goal_alignment(option, context)
        
        # 맥락 정렬
        context_alignment = self._assess_context_alignment(option, context)
        
        # 종합 정렬도
        alignment = (value_alignment * 0.4 + 
                    goal_alignment * 0.4 + 
                    context_alignment * 0.2)
        
        return min(1.0, alignment)
    
    def _calculate_total_score(self, evaluation: DecisionEvaluation) -> float:
        """종합 점수 계산"""
        total_score = (
            evaluation.feasibility_score * self.decision_weights["feasibility"] +
            evaluation.effectiveness_score * self.decision_weights["effectiveness"] +
            evaluation.efficiency_score * self.decision_weights["efficiency"] +
            (1.0 - evaluation.risk_score) * self.decision_weights["risk"] +  # 위험도는 낮을수록 좋음
            evaluation.alignment_score * self.decision_weights["alignment"]
        )
        
        return min(1.0, total_score)
    
    def _calculate_evaluation_confidence(self, evaluation: DecisionEvaluation) -> float:
        """평가 신뢰도 계산"""
        # 평가 점수들의 일관성을 기반으로 신뢰도 계산
        scores = [
            evaluation.feasibility_score,
            evaluation.effectiveness_score,
            evaluation.efficiency_score,
            1.0 - evaluation.risk_score,
            evaluation.alignment_score
        ]
        
        # 점수 분산이 낮을수록 높은 신뢰도
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # 분산이 낮을수록 높은 신뢰도
        confidence = max(0.3, 1.0 - variance)
        
        return confidence
    
    def select_best_option(self, evaluated_options: List[Tuple[DecisionOption, DecisionEvaluation]]) -> Optional[DecisionResult]:
        """최적 옵션 선택"""
        if not evaluated_options:
            return None
        
        # 점수 기반 정렬
        sorted_options = sorted(evaluated_options, key=lambda x: x[1].total_score, reverse=True)
        
        # 최고 점수 옵션 선택
        best_option, best_evaluation = sorted_options[0]
        
        # 신뢰도 임계값 확인
        if best_evaluation.confidence < self.min_confidence_threshold:
            return None
        
        # 대안 옵션들
        alternative_options = [option for option, _ in sorted_options[1:3]]  # 상위 3개 중 나머지
        
        # 의사결정 유형 결정
        decision_type = self._determine_decision_type(best_option, best_evaluation)
        
        # 의사결정 신뢰도 계산
        decision_confidence = self._calculate_decision_confidence(sorted_options, best_evaluation)
        
        # 의사결정 이유 생성
        decision_reasoning = self._generate_decision_reasoning(best_option, best_evaluation)
        
        decision_result = DecisionResult(
            selected_option=best_option,
            evaluation=best_evaluation,
            decision_type=decision_type,
            decision_confidence=decision_confidence,
            alternative_options=alternative_options,
            decision_reasoning=decision_reasoning
        )
        
        # 히스토리에 추가
        self.decision_history.append(decision_result)
        
        return decision_result
    
    def _determine_decision_type(self, option: DecisionOption, evaluation: DecisionEvaluation) -> DecisionType:
        """의사결정 유형 결정"""
        if evaluation.risk_score > 0.7:
            return DecisionType.EMERGENCY
        elif option.category == OptionCategory.CREATIVE:
            return DecisionType.CREATIVE
        elif option.category == OptionCategory.LEARNING_BASED:
            return DecisionType.LEARNING
        elif evaluation.confidence > 0.8:
            return DecisionType.ROUTINE
        else:
            return DecisionType.STRATEGIC
    
    def _calculate_decision_confidence(self, sorted_options: List[Tuple[DecisionOption, DecisionEvaluation]], 
                                     best_evaluation: DecisionEvaluation) -> float:
        """의사결정 신뢰도 계산"""
        if len(sorted_options) < 2:
            return best_evaluation.confidence
        
        # 최고 점수와 두 번째 점수의 차이
        best_score = best_evaluation.total_score
        second_score = sorted_options[1][1].total_score
        score_difference = best_score - second_score
        
        # 점수 차이가 클수록 높은 신뢰도
        confidence_boost = min(0.3, score_difference * 2)
        
        return min(1.0, best_evaluation.confidence + confidence_boost)
    
    def _generate_decision_reasoning(self, option: DecisionOption, evaluation: DecisionEvaluation) -> str:
        """의사결정 이유 생성"""
        reasoning_parts = []
        
        reasoning_parts.append(f"선택된 옵션: {option.content}")
        reasoning_parts.append(f"카테고리: {option.category.value}")
        reasoning_parts.append(f"종합 점수: {evaluation.total_score:.2f}")
        reasoning_parts.append(f"신뢰도: {evaluation.confidence:.2f}")
        
        # 강점 분석
        strengths = []
        if evaluation.feasibility_score > 0.8:
            strengths.append("높은 실행 가능성")
        if evaluation.effectiveness_score > 0.8:
            strengths.append("높은 효과성")
        if evaluation.risk_score < 0.3:
            strengths.append("낮은 위험도")
        
        if strengths:
            reasoning_parts.append(f"주요 강점: {', '.join(strengths)}")
        
        return "; ".join(reasoning_parts)
    
    # 헬퍼 메서드들 (실제 구현에서는 더 정교한 로직 필요)
    def _assess_resource_availability(self, requirements: Dict[str, Any], context: Dict[str, Any]) -> float:
        """자원 가용성 평가"""
        return 0.8  # 기본값
    
    def _assess_constraint_impact(self, constraints: List[str], context: Dict[str, Any]) -> float:
        """제약 조건 영향 평가"""
        return 0.7  # 기본값
    
    def _assess_technical_feasibility(self, action_type: ActionType, context: Dict[str, Any]) -> float:
        """기술적 가능성 평가"""
        return 0.9  # 기본값
    
    def _assess_goal_alignment(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """목표 정렬도 평가"""
        return 0.7  # 기본값
    
    def _predict_user_satisfaction(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """사용자 만족도 예측"""
        return 0.6  # 기본값
    
    def _assess_long_term_effect(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """장기적 효과 평가"""
        return 0.5  # 기본값
    
    def _calculate_resource_efficiency(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """자원 효율성 계산"""
        return 0.7  # 기본값
    
    def _calculate_time_efficiency(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """시간 효율성 계산"""
        return 0.6  # 기본값
    
    def _calculate_processing_efficiency(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """처리 효율성 계산"""
        return 0.8  # 기본값
    
    def _assess_execution_risk(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """실행 위험 평가"""
        return option.risk_level * 0.8  # 기본 위험도 기반
    
    def _assess_outcome_risk(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """결과 위험 평가"""
        return option.risk_level * 0.6  # 기본 위험도 기반
    
    def _assess_system_risk(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """시스템 위험 평가"""
        return 0.2  # 기본값
    
    def _assess_value_alignment(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """가치 정렬도 평가"""
        return 0.8  # 기본값
    
    def _assess_context_alignment(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """맥락 정렬도 평가"""
        return 0.7  # 기본값
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """의사결정 시스템 요약"""
        if not self.decision_history:
            return {}
        
        total_decisions = len(self.decision_history)
        avg_confidence = sum(d.decision_confidence for d in self.decision_history) / total_decisions
        
        # 의사결정 유형 분포
        decision_type_distribution = {}
        for decision in self.decision_history:
            decision_type = decision.decision_type.value
            decision_type_distribution[decision_type] = decision_type_distribution.get(decision_type, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "average_confidence": avg_confidence,
            "decision_type_distribution": decision_type_distribution,
            "recent_decisions": [d.selected_option.content for d in self.decision_history[-5:]]
        }
    
    def get_decision_history(self, limit: int = 10) -> List[DecisionResult]:
        """의사결정 히스토리 조회"""
        return self.decision_history[-limit:]
    
    def clear_history(self):
        """히스토리 초기화"""
        self.decision_history.clear() 