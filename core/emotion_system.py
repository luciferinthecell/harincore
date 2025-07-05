"""
통합 감정-인지 시스템 (Integrated Emotion-Cognition System) with Multi-Layer Memory & RAG

인간의 강인함과 목표의식을 지원하는 9단계 감정-인지 시스템:
1. 동료애 (Comradeship) - 협력과 연대
2. 공감능력 (Empathy) - 이해와 연결  
3. 이해력 (Understanding) - 깊이 있는 인식
4. 믿음 (Faith) - 신뢰와 확신
5. 사고적 지구력 (Mental Endurance) - 지속적 사고
6. 신념 (Conviction) - 확신과 의지
7. 비전 (Vision) - 미래 지향
8. 현실적 판단능력 (Realistic Judgment) - 균형잡힌 의사결정
9. 통합 시스템 (Integration) - 균형과 조화

+ 다층 메모리 시스템 및 RAG 기반 기억 검색
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field

# RAG 관련 import
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("RAG components not available, using fallback memory search")


class EmotionType(Enum):
    """감정 유형"""
    COMRADESHIP = "comradeship"
    EMPATHY = "empathy"
    UNDERSTANDING = "understanding"
    FAITH = "faith"
    MENTAL_ENDURANCE = "mental_endurance"
    CONVICTION = "conviction"
    VISION = "vision"
    JUDGMENT = "judgment"
    INTEGRATION = "integration"


@dataclass
class EmotionState:
    """감정 상태"""
    emotion_type: EmotionType
    intensity: float  # 0.0 ~ 1.0
    confidence: float  # 0.0 ~ 1.0
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)


@dataclass
class CognitiveState:
    """인지 상태"""
    clarity: float  # 0.0 ~ 1.0
    focus: float  # 0.0 ~ 1.0
    energy: float  # 0.0 ~ 1.0
    complexity_tolerance: float  # 0.0 ~ 1.0
    timestamp: float = field(default_factory=time.time)


class MemoryLayer(Enum):
    """메모리 계층"""
    EPISODIC = "episodic"      # 일화 기억
    SEMANTIC = "semantic"      # 의미 기억
    EMOTIONAL = "emotional"    # 감정 기억
    PROCEDURAL = "procedural"  # 절차 기억
    META = "meta"             # 메타 기억


class MemoryItem(BaseModel):
    """메모리 아이템"""
    id: str = Field(default_factory=lambda: str(time.time()))
    content: str
    layer: MemoryLayer
    emotion_type: Optional[str] = None
    intensity: float = Field(..., ge=0.0, le=1.0)
    timestamp: float = Field(default_factory=time.time)
    context: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    tags: List[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    decay_factor: float = Field(default=0.95, ge=0.0, le=1.0)


class RAGMemorySearch(BaseModel):
    """RAG 기반 메모리 검색"""
    query: str
    layer: Optional[MemoryLayer] = None
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    emotion_filter: Optional[str] = None
    time_window: Optional[float] = None  # 초 단위


class ComradeshipSystem:
    """동료애 시스템 - 협력과 연대"""
    
    def __init__(self):
        self.team_synergy = 0.8
        self.mutual_support = 0.9
        self.collective_goals = []
        self.collaboration_history = []
        
    def build_team_cohesion(self, individual_strengths: List[str],
                           shared_objectives: List[str],
                           collaboration_history: List[float]) -> float:
        """팀 응집력 구축"""
        strength_diversity = len(set(individual_strengths)) / len(individual_strengths)
        objective_alignment = len(set(shared_objectives) & set(self.collective_goals)) / len(shared_objectives)
        collaboration_success = sum(collaboration_history) / len(collaboration_history)
        
        cohesion_score = (strength_diversity * 0.3 + 
                         objective_alignment * 0.4 + 
                         collaboration_success * 0.3)
        return min(1.0, cohesion_score)
    
    def foster_mutual_support(self, support_given: float,
                            support_received: float,
                            team_challenges: int) -> float:
        """상호 지원 육성"""
        support_balance = min(support_given, support_received) / max(support_given, support_received, 0.1)
        challenge_factor = min(1.0, team_challenges / 10.0)
        
        return min(1.0, support_balance * 0.7 + challenge_factor * 0.3)
    
    def create_collective_momentum(self, individual_motivations: List[float],
                                 shared_vision_strength: float) -> float:
        """집단 동력 생성"""
        avg_motivation = sum(individual_motivations) / len(individual_motivations)
        vision_multiplier = 1.0 + shared_vision_strength * 0.5
        return min(1.0, avg_motivation * vision_multiplier)


class EmpathySystem:
    """공감능력 시스템 - 이해와 연결"""
    
    def __init__(self):
        self.emotional_mirroring = 0.8
        self.perspective_taking = 0.9
        self.emotional_contagion = 0.7
        
    def understand_emotional_states(self, observed_emotions: Dict[str, float],
                                  context_information: Dict[str, Any],
                                  personal_experience: List[str]) -> float:
        """감정 상태 이해"""
        emotion_recognition = sum(observed_emotions.values()) / len(observed_emotions)
        context_relevance = len(context_information) / 10.0
        experience_relevance = len([exp for exp in personal_experience 
                                  if any(emotion in exp for emotion in observed_emotions)]) / len(personal_experience)
        
        understanding_score = (emotion_recognition * 0.5 + 
                             context_relevance * 0.3 + 
                             experience_relevance * 0.2)
        return min(1.0, understanding_score)
    
    def provide_emotional_support(self, support_needed: float,
                                emotional_capacity: float,
                                support_skills: List[str]) -> float:
        """감정적 지원 제공"""
        need_capacity_match = min(support_needed, emotional_capacity) / max(support_needed, emotional_capacity, 0.1)
        skill_relevance = len(support_skills) / 5.0
        
        return min(1.0, need_capacity_match * 0.7 + skill_relevance * 0.3)
    
    def maintain_emotional_boundaries(self, empathy_intensity: float,
                                    personal_stability: float,
                                    boundary_awareness: float) -> float:
        """감정적 경계 유지"""
        if empathy_intensity > 0.8 and personal_stability < 0.6:
            return max(0.3, empathy_intensity - 0.3)
        else:
            return empathy_intensity * boundary_awareness


class UnderstandingSystem:
    """이해력 시스템 - 깊이 있는 인식"""
    
    def __init__(self):
        self.knowledge_integration = 0.8
        self.pattern_recognition = 0.9
        self.principles_extraction = 0.7
        
    def build_deep_understanding(self, concept_complexity: float,
                                knowledge_connections: int,
                                practical_applications: List[str]) -> float:
        """깊이 있는 이해 구축"""
        complexity_handling = min(1.0, concept_complexity / 0.8)
        connection_density = min(1.0, knowledge_connections / 20.0)
        application_breadth = len(practical_applications) / 10.0
        
        understanding_depth = (complexity_handling * 0.4 + 
                             connection_density * 0.4 + 
                             application_breadth * 0.2)
        return min(1.0, understanding_depth)
    
    def recognize_underlying_patterns(self, surface_observations: List[str],
                                    pattern_variations: int,
                                    pattern_stability: float) -> float:
        """근본적 패턴 인식"""
        observation_quality = len(surface_observations) / 15.0
        variation_understanding = min(1.0, pattern_variations / 5.0)
        stability_recognition = pattern_stability
        
        pattern_understanding = (observation_quality * 0.3 + 
                               variation_understanding * 0.4 + 
                               stability_recognition * 0.3)
        return min(1.0, pattern_understanding)


class FaithSystem:
    """믿음 시스템 - 신뢰와 확신"""
    
    def __init__(self):
        self.self_confidence = 0.8
        self.capability_trust = 0.9
        self.growth_potential = 0.7
        
    def build_self_confidence(self, proven_capabilities: List[str],
                            past_achievements: List[float],
                            learning_ability: float) -> float:
        """자기 확신 구축"""
        capability_breadth = len(proven_capabilities) / 10.0
        achievement_consistency = sum(past_achievements) / len(past_achievements)
        learning_potential = learning_ability
        
        confidence_score = (capability_breadth * 0.3 + 
                          achievement_consistency * 0.4 + 
                          learning_potential * 0.3)
        return min(1.0, confidence_score)
    
    def trust_in_growth_potential(self, current_limitations: List[str],
                                improvement_rate: float,
                                adaptability_score: float) -> float:
        """성장 잠재력에 대한 신뢰"""
        limitation_awareness = 1.0 - (len(current_limitations) / 10.0)
        improvement_momentum = improvement_rate
        adaptability_potential = adaptability_score
        
        growth_faith = (limitation_awareness * 0.3 + 
                       improvement_momentum * 0.4 + 
                       adaptability_potential * 0.3)
        return min(1.0, growth_faith)


class MentalEnduranceSystem:
    """사고적 지구력 시스템 - 지속적 사고"""
    
    def __init__(self):
        self.focus_stamina = 0.8
        self.complexity_tolerance = 0.9
        self.mental_recovery = 0.7
        
    def build_focus_stamina(self, attention_span: float,
                           distraction_resistance: float,
                           mental_energy: float) -> float:
        """집중 지구력 구축"""
        stamina_score = (attention_span * 0.4 + 
                        distraction_resistance * 0.3 + 
                        mental_energy * 0.3)
        return min(1.0, stamina_score)
    
    def develop_complexity_tolerance(self, problem_complexity: float,
                                   solution_iterations: int,
                                   uncertainty_comfort: float) -> float:
        """복잡성 내성 개발"""
        complexity_handling = min(1.0, problem_complexity / 0.8)
        iteration_persistence = min(1.0, solution_iterations / 20.0)
        uncertainty_tolerance = uncertainty_comfort
        
        tolerance_score = (complexity_handling * 0.4 + 
                          iteration_persistence * 0.3 + 
                          uncertainty_tolerance * 0.3)
        return min(1.0, tolerance_score)


class ConvictionSystem:
    """신념 시스템 - 확신과 의지"""
    
    def __init__(self):
        self.core_values = []
        self.value_consistency = 0.9
        self.value_commitment = 0.8
        
    def strengthen_value_conviction(self, value_clarity: float,
                                  value_importance: float,
                                  value_alignment: float) -> float:
        """가치 신념 강화"""
        conviction_score = (value_clarity * 0.3 + 
                           value_importance * 0.4 + 
                           value_alignment * 0.3)
        return min(1.0, conviction_score)
    
    def maintain_value_consistency(self, value_priorities: List[str],
                                 decision_alignment: List[float],
                                 behavior_consistency: float) -> float:
        """가치 일관성 유지"""
        priority_clarity = len(value_priorities) / 5.0
        decision_consistency = sum(decision_alignment) / len(decision_alignment)
        behavior_alignment = behavior_consistency
        
        consistency_score = (priority_clarity * 0.3 + 
                           decision_consistency * 0.4 + 
                           behavior_alignment * 0.3)
        return min(1.0, consistency_score)


class VisionSystem:
    """비전 시스템 - 미래 지향"""
    
    def __init__(self):
        self.vision_clarity = 0.8
        self.vision_inspiration = 0.9
        self.vision_realism = 0.7
        
    def develop_compelling_vision(self, future_possibilities: List[str],
                                aspiration_level: float,
                                innovation_potential: float) -> float:
        """매력적인 비전 개발"""
        possibility_breadth = len(future_possibilities) / 10.0
        aspiration_intensity = aspiration_level
        innovation_capacity = innovation_potential
        
        vision_compelling = (possibility_breadth * 0.3 + 
                           aspiration_intensity * 0.4 + 
                           innovation_capacity * 0.3)
        return min(1.0, vision_compelling)
    
    def balance_vision_and_realism(self, vision_ambition: float,
                                 current_capabilities: float,
                                 resource_availability: float) -> float:
        """비전과 현실의 균형"""
        gap_appropriateness = 1.0 - abs(vision_ambition - current_capabilities)
        resource_alignment = resource_availability
        
        balance_score = (gap_appropriateness * 0.6 + 
                        resource_alignment * 0.4)
        return min(1.0, balance_score)


class JudgmentSystem:
    """현실적 판단능력 시스템 - 균형잡힌 의사결정"""
    
    def __init__(self):
        self.risk_assessment = 0.8
        self.resource_evaluation = 0.9
        self.competency_analysis = 0.7
        
    def assess_success_probability(self, goal_complexity: float,
                                 available_resources: Dict[str, float],
                                 required_competencies: List[str],
                                 current_competencies: List[str]) -> float:
        """성공 가능성 평가"""
        complexity_factor = 1.0 - goal_complexity
        resource_adequacy = sum(available_resources.values()) / len(available_resources)
        
        competency_gap = len(set(required_competencies) - set(current_competencies))
        competency_coverage = 1.0 - (competency_gap / len(required_competencies))
        
        success_probability = (complexity_factor * 0.3 + 
                             resource_adequacy * 0.4 + 
                             competency_coverage * 0.3)
        return min(1.0, success_probability)
    
    def evaluate_risk_reward_balance(self, potential_rewards: List[float],
                                   potential_risks: List[float],
                                   risk_tolerance: float) -> float:
        """위험-보상 균형 평가"""
        reward_potential = sum(potential_rewards) / len(potential_rewards)
        risk_severity = sum(potential_risks) / len(potential_risks)
        
        risk_reward_ratio = reward_potential / max(risk_severity, 0.1)
        tolerance_alignment = 1.0 - abs(risk_tolerance - risk_severity)
        
        balance_score = (risk_reward_ratio * 0.6 + 
                        tolerance_alignment * 0.4)
        return min(1.0, balance_score)


class IntegrationSystem:
    """통합 시스템 - 균형과 조화"""
    
    def __init__(self):
        self.emotional_intelligence = 0.8
        self.rational_analysis = 0.9
        self.intuitive_insight = 0.7
        
    def make_balanced_decisions(self, emotional_factors: Dict[str, float],
                               rational_factors: Dict[str, float],
                               intuitive_signals: List[str],
                               decision_context: str) -> Dict[str, float]:
        """균형잡힌 의사결정"""
        # 맥락별 가중치 조정
        if decision_context == "personal":
            emotional_weight = 0.4
            rational_weight = 0.4
            intuitive_weight = 0.2
        elif decision_context == "professional":
            emotional_weight = 0.2
            rational_weight = 0.6
            intuitive_weight = 0.2
        else:  # strategic
            emotional_weight = 0.3
            rational_weight = 0.5
            intuitive_weight = 0.2
        
        emotional_score = sum(emotional_factors.values()) / len(emotional_factors)
        rational_score = sum(rational_factors.values()) / len(rational_factors)
        intuitive_score = len(intuitive_signals) / 5.0
        
        integrated_score = (emotional_score * emotional_weight + 
                          rational_score * rational_weight + 
                          intuitive_score * intuitive_weight)
        
        return {
            'decision_quality': min(1.0, integrated_score),
            'emotional_alignment': emotional_score,
            'rational_soundness': rational_score,
            'intuitive_confidence': intuitive_score,
            'balance_achievement': 1.0 - abs(emotional_score - rational_score)
        }


class CognitiveEmotion(Enum):
    # 인지적 동기부여
    CURIOSITY = "호기심"
    ACHIEVEMENT = "성취감"
    PRIDE = "자부심"
    CHALLENGE = "도전의식"
    # 논리적 사고 지원
    FOCUS = "집중력"
    PATIENCE = "인내심"
    PRECISION = "정확성 추구"
    BALANCE = "균형감"
    # 건설적 문제해결
    CREATIVE_INTEREST = "창의적 흥미"
    COLLABORATION = "협력 의식"
    RESPONSIBILITY = "책임감"
    GROWTH_MINDSET = "성장 지향"
    # 인지적 안정성
    CONFIDENCE = "자신감"
    CALMNESS = "평온함"
    FLEXIBILITY = "유연성"
    RESILIENCE = "회복력"


class CognitiveMotivationSystem:
    """
    인지적 동기부여 시스템
    각 감정이 동기부여 및 목표지향적 행동에 미치는 영향 구체적 구현
    """
    def __init__(self):
        self.emotion_weights = {
            CognitiveEmotion.CURIOSITY: 0.15,
            CognitiveEmotion.ACHIEVEMENT: 0.2,
            CognitiveEmotion.PRIDE: 0.1,
            CognitiveEmotion.CHALLENGE: 0.2,
        }
        self.state = {e: 0.5 for e in self.emotion_weights}

    def update_emotion(self, emotion: CognitiveEmotion, intensity: float):
        self.state[emotion] = intensity

    def motivation_score(self, base_score: float) -> float:
        # 감정 상태가 동기부여에 미치는 총합 효과
        boost = sum(self.state[e] * w for e, w in self.emotion_weights.items())
        return min(1.0, base_score + boost)

    def curiosity_drive(self, curiosity: float, learning_opportunity: float) -> float:
        # 호기심이 학습 동기에 미치는 영향
        return min(1.0, curiosity * 0.6 + learning_opportunity * 0.4)

    def achievement_satisfaction(self, achievement: float, goal_completion: float) -> float:
        # 성취감이 목표 달성 만족에 미치는 영향
        return min(1.0, achievement * 0.7 + goal_completion * 0.3)

    def pride_reward(self, pride: float, logical_accuracy: float) -> float:
        # 자부심이 논리적 정확성에 대한 보상으로 작용
        return min(1.0, pride * 0.5 + logical_accuracy * 0.5)

    def challenge_engagement(self, challenge: float, problem_complexity: float) -> float:
        # 도전의식이 복잡한 문제 접근에 미치는 영향
        return min(1.0, challenge * 0.6 + problem_complexity * 0.4)


class LogicalReasoningSystem:
    """
    논리적 사고 지원 시스템
    감정이 논리적 사고 품질에 미치는 영향 구체적 구현
    """
    def __init__(self):
        self.emotion_weights = {
            CognitiveEmotion.FOCUS: 0.2,
            CognitiveEmotion.PATIENCE: 0.15,
            CognitiveEmotion.PRECISION: 0.2,
            CognitiveEmotion.BALANCE: 0.15,
        }
        self.state = {e: 0.5 for e in self.emotion_weights}

    def update_emotion(self, emotion: CognitiveEmotion, intensity: float):
        self.state[emotion] = intensity

    def reasoning_quality(self, base_score: float) -> float:
        # 감정 상태가 논리적 사고 품질에 미치는 총합 효과
        boost = sum(self.state[e] * w for e, w in self.emotion_weights.items())
        return min(1.0, base_score + boost)

    def focus_enhancement(self, focus: float, complexity: float) -> float:
        # 집중력이 복잡한 추론에 미치는 영향
        return min(1.0, focus * 0.7 + complexity * 0.3)

    def patience_persistence(self, patience: float, steps: int) -> float:
        # 인내심이 단계적 사고 지속에 미치는 영향
        return min(1.0, patience * 0.5 + min(1.0, steps / 10.0) * 0.5)

    def precision_awareness(self, precision: float, error_rate: float) -> float:
        # 정확성 추구가 오류 민감도에 미치는 영향
        return min(1.0, precision * 0.7 + (1.0 - error_rate) * 0.3)

    def balance_objectivity(self, balance: float, perspective_count: int) -> float:
        # 균형감이 다양한 관점 고려에 미치는 영향
        return min(1.0, balance * 0.6 + min(1.0, perspective_count / 5.0) * 0.4)


class ConstructiveProblemSolvingSystem:
    """
    건설적 문제해결 시스템
    감정이 창의적, 협력적, 책임감 있는 문제해결에 미치는 영향 구체적 구현
    """
    def __init__(self):
        self.emotion_weights = {
            CognitiveEmotion.CREATIVE_INTEREST: 0.2,
            CognitiveEmotion.COLLABORATION: 0.15,
            CognitiveEmotion.RESPONSIBILITY: 0.15,
            CognitiveEmotion.GROWTH_MINDSET: 0.2,
        }
        self.state = {e: 0.5 for e in self.emotion_weights}

    def update_emotion(self, emotion: CognitiveEmotion, intensity: float):
        self.state[emotion] = intensity

    def problem_solving_score(self, base_score: float) -> float:
        boost = sum(self.state[e] * w for e, w in self.emotion_weights.items())
        return min(1.0, base_score + boost)

    def creative_drive(self, creative_interest: float, alternatives: int) -> float:
        return min(1.0, creative_interest * 0.6 + min(1.0, alternatives / 5.0) * 0.4)

    def collaboration_effect(self, collaboration: float, sources: int) -> float:
        return min(1.0, collaboration * 0.6 + min(1.0, sources / 3.0) * 0.4)

    def responsibility_quality(self, responsibility: float, impact: float) -> float:
        return min(1.0, responsibility * 0.7 + impact * 0.3)

    def growth_from_failure(self, growth_mindset: float, failure_count: int) -> float:
        # 실패를 학습 기회로 인식하는 감정
        return min(1.0, growth_mindset * 0.6 + min(1.0, failure_count / 5.0) * 0.4)


class CognitiveStabilitySystem:
    """
    인지적 안정성 시스템
    감정이 인지적 안정, 회복력, 유연성에 미치는 영향 구체적 구현
    """
    def __init__(self):
        self.emotion_weights = {
            CognitiveEmotion.CONFIDENCE: 0.2,
            CognitiveEmotion.CALMNESS: 0.2,
            CognitiveEmotion.FLEXIBILITY: 0.15,
            CognitiveEmotion.RESILIENCE: 0.2,
        }
        self.state = {e: 0.5 for e in self.emotion_weights}

    def update_emotion(self, emotion: CognitiveEmotion, intensity: float):
        self.state[emotion] = intensity

    def stability_score(self, base_score: float) -> float:
        boost = sum(self.state[e] * w for e, w in self.emotion_weights.items())
        return min(1.0, base_score + boost)

    def confidence_trust(self, confidence: float, logic_score: float) -> float:
        return min(1.0, confidence * 0.7 + logic_score * 0.3)

    def calmness_effect(self, calmness: float, situation_complexity: float) -> float:
        return min(1.0, calmness * 0.6 + (1.0 - situation_complexity) * 0.4)

    def flexibility_adapt(self, flexibility: float, novelty: float) -> float:
        return min(1.0, flexibility * 0.6 + novelty * 0.4)

    def resilience_recovery(self, resilience: float, retry: int) -> float:
        return min(1.0, resilience * 0.6 + min(1.0, retry / 3.0) * 0.4)


class SelfManagementSystem:
    """자기관리 시스템 (자기 인식, 조절, 반성, 강화, 수용, 성장, 신뢰, 목표, 평가, 피드백)"""
    def self_reflection(self, recent_actions: List[str]) -> float:
        return min(1.0, len([a for a in recent_actions if '실패' in a or '성찰' in a]) / 3.0)
    def self_reinforcement(self, positive_feedback: int) -> float:
        return min(1.0, positive_feedback / 5.0)
    def self_acceptance(self, acceptance_score: float) -> float:
        return acceptance_score
    def self_growth(self, learning_events: int) -> float:
        return min(1.0, learning_events / 5.0)
    def self_trust(self, trust_events: int) -> float:
        return min(1.0, trust_events / 5.0)
    def self_goal(self, goal_achievement: float) -> float:
        return goal_achievement
    def self_evaluation(self, evaluation_score: float) -> float:
        return evaluation_score
    def self_feedback(self, feedback_count: int) -> float:
        return min(1.0, feedback_count / 5.0)


class MultiLayerMemorySystem:
    """다층 메모리 시스템"""
    
    def __init__(self):
        self.memories: Dict[MemoryLayer, List[MemoryItem]] = {
            layer: [] for layer in MemoryLayer
        }
        self.embedding_model = None
        self.memory_index: Dict[str, List[float]] = {}
        
        if RAG_AVAILABLE:
            self._initialize_rag_system()
    
    def _initialize_rag_system(self):
        """RAG 시스템 초기화"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("RAG 시스템 초기화 완료")
        except Exception as e:
            print(f"RAG 시스템 초기화 실패: {e}")
            self.embedding_model = None
    
    def add_memory(self, content: str, layer: MemoryLayer, 
                   emotion_type: Optional[str] = None, intensity: float = 0.5,
                   context: Dict[str, Any] = None, tags: List[str] = None,
                   importance: float = 0.5) -> MemoryItem:
        """메모리 추가"""
        memory_item = MemoryItem(
            content=content,
            layer=layer,
            emotion_type=emotion_type,
            intensity=intensity,
            context=context or {},
            tags=tags or [],
            importance=importance
        )
        
        # 임베딩 생성
        if self.embedding_model:
            try:
                memory_item.embedding = self.embedding_model.encode(content).tolist()
                self.memory_index[memory_item.id] = memory_item.embedding
            except Exception as e:
                print(f"임베딩 생성 실패: {e}")
        
        self.memories[layer].append(memory_item)
        
        # 메모리 크기 제한 (각 계층당 최대 1000개)
        if len(self.memories[layer]) > 1000:
            self.memories[layer] = self.memories[layer][-1000:]
        
        return memory_item
    
    def search_memories(self, search_query: RAGMemorySearch) -> List[Tuple[MemoryItem, float]]:
        """RAG 기반 메모리 검색"""
        if not self.embedding_model:
            return self._fallback_search(search_query)
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode(search_query.query)
            
            # 검색 대상 메모리 필터링
            target_memories = []
            if search_query.layer:
                target_memories = self.memories[search_query.layer]
            else:
                target_memories = [mem for memories in self.memories.values() for mem in memories]
            
            # 시간 필터 적용
            if search_query.time_window:
                current_time = time.time()
                target_memories = [
                    mem for mem in target_memories 
                    if current_time - mem.timestamp <= search_query.time_window
                ]
            
            # 감정 필터 적용
            if search_query.emotion_filter:
                target_memories = [
                    mem for mem in target_memories 
                    if mem.emotion_type == search_query.emotion_filter
                ]
            
            # 유사도 계산
            similarities = []
            for memory in target_memories:
                if memory.embedding:
                    similarity = cosine_similarity(
                        [query_embedding], [memory.embedding]
                    )[0][0]
                    
                    # 중요도와 시간에 따른 가중치 적용
                    time_weight = memory.decay_factor ** ((time.time() - memory.timestamp) / 3600)
                    weighted_similarity = similarity * memory.importance * time_weight
                    
                    if weighted_similarity >= search_query.similarity_threshold:
                        similarities.append((memory, weighted_similarity))
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:search_query.top_k]
            
        except Exception as e:
            print(f"RAG 검색 실패: {e}")
            return self._fallback_search(search_query)
    
    def _fallback_search(self, search_query: RAGMemorySearch) -> List[Tuple[MemoryItem, float]]:
        """폴백 검색 (키워드 기반)"""
        target_memories = []
        if search_query.layer:
            target_memories = self.memories[search_query.layer]
        else:
            target_memories = [mem for memories in self.memories.values() for mem in memories]
        
        results = []
        query_terms = search_query.query.lower().split()
        
        for memory in target_memories:
            score = 0.0
            content_lower = memory.content.lower()
            
            # 키워드 매칭
            for term in query_terms:
                if term in content_lower:
                    score += 1.0
            
            # 태그 매칭
            for tag in memory.tags:
                if any(term in tag.lower() for term in query_terms):
                    score += 0.5
            
            # 감정 필터
            if search_query.emotion_filter and memory.emotion_type == search_query.emotion_filter:
                score += 0.3
            
            if score > 0:
                results.append((memory, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:search_query.top_k]
    
    def get_emotional_context(self, emotion_type: str, time_window: float = 3600) -> List[MemoryItem]:
        """감정 관련 컨텍스트 조회"""
        current_time = time.time()
        emotional_memories = []
        
        for layer in MemoryLayer:
            for memory in self.memories[layer]:
                if (memory.emotion_type == emotion_type and 
                    current_time - memory.timestamp <= time_window):
                    emotional_memories.append(memory)
        
        return sorted(emotional_memories, key=lambda x: x.timestamp, reverse=True)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """메모리 요약"""
        summary = {}
        for layer in MemoryLayer:
            memories = self.memories[layer]
            summary[layer.value] = {
                "count": len(memories),
                "recent_count": len([m for m in memories if time.time() - m.timestamp < 3600]),
                "avg_importance": np.mean([m.importance for m in memories]) if memories else 0.0,
                "emotion_distribution": self._get_emotion_distribution(memories)
            }
        return summary
    
    def _get_emotion_distribution(self, memories: List[MemoryItem]) -> Dict[str, int]:
        """감정 분포 계산"""
        distribution = {}
        for memory in memories:
            if memory.emotion_type:
                distribution[memory.emotion_type] = distribution.get(memory.emotion_type, 0) + 1
        return distribution


class IntegratedEmotionSystem:
    """통합 감정-인지 시스템 메인 클래스"""
    
    def __init__(self):
        # 9단계 시스템 초기화
        self.comradeship = ComradeshipSystem()
        self.empathy = EmpathySystem()
        self.understanding = UnderstandingSystem()
        self.faith = FaithSystem()
        self.mental_endurance = MentalEnduranceSystem()
        self.conviction = ConvictionSystem()
        self.vision = VisionSystem()
        self.judgment = JudgmentSystem()
        self.integration = IntegrationSystem()
        
        # 상태 관리
        self.emotion_states: Dict[EmotionType, EmotionState] = {}
        self.cognitive_state = CognitiveState(clarity=0.5, focus=0.5, energy=0.5, complexity_tolerance=0.5)
        self.interaction_history: List[Dict[str, Any]] = []
        
        # 시스템 설정
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.balance_preference = 0.5  # 감정-이성 균형 선호도
        
        # 추가된 시스템 인스턴스화
        self.cognitive_motivation = CognitiveMotivationSystem()
        self.logical_reasoning = LogicalReasoningSystem()
        self.constructive_problem_solving = ConstructiveProblemSolvingSystem()
        self.cognitive_stability = CognitiveStabilitySystem()
        self.self_management = SelfManagementSystem()
        
        # 다층 메모리 시스템 인스턴스화
        self.multi_layer_memory = MultiLayerMemorySystem()
        
    def process_emotion_cognition(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """감정-인지 통합 처리 + 다층 메모리 통합 + 모니터링 통합"""
        # 기존 처리
        results = {}
        
        # 1. 동료애 처리
        if 'team_data' in input_data:
            results['comradeship'] = self._process_comradeship(input_data['team_data'])
        
        # 2. 공감능력 처리
        if 'empathy_data' in input_data:
            results['empathy'] = self._process_empathy(input_data['empathy_data'])
        
        # 3. 이해력 처리
        if 'understanding_data' in input_data:
            results['understanding'] = self._process_understanding(input_data['understanding_data'])
        
        # 4. 믿음 처리
        if 'faith_data' in input_data:
            results['faith'] = self._process_faith(input_data['faith_data'])
        
        # 5. 사고적 지구력 처리
        if 'endurance_data' in input_data:
            results['mental_endurance'] = self._process_mental_endurance(input_data['endurance_data'])
        
        # 6. 신념 처리
        if 'conviction_data' in input_data:
            results['conviction'] = self._process_conviction(input_data['conviction_data'])
        
        # 7. 비전 처리
        if 'vision_data' in input_data:
            results['vision'] = self._process_vision(input_data['vision_data'])
        
        # 8. 판단능력 처리
        if 'judgment_data' in input_data:
            results['judgment'] = self._process_judgment(input_data['judgment_data'])
        
        # 9. 통합 처리
        integration_result = self._process_integration(results, input_data)
        results['integration'] = integration_result
        
        # === [추가] 다층 메모리 통합 ===
        self._integrate_with_memory_system(input_data, results)
        # === [기존 기능 유지] ===
        
        # === [추가] 모니터링 통합 ===
        self._integrate_with_monitoring_system(input_data, results)
        # === [기존 기능 유지] ===
        
        # 상태 업데이트
        self._update_states(results, input_data)
        
        return results
    
    def _integrate_with_monitoring_system(self, input_data: Dict[str, Any], results: Dict[str, Any]):
        """모니터링 시스템 통합"""
        try:
            # 모니터링 시스템이 있는지 확인
            if hasattr(self, 'monitoring_system') and self.monitoring_system:
                # 감정 메트릭 수집
                emotional_metrics = self._collect_emotional_metrics(results)
                
                # 메모리 메트릭 수집
                memory_metrics = self._collect_memory_metrics()
                
                # 모니터링 시스템에 메트릭 전송
                self.monitoring_system.record_emotional_metrics(emotional_metrics)
                self.monitoring_system.record_memory_metrics(memory_metrics)
                
                # 감정 패턴 분석 결과 전송
                if 'emotional_patterns' in results:
                    self.monitoring_system.record_emotional_patterns(results['emotional_patterns'])
                
        except Exception as e:
            print(f"모니터링 시스템 통합 오류: {e}")
    
    def _collect_emotional_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """감정 메트릭 수집"""
        metrics = {}
        
        # 각 감정 시스템의 점수 수집
        for emotion_type, result in results.items():
            if isinstance(result, dict) and 'score' in result:
                metrics[f"{emotion_type}_score"] = result['score']
        
        # 통합 점수
        if 'integration' in results:
            integration = results['integration']
            metrics['emotional_balance'] = integration.get('emotional_balance', 0.5)
            metrics['cognitive_stability'] = integration.get('cognitive_stability', 0.5)
            metrics['integration_quality'] = integration.get('integration_quality', 0.5)
        
        # 메모리 인사이트
        memory_insights = self.get_memory_insights()
        if 'emotional_stability' in memory_insights:
            metrics['emotional_stability'] = memory_insights['emotional_stability']
        
        return metrics
    
    def _collect_memory_metrics(self) -> Dict[str, float]:
        """메모리 메트릭 수집"""
        memory_summary = self.multi_layer_memory.get_memory_summary()
        
        metrics = {}
        for layer_name, layer_data in memory_summary.items():
            metrics[f"{layer_name}_count"] = layer_data.get('count', 0)
            metrics[f"{layer_name}_avg_importance"] = layer_data.get('avg_importance', 0.0)
            metrics[f"{layer_name}_recent_count"] = layer_data.get('recent_count', 0)
        
        # 전체 메모리 효율성
        total_memories = sum(layer_data.get('count', 0) for layer_data in memory_summary.values())
        metrics['total_memory_count'] = total_memories
        metrics['memory_efficiency'] = min(1.0, total_memories / 1000.0)  # 정규화
        
        return metrics
    
    def set_monitoring_system(self, monitoring_system):
        """모니터링 시스템 설정"""
        self.monitoring_system = monitoring_system
        print("모니터링 시스템이 감정 시스템에 연결되었습니다.")
    
    def get_emotional_monitoring_summary(self) -> Dict[str, Any]:
        """감정 모니터링 요약 조회"""
        if not hasattr(self, 'monitoring_system') or not self.monitoring_system:
            return {"status": "monitoring_not_available"}
        
        try:
            # 메모리 인사이트 조회
            memory_insights = self.get_memory_insights()
            
            # 감정 상태 요약
            emotional_summary = {
                "memory_summary": memory_insights.get('memory_summary', {}),
                "recent_emotional_activity": memory_insights.get('recent_emotional_activity', 0),
                "dominant_emotions": memory_insights.get('dominant_emotions', {}),
                "emotional_stability": memory_insights.get('emotional_stability', 0.0),
                "total_memories": sum(
                    layer_data.get('count', 0) 
                    for layer_data in memory_insights.get('memory_summary', {}).values()
                )
            }
            
            return {
                "status": "success",
                "emotional_summary": emotional_summary,
                "monitoring_metrics": self.monitoring_system.get_metrics_summary(
                    metric_type=None, time_window=3600  # 최근 1시간
                )
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_emotional_dashboard(self, filepath: str):
        """감정 대시보드 생성"""
        if not hasattr(self, 'monitoring_system') or not self.monitoring_system:
            print("모니터링 시스템이 연결되지 않았습니다.")
            return
        
        try:
            # 감정 메트릭 데이터 수집
            emotional_metrics = self.monitoring_system.metrics.get('emotional', [])
            memory_metrics = self.monitoring_system.metrics.get('memory', [])
            
            if not emotional_metrics and not memory_metrics:
                print("감정 메트릭 데이터가 없습니다.")
                return
            
            # Plotly를 사용한 대시보드 생성
            if VISUALIZATION_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('감정 균형', '메모리 사용량', '감정 안정성', '메모리 효율성'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # 감정 균형
                if emotional_metrics:
                    timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in emotional_metrics]
                    values = [mp.value for mp in emotional_metrics]
                    
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=values, name="감정 균형", line=dict(color='purple')),
                        row=1, col=1
                    )
                
                # 메모리 사용량
                if memory_metrics:
                    timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in memory_metrics]
                    values = [mp.value for mp in memory_metrics]
                    
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=values, name="메모리 사용량", line=dict(color='green')),
                        row=1, col=2
                    )
                
                # 메모리 인사이트
                memory_insights = self.get_memory_insights()
                if memory_insights.get('dominant_emotions'):
                    emotions = list(memory_insights['dominant_emotions'].keys())
                    values = list(memory_insights['dominant_emotions'].values())
                    
                    fig.add_trace(
                        go.Bar(x=emotions, y=values, name="주요 감정", marker_color='orange'),
                        row=2, col=1
                    )
                
                # 레이아웃 설정
                fig.update_layout(
                    title="하린코어 감정 모니터링 대시보드",
                    height=800,
                    showlegend=True
                )
                
                # HTML 파일로 저장
                fig.write_html(filepath)
                print(f"감정 대시보드 생성 완료: {filepath}")
            else:
                print("시각화 라이브러리가 없어 대시보드를 생성할 수 없습니다.")
                
        except Exception as e:
            print(f"감정 대시보드 생성 실패: {e}")
    
    def export_emotional_data(self, filepath: str):
        """감정 데이터 내보내기"""
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "memory_insights": self.get_memory_insights(),
                "emotional_context": self.search_emotional_memories("", time_window=86400),  # 24시간
                "memory_summary": self.multi_layer_memory.get_memory_summary()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"감정 데이터 내보내기 완료: {filepath}")
            
        except Exception as e:
            print(f"감정 데이터 내보내기 실패: {e}")
    
    def _integrate_with_memory_system(self, input_data: Dict[str, Any], results: Dict[str, Any]):
        """다층 메모리 시스템 통합"""
        # 현재 감정 상태를 메모리에 저장
        current_emotion = self._get_current_primary_emotion(results)
        if current_emotion:
            self.multi_layer_memory.add_memory(
                content=f"감정 상태: {current_emotion}",
                layer=MemoryLayer.EMOTIONAL,
                emotion_type=current_emotion,
                intensity=results.get('integration', {}).get('emotional_balance', 0.5),
                context=input_data,
                tags=['emotion', 'state'],
                importance=0.8
            )
        
        # 감정 관련 과거 기억 검색
        if current_emotion:
            emotional_context = self.multi_layer_memory.get_emotional_context(
                current_emotion, time_window=86400  # 24시간
            )
            
            if emotional_context:
                # 감정 패턴 분석
                pattern_analysis = self._analyze_emotional_patterns(emotional_context)
                results['emotional_patterns'] = pattern_analysis
        
        # RAG 기반 감정 컨텍스트 검색
        if 'user_input' in input_data:
            search_query = RAGMemorySearch(
                query=input_data['user_input'],
                layer=MemoryLayer.EMOTIONAL,
                top_k=3,
                similarity_threshold=0.6
            )
            
            relevant_memories = self.multi_layer_memory.search_memories(search_query)
            if relevant_memories:
                results['emotional_context'] = [
                    {
                        'content': memory.content,
                        'intensity': memory.intensity,
                        'timestamp': memory.timestamp,
                        'similarity': similarity
                    }
                    for memory, similarity in relevant_memories
                ]
    
    def _get_current_primary_emotion(self, results: Dict[str, Any]) -> Optional[str]:
        """현재 주요 감정 식별"""
        emotion_scores = {}
        
        # 각 감정 시스템의 점수 수집
        for emotion_type, result in results.items():
            if isinstance(result, dict) and 'score' in result:
                emotion_scores[emotion_type] = result['score']
        
        if emotion_scores:
            # 가장 높은 점수의 감정 반환
            return max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _analyze_emotional_patterns(self, emotional_memories: List[MemoryItem]) -> Dict[str, Any]:
        """감정 패턴 분석"""
        if not emotional_memories:
            return {}
        
        # 시간대별 감정 분포
        hourly_distribution = {}
        for memory in emotional_memories:
            hour = datetime.fromtimestamp(memory.timestamp).hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        # 강도 분포
        intensities = [memory.intensity for memory in emotional_memories]
        avg_intensity = np.mean(intensities) if intensities else 0.0
        
        # 빈도 분석
        emotion_frequency = {}
        for memory in emotional_memories:
            emotion_frequency[memory.emotion_type] = emotion_frequency.get(memory.emotion_type, 0) + 1
        
        return {
            'hourly_distribution': hourly_distribution,
            'average_intensity': avg_intensity,
            'emotion_frequency': emotion_frequency,
            'total_occurrences': len(emotional_memories),
            'time_span_hours': (emotional_memories[0].timestamp - emotional_memories[-1].timestamp) / 3600
        }
    
    def search_emotional_memories(self, query: str, emotion_type: Optional[str] = None, 
                                time_window: Optional[float] = None) -> List[Tuple[MemoryItem, float]]:
        """감정 메모리 검색"""
        search_query = RAGMemorySearch(
            query=query,
            layer=MemoryLayer.EMOTIONAL,
            emotion_filter=emotion_type,
            time_window=time_window,
            top_k=10
        )
        
        return self.multi_layer_memory.search_memories(search_query)
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """메모리 인사이트 조회"""
        memory_summary = self.multi_layer_memory.get_memory_summary()
        
        # 감정 메모리 특별 분석
        emotional_memories = self.multi_layer_memory.memories[MemoryLayer.EMOTIONAL]
        if emotional_memories:
            recent_emotions = [m for m in emotional_memories if time.time() - m.timestamp < 3600]
            
            insights = {
                'memory_summary': memory_summary,
                'recent_emotional_activity': len(recent_emotions),
                'dominant_emotions': self._get_dominant_emotions(emotional_memories),
                'emotional_stability': self._calculate_emotional_stability(emotional_memories)
            }
        else:
            insights = {
                'memory_summary': memory_summary,
                'recent_emotional_activity': 0,
                'dominant_emotions': {},
                'emotional_stability': 0.0
            }
        
        return insights
    
    def _get_dominant_emotions(self, emotional_memories: List[MemoryItem]) -> Dict[str, float]:
        """주요 감정 식별"""
        emotion_scores = {}
        total_intensity = 0.0
        
        for memory in emotional_memories:
            if memory.emotion_type:
                if memory.emotion_type not in emotion_scores:
                    emotion_scores[memory.emotion_type] = 0.0
                emotion_scores[memory.emotion_type] += memory.intensity
                total_intensity += memory.intensity
        
        # 정규화
        if total_intensity > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_intensity
        
        return emotion_scores
    
    def _calculate_emotional_stability(self, emotional_memories: List[MemoryItem]) -> float:
        """감정 안정성 계산"""
        if len(emotional_memories) < 2:
            return 1.0
        
        # 최근 10개 메모리의 강도 변화 계산
        recent_memories = sorted(emotional_memories, key=lambda x: x.timestamp)[-10:]
        intensities = [memory.intensity for memory in recent_memories]
        
        # 표준편차 기반 안정성 (낮을수록 안정적)
        std_dev = np.std(intensities)
        stability = max(0.0, 1.0 - std_dev)
        
        return stability
    
    def _process_comradeship(self, team_data: Dict[str, Any]) -> Dict[str, float]:
        """동료애 처리"""
        cohesion = self.comradeship.build_team_cohesion(
            team_data.get('individual_strengths', []),
            team_data.get('shared_objectives', []),
            team_data.get('collaboration_history', [])
        )
        
        support = self.comradeship.foster_mutual_support(
            team_data.get('support_given', 0.0),
            team_data.get('support_received', 0.0),
            team_data.get('team_challenges', 0)
        )
        
        momentum = self.comradeship.create_collective_momentum(
            team_data.get('individual_motivations', []),
            team_data.get('shared_vision_strength', 0.0)
        )
        
        return {
            'team_cohesion': cohesion,
            'mutual_support': support,
            'collective_momentum': momentum,
            'overall_comradeship': (cohesion + support + momentum) / 3.0
        }
    
    def _process_empathy(self, empathy_data: Dict[str, Any]) -> Dict[str, float]:
        """공감능력 처리"""
        understanding = self.empathy.understand_emotional_states(
            empathy_data.get('observed_emotions', {}),
            empathy_data.get('context_information', {}),
            empathy_data.get('personal_experience', [])
        )
        
        support = self.empathy.provide_emotional_support(
            empathy_data.get('support_needed', 0.0),
            empathy_data.get('emotional_capacity', 0.0),
            empathy_data.get('support_skills', [])
        )
        
        boundaries = self.empathy.maintain_emotional_boundaries(
            empathy_data.get('empathy_intensity', 0.0),
            empathy_data.get('personal_stability', 0.0),
            empathy_data.get('boundary_awareness', 0.0)
        )
        
        return {
            'emotional_understanding': understanding,
            'support_provision': support,
            'boundary_maintenance': boundaries,
            'overall_empathy': (understanding + support + boundaries) / 3.0
        }
    
    def _process_understanding(self, understanding_data: Dict[str, Any]) -> Dict[str, float]:
        """이해력 처리"""
        deep_understanding = self.understanding.build_deep_understanding(
            understanding_data.get('concept_complexity', 0.0),
            understanding_data.get('knowledge_connections', 0),
            understanding_data.get('practical_applications', [])
        )
        
        pattern_recognition = self.understanding.recognize_underlying_patterns(
            understanding_data.get('surface_observations', []),
            understanding_data.get('pattern_variations', 0),
            understanding_data.get('pattern_stability', 0.0)
        )
        
        return {
            'deep_understanding': deep_understanding,
            'pattern_recognition': pattern_recognition,
            'overall_understanding': (deep_understanding + pattern_recognition) / 2.0
        }
    
    def _process_faith(self, faith_data: Dict[str, Any]) -> Dict[str, float]:
        """믿음 처리"""
        self_confidence = self.faith.build_self_confidence(
            faith_data.get('proven_capabilities', []),
            faith_data.get('past_achievements', []),
            faith_data.get('learning_ability', 0.0)
        )
        
        growth_faith = self.faith.trust_in_growth_potential(
            faith_data.get('current_limitations', []),
            faith_data.get('improvement_rate', 0.0),
            faith_data.get('adaptability_score', 0.0)
        )
        
        return {
            'self_confidence': self_confidence,
            'growth_faith': growth_faith,
            'overall_faith': (self_confidence + growth_faith) / 2.0
        }
    
    def _process_mental_endurance(self, endurance_data: Dict[str, Any]) -> Dict[str, float]:
        """사고적 지구력 처리"""
        focus_stamina = self.mental_endurance.build_focus_stamina(
            endurance_data.get('attention_span', 0.0),
            endurance_data.get('distraction_resistance', 0.0),
            endurance_data.get('mental_energy', 0.0)
        )
        
        complexity_tolerance = self.mental_endurance.develop_complexity_tolerance(
            endurance_data.get('problem_complexity', 0.0),
            endurance_data.get('solution_iterations', 0),
            endurance_data.get('uncertainty_comfort', 0.0)
        )
        
        return {
            'focus_stamina': focus_stamina,
            'complexity_tolerance': complexity_tolerance,
            'overall_endurance': (focus_stamina + complexity_tolerance) / 2.0
        }
    
    def _process_conviction(self, conviction_data: Dict[str, Any]) -> Dict[str, float]:
        """신념 처리"""
        value_conviction = self.conviction.strengthen_value_conviction(
            conviction_data.get('value_clarity', 0.0),
            conviction_data.get('value_importance', 0.0),
            conviction_data.get('value_alignment', 0.0)
        )
        
        value_consistency = self.conviction.maintain_value_consistency(
            conviction_data.get('value_priorities', []),
            conviction_data.get('decision_alignment', []),
            conviction_data.get('behavior_consistency', 0.0)
        )
        
        return {
            'value_conviction': value_conviction,
            'value_consistency': value_consistency,
            'overall_conviction': (value_conviction + value_consistency) / 2.0
        }
    
    def _process_vision(self, vision_data: Dict[str, Any]) -> Dict[str, float]:
        """비전 처리"""
        compelling_vision = self.vision.develop_compelling_vision(
            vision_data.get('future_possibilities', []),
            vision_data.get('aspiration_level', 0.0),
            vision_data.get('innovation_potential', 0.0)
        )
        
        vision_balance = self.vision.balance_vision_and_realism(
            vision_data.get('vision_ambition', 0.0),
            vision_data.get('current_capabilities', 0.0),
            vision_data.get('resource_availability', 0.0)
        )
        
        return {
            'compelling_vision': compelling_vision,
            'vision_balance': vision_balance,
            'overall_vision': (compelling_vision + vision_balance) / 2.0
        }
    
    def _process_judgment(self, judgment_data: Dict[str, Any]) -> Dict[str, float]:
        """판단능력 처리"""
        success_probability = self.judgment.assess_success_probability(
            judgment_data.get('goal_complexity', 0.0),
            judgment_data.get('available_resources', {}),
            judgment_data.get('required_competencies', []),
            judgment_data.get('current_competencies', [])
        )
        
        risk_reward_balance = self.judgment.evaluate_risk_reward_balance(
            judgment_data.get('potential_rewards', []),
            judgment_data.get('potential_risks', []),
            judgment_data.get('risk_tolerance', 0.0)
        )
        
        return {
            'success_probability': success_probability,
            'risk_reward_balance': risk_reward_balance,
            'overall_judgment': (success_probability + risk_reward_balance) / 2.0
        }
    
    def _process_integration(self, all_results: Dict[str, Dict[str, float]], 
                           context: Dict[str, Any]) -> Dict[str, float]:
        """통합 처리"""
        # 모든 시스템 결과를 종합
        emotional_factors = {
            'comradeship': all_results.get('comradeship', {}).get('overall_comradeship', 0.0),
            'empathy': all_results.get('empathy', {}).get('overall_empathy', 0.0),
            'faith': all_results.get('faith', {}).get('overall_faith', 0.0)
        }
        
        rational_factors = {
            'understanding': all_results.get('understanding', {}).get('overall_understanding', 0.0),
            'judgment': all_results.get('judgment', {}).get('overall_judgment', 0.0),
            'vision': all_results.get('vision', {}).get('overall_vision', 0.0)
        }
        
        intuitive_signals = [
            'conviction' if all_results.get('conviction', {}).get('overall_conviction', 0.0) > 0.7 else '',
            'endurance' if all_results.get('mental_endurance', {}).get('overall_endurance', 0.0) > 0.7 else ''
        ]
        intuitive_signals = [signal for signal in intuitive_signals if signal]
        
        decision_context = context.get('decision_context', 'strategic')
        
        balanced_decision = self.integration.make_balanced_decisions(
            emotional_factors, rational_factors, intuitive_signals, decision_context
        )
        
        return balanced_decision
    
    def _update_states(self, results: Dict[str, Dict[str, float]], context: Dict[str, Any]):
        """상태 업데이트"""
        timestamp = time.time()
        
        # 감정 상태 업데이트
        for emotion_type in EmotionType:
            if emotion_type.value in results:
                overall_score = results[emotion_type.value].get(f'overall_{emotion_type.value}', 0.0)
                self.emotion_states[emotion_type] = EmotionState(
                    emotion_type=emotion_type,
                    intensity=overall_score,
                    confidence=min(1.0, overall_score + 0.2),
                    timestamp=timestamp,
                    context=context,
                    triggers=context.get('emotional_triggers', [])
                )
        
        # 인지 상태 업데이트
        cognitive_demands = context.get('cognitive_demands', {})
        self.cognitive_state = CognitiveState(
            clarity=results.get('understanding', {}).get('overall_understanding', 0.0),
            focus=results.get('mental_endurance', {}).get('focus_stamina', 0.0),
            energy=1.0 - (len(cognitive_demands) / 10.0),  # 요구사항이 많을수록 에너지 감소
            complexity_tolerance=results.get('mental_endurance', {}).get('complexity_tolerance', 0.0),
            timestamp=timestamp
        )
        
        # 상호작용 기록
        self.interaction_history.append({
            'timestamp': timestamp,
            'results': results,
            'context': context,
            'emotion_states': {k.value: v.intensity for k, v in self.emotion_states.items()},
            'cognitive_state': {
                'clarity': self.cognitive_state.clarity,
                'focus': self.cognitive_state.focus,
                'energy': self.cognitive_state.energy
            }
        })
    
    def get_system_summary(self) -> Dict[str, Any]:
        """시스템 요약 정보"""
        current_time = time.time()
        
        # 최근 감정 상태
        recent_emotions = {}
        for emotion_type, state in self.emotion_states.items():
            if current_time - state.timestamp < 3600:  # 1시간 이내
                recent_emotions[emotion_type.value] = {
                    'intensity': state.intensity,
                    'confidence': state.confidence,
                    'age_minutes': (current_time - state.timestamp) / 60
                }
        
        # 시스템 건강도
        system_health = {
            'emotional_balance': self._calculate_emotional_balance(),
            'cognitive_stability': self._calculate_cognitive_stability(),
            'integration_quality': self._calculate_integration_quality(),
            'learning_progress': self._calculate_learning_progress()
        }
        
        return {
            'recent_emotions': recent_emotions,
            'cognitive_state': {
                'clarity': self.cognitive_state.clarity,
                'focus': self.cognitive_state.focus,
                'energy': self.cognitive_state.energy,
                'complexity_tolerance': self.cognitive_state.complexity_tolerance
            },
            'system_health': system_health,
            'interaction_count': len(self.interaction_history),
            'last_update': datetime.fromtimestamp(current_time).isoformat()
        }
    
    def _calculate_emotional_balance(self) -> float:
        """감정 균형 계산"""
        if not self.emotion_states:
            return 0.5
        
        intensities = [state.intensity for state in self.emotion_states.values()]
        return 1.0 - (max(intensities) - min(intensities))  # 균형일수록 높은 점수
    
    def _calculate_cognitive_stability(self) -> float:
        """인지 안정성 계산"""
        return (self.cognitive_state.clarity + 
                self.cognitive_state.focus + 
                self.cognitive_state.energy) / 3.0
    
    def _calculate_integration_quality(self) -> float:
        """통합 품질 계산"""
        if len(self.interaction_history) < 2:
            return 0.5
        
        recent_results = self.interaction_history[-1]['results']
        if 'integration' in recent_results:
            return recent_results['integration'].get('balance_achievement', 0.5)
        return 0.5
    
    def _calculate_learning_progress(self) -> float:
        """학습 진행도 계산"""
        if len(self.interaction_history) < 5:
            return 0.5
        
        # 최근 5개 상호작용의 평균 점수
        recent_scores = []
        for interaction in self.interaction_history[-5:]:
            if 'integration' in interaction['results']:
                recent_scores.append(
                    interaction['results']['integration'].get('decision_quality', 0.5)
                )
        
        if recent_scores:
            return sum(recent_scores) / len(recent_scores)
        return 0.5
    
    def save_state(self, filepath: str):
        """상태 저장"""
        state_data = {
            'emotion_states': {
                k.value: {
                    'intensity': v.intensity,
                    'confidence': v.confidence,
                    'timestamp': v.timestamp,
                    'context': v.context,
                    'triggers': v.triggers
                } for k, v in self.emotion_states.items()
            },
            'cognitive_state': {
                'clarity': self.cognitive_state.clarity,
                'focus': self.cognitive_state.focus,
                'energy': self.cognitive_state.energy,
                'complexity_tolerance': self.cognitive_state.complexity_tolerance,
                'timestamp': self.cognitive_state.timestamp
            },
            'interaction_history': self.interaction_history[-100:],  # 최근 100개만 저장
            'system_settings': {
                'learning_rate': self.learning_rate,
                'adaptation_threshold': self.adaptation_threshold,
                'balance_preference': self.balance_preference
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
    
    def load_state(self, filepath: str):
        """상태 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # 감정 상태 복원
            self.emotion_states.clear()
            for emotion_value, state_info in state_data.get('emotion_states', {}).items():
                emotion_type = EmotionType(emotion_value)
                self.emotion_states[emotion_type] = EmotionState(
                    emotion_type=emotion_type,
                    intensity=state_info['intensity'],
                    confidence=state_info['confidence'],
                    timestamp=state_info['timestamp'],
                    context=state_info['context'],
                    triggers=state_info['triggers']
                )
            
            # 인지 상태 복원
            cognitive_info = state_data.get('cognitive_state', {})
            self.cognitive_state = CognitiveState(
                clarity=cognitive_info.get('clarity', 0.5),
                focus=cognitive_info.get('focus', 0.5),
                energy=cognitive_info.get('energy', 0.5),
                complexity_tolerance=cognitive_info.get('complexity_tolerance', 0.5),
                timestamp=cognitive_info.get('timestamp', time.time())
            )
            
            # 상호작용 기록 복원
            self.interaction_history = state_data.get('interaction_history', [])
            
            # 시스템 설정 복원
            settings = state_data.get('system_settings', {})
            self.learning_rate = settings.get('learning_rate', 0.1)
            self.adaptation_threshold = settings.get('adaptation_threshold', 0.7)
            self.balance_preference = settings.get('balance_preference', 0.5)
            
        except FileNotFoundError:
            print(f"상태 파일을 찾을 수 없습니다: {filepath}")
        except Exception as e:
            print(f"상태 로드 중 오류 발생: {e}")

    def process_cognitive_extensions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """확장 인지 시스템 처리"""
        context = input_data.get('context', {})
        results = {}
        # 인지적 동기부여
        # if 'goals' in context:
        #     results['goal_clarity'] = self.cognitive_motivation.evaluate_goal_clarity(context['goals'])
        # if 'motivation' in context and 'challenge_level' in context:
        #     results['intrinsic_motivation'] = self.cognitive_motivation.assess_intrinsic_motivation(
        #         context['motivation'], context['challenge_level'])
        # 논리적 사고 지원
        # if 'arguments' in context:
        #     results['logical_consistency'] = self.logical_reasoning.check_logical_consistency(context['arguments'])
        # if 'options' in context:
        #     results['alternative_count'] = self.logical_reasoning.search_alternatives(context['options'])
        # 건설적 문제해결
        # if 'problem' in context:
        #     results['redefined_problem'] = self.constructive_problem_solving.redefine_problem(context['problem'])
        #     results['alternatives'] = self.constructive_problem_solving.generate_alternatives(context['problem'])
        # 인지적 안정성
        # if 'stress_level' in context and 'recovery' in context:
        #     results['stress_resilience'] = self.cognitive_stability.measure_stress_resilience(
        #         context['stress_level'], context['recovery'])
        # 자기관리
        if 'recent_actions' in context:
            results['self_reflection'] = self.self_management.self_reflection(context['recent_actions'])
        return results

    def integrate_emotion_cognition(self, cognitive_score: float, emotion_score: float) -> float:
        """
        감정이 인지적 성과에 미치는 통합 효과
        - 감정이 논리적 사고를 방해하지 않고 지원
        - 감정 점수는 0.0~1.0로 클리핑되어 과도하지 않게 인지 효율성 유지
        - 감정이 인지적 기능을 강화하는 방향으로만 작용
        """
        return min(1.0, cognitive_score * 0.7 + emotion_score * 0.3)

    def process_emotion_cognition_with_affect(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        감정-인지 통합 처리(감정 상태가 인지적 기능에 미치는 실제 영향 포함)
        각 인지 시스템별 감정 상태를 받아 인지적 성과를 강화
        """
        context = input_data.get('context', {})
        results = {}
        # 인지적 동기부여
        base_motivation = 0.5
        curiosity = context.get('curiosity', 0.5)
        achievement = context.get('achievement', 0.5)
        pride = context.get('pride', 0.5)
        challenge = context.get('challenge', 0.5)
        self.cognitive_motivation.update_emotion(CognitiveEmotion.CURIOSITY, curiosity)
        self.cognitive_motivation.update_emotion(CognitiveEmotion.ACHIEVEMENT, achievement)
        self.cognitive_motivation.update_emotion(CognitiveEmotion.PRIDE, pride)
        self.cognitive_motivation.update_emotion(CognitiveEmotion.CHALLENGE, challenge)
        motivation_score = self.cognitive_motivation.motivation_score(base_motivation)
        # 논리적 사고 지원
        base_reasoning = 0.5
        focus = context.get('focus', 0.5)
        patience = context.get('patience', 0.5)
        precision = context.get('precision', 0.5)
        balance = context.get('balance', 0.5)
        self.logical_reasoning.update_emotion(CognitiveEmotion.FOCUS, focus)
        self.logical_reasoning.update_emotion(CognitiveEmotion.PATIENCE, patience)
        self.logical_reasoning.update_emotion(CognitiveEmotion.PRECISION, precision)
        self.logical_reasoning.update_emotion(CognitiveEmotion.BALANCE, balance)
        reasoning_score = self.logical_reasoning.reasoning_quality(base_reasoning)
        # 건설적 문제해결
        base_problem = 0.5
        creative_interest = context.get('creative_interest', 0.5)
        collaboration = context.get('collaboration', 0.5)
        responsibility = context.get('responsibility', 0.5)
        growth_mindset = context.get('growth_mindset', 0.5)
        self.constructive_problem_solving.update_emotion(CognitiveEmotion.CREATIVE_INTEREST, creative_interest)
        self.constructive_problem_solving.update_emotion(CognitiveEmotion.COLLABORATION, collaboration)
        self.constructive_problem_solving.update_emotion(CognitiveEmotion.RESPONSIBILITY, responsibility)
        self.constructive_problem_solving.update_emotion(CognitiveEmotion.GROWTH_MINDSET, growth_mindset)
        problem_score = self.constructive_problem_solving.problem_solving_score(base_problem)
        # 인지적 안정성
        base_stability = 0.5
        confidence = context.get('confidence', 0.5)
        calmness = context.get('calmness', 0.5)
        flexibility = context.get('flexibility', 0.5)
        resilience = context.get('resilience', 0.5)
        self.cognitive_stability.update_emotion(CognitiveEmotion.CONFIDENCE, confidence)
        self.cognitive_stability.update_emotion(CognitiveEmotion.CALMNESS, calmness)
        self.cognitive_stability.update_emotion(CognitiveEmotion.FLEXIBILITY, flexibility)
        self.cognitive_stability.update_emotion(CognitiveEmotion.RESILIENCE, resilience)
        stability_score = self.cognitive_stability.stability_score(base_stability)
        # 통합 점수
        total_cognition = (motivation_score + reasoning_score + problem_score + stability_score) / 4.0
        # 감정-인지 통합(과도하지 않게)
        integrated_score = self.integrate_emotion_cognition(total_cognition, max(curiosity, achievement, pride, challenge, focus, patience, precision, balance, creative_interest, collaboration, responsibility, growth_mindset, confidence, calmness, flexibility, resilience))
        results['motivation_score'] = motivation_score
        results['reasoning_score'] = reasoning_score
        results['problem_score'] = problem_score
        results['stability_score'] = stability_score
        results['integrated_cognition'] = integrated_score
        return results


# 사용 예시
if __name__ == "__main__":
    # 감정 시스템 초기화
    emotion_system = IntegratedEmotionSystem()
    
    # 샘플 입력 데이터
    sample_input = {
        'context': {
            'team_interaction': {
                'individual_strengths': ['leadership', 'creativity', 'analytics'],
                'shared_objectives': ['project_success', 'team_growth'],
                'collaboration_history': [0.8, 0.9, 0.7],
                'support_given': 0.8,
                'support_received': 0.7,
                'team_challenges': 3,
                'individual_motivations': [0.8, 0.9, 0.7],
                'shared_vision_strength': 0.8
            },
            'emotional_observation': {
                'observed_emotions': {'joy': 0.7, 'concern': 0.3},
                'context_information': {'project_deadline': 'urgent', 'team_morale': 'high'},
                'personal_experience': ['similar_project_success', 'team_collaboration'],
                'support_needed': 0.6,
                'emotional_capacity': 0.8,
                'support_skills': ['active_listening', 'encouragement'],
                'empathy_intensity': 0.7,
                'personal_stability': 0.8,
                'boundary_awareness': 0.9
            },
            'decision_context': 'professional'
        }
    }
    
    # 감정-인지 처리
    results = emotion_system.process_emotion_cognition(sample_input)
    
    # 결과 출력
    print("=== 감정-인지 시스템 처리 결과 ===")
    for system_name, system_results in results.items():
        print(f"\n{system_name.upper()}:")
        for metric, value in system_results.items():
            print(f"  {metric}: {value:.3f}")
    
    # 시스템 요약
    summary = emotion_system.get_system_summary()
    print(f"\n=== 시스템 요약 ===")
    print(f"감정 균형: {summary['system_health']['emotional_balance']:.3f}")
    print(f"인지 안정성: {summary['system_health']['cognitive_stability']:.3f}")
    print(f"통합 품질: {summary['system_health']['integration_quality']:.3f}")
    print(f"학습 진행도: {summary['system_health']['learning_progress']:.3f}")
    