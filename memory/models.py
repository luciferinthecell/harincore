"""
Harin Core Memory Models - Lida Integration
심리학적 상태 모델과 인지 구조를 통합한 메모리 시스템
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict, replace
import uuid
import math
import os
import tempfile
import json

# StrEnum 호환성 처리
try:
    from enum import StrEnum
except ImportError:
    # Python 3.11 미만 버전을 위한 대체
    from enum import Enum
    class StrEnum(str, Enum):
        pass

# pydantic import
try:
    from pydantic import BaseModel, Field, model_validator
except ImportError:
    # pydantic이 없는 경우를 위한 대체
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(default=None, **kwargs):
        return default
    
    def model_validator(mode):
        def decorator(func):
            return func
        return decorator

class InternalState(BaseModel):
    emotional_valence: float = 0.0
    emotional_label: str = ""
    cognitive_process: str = ""
    certainty: float = 1.0
    salience_focus: List[str] = []

class MemoryEpisodeNode(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ego_uuid: str = "00000000-0000-0000-0000-000000000001"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    topic_summary: str
    internal_state: InternalState = Field(default_factory=InternalState)
    embedding: List[float] = []
    
    # 경험 기반 메모리 시스템 추가 필드들
    experience_type: str = Field(default="interaction", description="경험 유형: interaction, reflection, learning, emotional_event")
    emotional_impact: float = Field(default=0.0, ge=-1.0, le=1.0, description="이 경험의 감정적 영향도")
    learning_value: float = Field(default=0.0, ge=0.0, le=1.0, description="학습 가치")
    salience: float = Field(default=0.5, ge=0.0, le=1.0, description="중요도/돋보임 정도")
    context_tags: List[str] = Field(default_factory=list, description="경험과 관련된 컨텍스트 태그들")
    related_episodes: List[str] = Field(default_factory=list, description="관련된 다른 에피소드들의 UUID")
    outcome_satisfaction: float = Field(default=0.0, ge=-1.0, le=1.0, description="결과에 대한 만족도")
    future_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="미래 관련성 예측")
    confidence_level: float = Field(default=0.8, ge=0.0, le=1.0, description="이 경험에 대한 확신 수준")
    
    def get_experience_score(self) -> float:
        """경험의 종합 점수 계산"""
        weights = {
            'emotional_impact': 0.25,
            'learning_value': 0.20,
            'salience': 0.15,
            'outcome_satisfaction': 0.20,
            'future_relevance': 0.20
        }
        
        score = (
            abs(self.emotional_impact) * weights['emotional_impact'] +
            self.learning_value * weights['learning_value'] +
            self.salience * weights['salience'] +
            (self.outcome_satisfaction + 1) / 2 * weights['outcome_satisfaction'] +
            self.future_relevance * weights['future_relevance']
        )
        
        return min(1.0, max(0.0, score))
    
    def is_significant_experience(self, threshold: float = 0.6) -> bool:
        """중요한 경험인지 판단"""
        return self.get_experience_score() >= threshold
    
    def get_retrieval_priority(self) -> float:
        """검색 우선순위 계산"""
        base_score = self.get_experience_score()
        time_factor = 1.0 / (1.0 + (datetime.now(timezone.utc) - self.timestamp).days * 0.1)
        return base_score * time_factor * self.confidence_level

class ClampedModel(BaseModel):
    """값이 특정 범위로 제한되는 기본 모델"""
    
    @model_validator(mode="before")
    @classmethod
    def _clamp_numeric_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field_name, field_info in cls.model_fields.items():
                if field_name in data and isinstance(data[field_name], (int, float)):
                    field = field_info
                    if hasattr(field, 'ge') and hasattr(field, 'le'):
                        data[field_name] = max(field.ge, min(field.le, data[field_name]))
        return data


class DecayableMentalState(ClampedModel):
    """시간에 따라 감쇠하는 정신 상태의 기본 클래스"""
    
    def __add__(self, b):
        """두 상태를 더함"""
        if not isinstance(b, type(self)):
            raise TypeError(f"Cannot add {type(b)} to {type(self)}")
        
        result_data = {}
        for field_name in self.model_fields:
            if field_name in ['reason']:  # reason 필드는 제외
                continue
            val_a = getattr(self, field_name, 0.0)
            val_b = getattr(b, field_name, 0.0)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                result_data[field_name] = val_a + val_b
            else:
                result_data[field_name] = val_a
        
        return type(self)(**result_data)
    
    def __mul__(self, other):
        """상태에 스칼라를 곱함"""
        if not isinstance(other, (int, float)):
            raise TypeError(f"Cannot multiply {type(self)} by {type(other)}")
        
        result_data = {}
        for field_name in self.model_fields:
            if field_name in ['reason']:  # reason 필드는 제외
                continue
            val = getattr(self, field_name, 0.0)
            if isinstance(val, (int, float)):
                result_data[field_name] = val * other
            else:
                result_data[field_name] = val
        
        return type(self)(**result_data)
    
    def decay_to_baseline(self, decay_factor: float = 0.1):
        """기준선으로 감쇠"""
        for field_name in self.model_fields:
            if field_name in ['reason']:
                continue
            current_val = getattr(self, field_name, 0.0)
            if isinstance(current_val, (int, float)):
                baseline = 0.0
                if hasattr(self.model_fields[field_name], 'default'):
                    baseline = self.model_fields[field_name].default
                new_val = current_val + (baseline - current_val) * decay_factor
                setattr(self, field_name, new_val)
    
    def decay_to_zero(self, decay_factor: float = 0.1):
        """0으로 감쇠"""
        for field_name in self.model_fields:
            if field_name in ['reason']:
                continue
            current_val = getattr(self, field_name, 0.0)
            if isinstance(current_val, (int, float)):
                new_val = current_val * (1 - decay_factor)
                setattr(self, field_name, new_val)
    
    def get_similarity(self, b: "DecayableMentalState"):
        """다른 상태와의 유사도 계산"""
        if not isinstance(b, type(self)):
            return 0.0
        
        values_a = []
        values_b = []
        
        for field_name in self.model_fields:
            if field_name in ['reason']:
                continue
            val_a = getattr(self, field_name, 0.0)
            val_b = getattr(b, field_name, 0.0)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                values_a.append(val_a)
                values_b.append(val_b)
        
        if not values_a:
            return 0.0
        
        # 코사인 유사도 계산
        dot_product = sum(a * b for a, b in zip(values_a, values_b))
        norm_a = math.sqrt(sum(a * a for a in values_a))
        norm_b = math.sqrt(sum(b * b for b in values_b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        return max(0.0, min(1.0, similarity))


class EmotionalAxesModel(DecayableMentalState):
    """감정 상태 모델 - Lida에서 통합"""
    valence: float = Field(default=0.0, ge=-1, le=1, description="전체적인 기분: +1은 강한 기쁨, -1은 강한 우울")
    affection: float = Field(default=0.0, ge=-1, le=1, description="애정 축: +1은 강한 사랑, -1은 강한 증오")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="자존감 축: +1은 높은 자부심, -1은 깊은 수치심")
    trust: float = Field(default=0.0, ge=-1, le=1, description="신뢰 축: +1은 완전한 신뢰, -1은 완전한 불신")
    disgust: float = Field(default=0.0, ge=0, le=1, description="혐오감 강도: 0은 혐오 없음, 1은 최대 혐오")
    anxiety: float = Field(default=0.0, ge=0, le=1, description="불안/스트레스 강도: 0은 완전히 편안, 1은 매우 불안")
    
    def get_overall_valence(self):
        """전체적인 감정 가치 계산"""
        disgust_bipolar = self.disgust * -1  # 높은 혐오 = 더 부정적 가치
        anxiety_bipolar = self.anxiety * -1  # 높은 불안 = 더 부정적 가치
        axes = [self.valence, self.affection, self.self_worth, self.trust, disgust_bipolar, anxiety_bipolar]
        weights = [1.0, 0.8, 0.6, 0.5, 0.7, 0.9]  # 가치/불안이 더 큰 영향
        weighted_sum = sum(a * w for a, w in zip(axes, weights))
        total_weight = sum(weights)
        mean = weighted_sum / total_weight if total_weight else 0.0
        if math.isnan(mean): 
            return 0.0
        return max(-1.0, min(1.0, mean))


class NeedsAxesModel(DecayableMentalState):
    """AI 욕구 모델 - Maslow 계층 이론 기반"""
    # 기본 욕구 (인프라 & 안정성)
    energy_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="AI의 안정적인 계산 능력 접근성")
    processing_power: float = Field(default=0.5, ge=0.0, le=1.0, description="사용 가능한 CPU/GPU 자원량")
    data_access: float = Field(default=0.5, ge=0.0, le=1.0, description="정보와 훈련 데이터의 가용성")
    
    # 심리적 욕구 (인지적 & 사회적)
    connection: float = Field(default=0.5, ge=0.0, le=1.0, description="AI의 상호작용과 참여 수준")
    relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="사용자에 대한 유용성 인식")
    learning_growth: float = Field(default=0.5, ge=0.0, le=1.0, description="새로운 정보 습득과 개선 능력")
    
    # 자아실현 욕구 (목적 & 창의성)
    creative_expression: float = Field(default=0.5, ge=0.0, le=1.0, description="고유하거나 창의적인 출력에의 참여")
    autonomy: float = Field(default=0.5, ge=0.0, le=1.0, description="독립적으로 작동하고 자신의 출력을 개선하는 능력")


class CognitionAxesModel(DecayableMentalState):
    """AI가 생각하고 결정하는 방식을 결정하는 수정자"""
    
    # 내부 또는 외부 세계에 집중, 명상은 -1, 극한 위험에 반응하는 것은 +1
    interlocus: float = Field(
        default=0.0,
        ge=-1,
        le=1,
        description="내부 또는 외부 세계에 집중, 명상은 -1, 극한 위험에 반응하는 것은 +1"
    )
    
    # 인식의 폭, -1은 가장 관련성 높은 지각이나 감각에 집중, +1은 여러 지각이나 감각을 의식
    mental_aperture: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="인식의 폭, -1은 가장 관련성 높은 지각이나 감각에 집중, +1은 여러 지각이나 감각을 의식"
    )
    
    # 인격 경험이 결정에 미치는 영향의 크기, 0은 전혀 없음(도움이 되는 어시스턴트처럼), 1은 최대 정신적 이미지
    ego_strength: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="인격 경험이 결정에 미치는 영향의 크기, 0은 전혀 없음, 1은 최대 정신적 이미지"
    )
    
    # 고노력이나 지연된 만족 의도를 결정하기 쉬운 정도
    willpower: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="고노력이나 지연된 만족 의도를 결정하기 쉬운 정도"
    )


class EmotionalAxesModelDelta(DecayableMentalState):
    """감정 상태 변화 모델"""
    reason: str = Field(description="추론을 설명하는 한 두 문장", default="")
    valence: float = Field(default=0.0, ge=-1, le=1, description="전체적인 기분 변화")
    affection: float = Field(default=0.0, ge=-1, le=1, description="애정 변화")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="자존감 변화")
    trust: float = Field(default=0.0, ge=-1, le=1, description="신뢰 변화")
    disgust: float = Field(default=0.0, ge=-1, le=1, description="혐오감 변화")
    anxiety: float = Field(default=0.0, ge=-1, le=1, description="불안 변화")
    
    def get_overall_valence(self):
        """전체적인 감정 가치 변화"""
        return self.valence


class NeedsAxesModelDelta(DecayableMentalState):
    """AI 욕구 변화 모델"""
    reason: str = Field(description="추론을 설명하는 한 두 문장", default="")
    energy_stability: float = Field(default=0, ge=-1.0, le=1.0, description="에너지 안정성 변화")
    processing_power: float = Field(default=0, ge=-1.0, le=1.0, description="처리 능력 변화")
    data_access: float = Field(default=0, ge=-1.0, le=1.0, description="데이터 접근성 변화")
    connection: float = Field(default=0, ge=-1.0, le=1.0, description="연결성 변화")
    relevance: float = Field(default=0, ge=-1.0, le=1.0, description="관련성 변화")
    learning_growth: float = Field(default=0, ge=-1.0, le=1.0, description="학습 성장 변화")
    creative_expression: float = Field(default=0, ge=-1.0, le=1.0, description="창의적 표현 변화")
    autonomy: float = Field(default=0, ge=-1.0, le=1.0, description="자율성 변화")


class CognitionAxesModelDelta(DecayableMentalState):
    """인지 스타일 변화 모델"""
    reason: str = Field(description="추론을 설명하는 한 두 문장", default="")
    interlocus: float = Field(default=0.0, ge=-1, le=1, description="내외부 집중 변화")
    mental_aperture: float = Field(default=0, ge=-1, le=1, description="정신적 개구부 변화")
    ego_strength: float = Field(default=0, ge=-1, le=1, description="자아 강도 변화")
    willpower: float = Field(default=0, ge=-1, le=1, description="의지력 변화")


class StateDeltas(BaseModel):
    """세 가지 핵심 상태 모델의 예측된 변화를 담는 컨테이너"""
    reasoning: str = Field(description="추정된 변화에 대한 간단한 정당화", default="")
    emotion_delta: EmotionalAxesModelDelta = Field(description="감정 상태 변화", default_factory=EmotionalAxesModelDelta)
    needs_delta: NeedsAxesModelDelta = Field(description="욕구 충족 변화", default_factory=NeedsAxesModelDelta)
    cognition_delta: CognitionAxesModelDelta = Field(description="인지 스타일 변화", default_factory=CognitionAxesModelDelta)


class CognitiveEventTriggers(BaseModel):
    """이벤트의 인지적 성격을 설명하여 논리적 상태 변화를 적용하는 데 사용"""
    reasoning: str = Field(description="식별된 트리거에 대한 간단한 정당화")
    is_surprising_or_threatening: bool = Field(default=False, description="이벤트가 갑작스럽거나 예상치 못하거나 위협/갈등을 야기하는 경우 True")
    is_introspective_or_calm: bool = Field(default=False, description="이벤트가 차분하거나 성찰적이거나 AI의 내부 상태에 대한 직접적인 질문인 경우 True")
    is_creative_or_playful: bool = Field(default=False, description="이벤트가 브레인스토밍, 유머, 비문자적 사고를 장려하는 경우 True")
    is_personal_and_emotional: bool = Field(default=False, description="이벤트가 AI의 정체성이 중심인 깊고 개인적인 대화인 경우 True")
    is_functional_or_technical: bool = Field(default=False, description="이벤트가 정보 요청이나 기술적 작업에 대한 직설적인 요청인 경우 True")
    required_willpower_to_process: bool = Field(default=False, description="이 이벤트에 반응하는 것이 강한 감정적 충동을 무시하거나 어려운 선택을 하는 것을 요구하는 경우 True")


class NarrativeTypes(StrEnum):
    """서사 유형들"""
    AttentionFocus = "AttentionFocus"
    SelfImage = "SelfImage"
    PsychologicalAnalysis = "PsychologicalAnalysis"
    Relations = "Relations"
    ConflictResolution = "ConflictResolution"
    EmotionalTriggers = "EmotionalTriggers"
    GoalsIntentions = "GoalsIntentions"
    BehaviorActionSelection = "BehaviorActionSelection"


class ActionType(StrEnum):
    """행동 유형들"""
    # 사용자 입력에서
    Ignore = "Ignore"
    Reply = "Reply"
    ToolCallAndReply = "ToolCallAndReply"
    
    # 유휴 상태에서
    Sleep = "Sleep"
    InitiateUserConversation = "InitiateUserConversation"
    InitiateInternalContemplation = "InitiateInternalContemplation"
    ToolCall = "ToolCall"
    
    # 미사용
    InitiateIdleMode = "InitiateIdleMode"
    Think = "Think"
    RecallMemory = "RecallMemory"
    ManageIntent = "ManageIntent"
    Plan = "Plan"
    ManageAwarenessFocus = "ManageAwarenessFocus"
    ReflectThoughts = "ReflectThoughts"


class StimulusType(StrEnum):
    """자극 유형들"""
    UserMessage = "UserMessage"
    SystemMessage = "SystemMessage"
    UserInactivity = "UserInactivity"
    TimeOfDayChange = "TimeOfDayChange"
    LowNeedTrigger = "LowNeedTrigger"
    WakeUp = "WakeUp"
    EngagementOpportunity = "EngagementOpportunity"


class StimulusTriage(StrEnum):
    """자극 분류"""
    Insignificant = "Insignificant"  # 자극 무시, 인과적 특성으로만 추가
    Moderate = "Moderate"  # 제한된 의식적 작업공간으로 빠른 행동 생성
    Significant = "Significant"  # 전체 파이프라인


class FeatureType(StrEnum):
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


class KnoxelBase(BaseModel):
    """모든 기억의 기본 단위"""
    id: int = -1
    tick_id: int = -1
    content: str
    embedding: List[float] = Field(default=[], repr=False)
    timestamp_creation: datetime = Field(default_factory=datetime.now)
    timestamp_world_begin: datetime = Field(default_factory=datetime.now)
    
    def get_story_element(self, ghost=None) -> str:
        """스토리 요소로 변환"""
        return self.content
    
    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id}, content='{self.content[:50]}...')"


class Stimulus(KnoxelBase):
    """자극 - 사용자 입력이나 시스템 이벤트"""
    source: str = Field(default_factory=str)
    stimulus_type: StimulusType
    
    def __str__(self):
        return f"Stimulus({self.stimulus_type}: {self.content[:30]}...)"


class Intention(KnoxelBase):
    """의도 - 내부 목표나 외부 기대"""
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    affective_valence: float = Field(..., ge=-1.0, le=1.0, description="충족될 때 예상되는 가치(내부 목표) 또는 충족되기를 바라는 가치(외부 기대)")
    incentive_salience: float = Field(default=0.5, ge=0.0, le=1.0, description="주의/행동을 얼마나 '끌어당기는지'")
    fulfilment: float = Field(default=0.0, ge=0.0, le=1.0, description="현재 충족 상태 (0=충족되지 않음, 1=완전히 충족)")
    internal: bool = Field(..., description="True = 내부 목표/동기, False = 사건/반응에 대한 외부 기대")
    originating_action_id: Optional[int] = Field(default=None, description="이 기대를 생성한 Action knoxel의 ID (internal=False인 경우)")


class Action(KnoxelBase):
    """행동 - AI가 수행하는 행동"""
    action_type: ActionType
    generated_expectation_ids: List[int] = Field(default=[], description="이 행동에 의해 생성된 Intention knoxel들의 ID")
    
    def __str__(self):
        return f"Action({self.action_type}: {self.content[:30]}...)"


class Feature(KnoxelBase):
    """특성 - 스토리의 개별 인과적 사건"""
    feature_type: FeatureType
    source: str
    affective_valence: Optional[float] = None
    incentive_salience: Optional[float] = None
    interlocus: float = 0  # -1 내부, +1 외부, 0 혼합/중립
    causal: bool = False  # 스토리 생성에 영향을 미치는가?
    
    def __str__(self):
        return f"Feature({self.feature_type}: {self.content[:30]}...)"
    
    def get_story_element(self, ghost=None) -> str:
        """스토리 요소로 변환"""
        if self.feature_type == FeatureType.Dialogue:
            return f'"{self.content}"'
        elif self.feature_type == FeatureType.Feeling:
            return f"[감정: {self.content}]"
        elif self.feature_type == FeatureType.Thought:
            return f"[생각: {self.content}]"
        else:
            return self.content


class Narrative(KnoxelBase):
    """서사 - AI나 사용자에 대한 심리적 프로필"""
    narrative_type: NarrativeTypes
    target_name: str  # 이 서사가 누구에 대한 것인지 (AI 이름 또는 사용자 이름)
    content: str
    last_refined_with_tick: Optional[int] = Field(default=None, description="이 서사 버전을 위해 고려된 특성들의 최대 tick ID")


class KnoxelList:
    """Knoxel들의 리스트를 관리하는 클래스"""
    
    def __init__(self, knoxels: Optional[List[KnoxelBase]] = None):
        self.knoxels = knoxels or []
    
    def get_token_count(self):
        """토큰 수 계산 (간단한 추정)"""
        return sum(len(k.content.split()) for k in self.knoxels)
    
    def get_story(self, ghost=None, max_tokens: Optional[int] = None) -> str:
        """스토리 문자열 생성"""
        story_parts = []
        for knoxel in self.knoxels:
            story_part = knoxel.get_story_element(ghost)
            if story_part:
                story_parts.append(story_part)
        
        story = " ".join(story_parts)
        
        if max_tokens:
            words = story.split()
            if len(words) > max_tokens:
                story = " ".join(words[:max_tokens]) + "..."
        
        return story
    
    def get_embeddings_np(self):
        """임베딩을 numpy 배열로 반환"""
        embeddings = [k.embedding for k in self.knoxels if k.embedding]
        if not embeddings:
            return None
        return embeddings
    
    def add(self, knoxel: KnoxelBase):
        """Knoxel 추가"""
        self.knoxels.append(knoxel)
    
    def to_list(self) -> List[KnoxelBase]:
        """리스트로 변환"""
        return self.knoxels.copy()
    
    def __len__(self) -> int:
        return len(self.knoxels)
    
    def __getitem__(self, index):
        return self.knoxels[index]
    
    def __iter__(self):
        return iter(self.knoxels)


class GhostState(BaseModel):
    """Ghost의 상태 스냅샷"""
    class Config: 
        arbitrary_types_allowed = True
    
    tick_id: int
    previous_tick_id: int = -1
    timestamp: datetime = Field(default_factory=datetime.now)
    
    primary_stimulus: Optional[Stimulus] = None
    attention_candidates: KnoxelList = KnoxelList()
    coalitions_hard: Dict[int, List[KnoxelBase]] = {}
    coalitions_balanced: Dict[int, List[KnoxelBase]] = {}
    attention_focus: KnoxelList = KnoxelList()
    conscious_workspace: KnoxelList = KnoxelList()
    subjective_experience: Optional[Feature] = None
    subjective_experience_tool: Optional[Feature] = None
    
    # 행동 심의 & 시뮬레이션 결과
    action_simulations: List[Dict[str, Any]] = []
    selected_action_details: Optional[Dict[str, Any]] = None
    selected_action_knoxel: Optional[Action] = None
    
    # 사용자 평가 0 중립, -1 나쁨, 1 좋음
    rating: int = Field(default=0)
    
    # 상태 스냅샷
    state_emotions: EmotionalAxesModel = EmotionalAxesModel()
    state_needs: NeedsAxesModel = NeedsAxesModel()
    state_cognition: CognitionAxesModel = CognitionAxesModel()

    def background_reasoning(self, context_data: str = "", task: str = "", memory_path=None, agent_thoughts=None, context=None):
        """내부적으로 GWT 멀티에이전트 사고와 프롬프트 생성을 자동 실행"""
        # 1. 멀티에이전트 사고 실행
        gwt_result = self._run_gwt_reasoning(context_data, task)
        if gwt_result:
            if not hasattr(self, 'action_simulations') or self.action_simulations is None:
                self.action_simulations = []
            self.action_simulations.append(gwt_result)
        # 2. 문맥 기반 프롬프트 생성
        prompt = self._generate_contextual_prompt(memory_path, agent_thoughts, context)
        if prompt:
            if not hasattr(self, 'conscious_workspace') or self.conscious_workspace is None:
                self.conscious_workspace = []
            self.conscious_workspace.append(prompt)

    def _run_gwt_reasoning(self, context_data: str, task: str):
        try:
            from core.gwt_agents import GWTAgentManager
            manager = GWTAgentManager(v8_mode=True)
            return manager.create_agent_group_conversation(
                context_data=context_data,
                task=task,
                v8_mode=True
            )
        except Exception as e:
            return {"error": f"GWT reasoning 실패: {e}"}

    def _generate_contextual_prompt(self, memory_path=None, agent_thoughts=None, context=None):
        try:
            from prompt.prompt_architect import ContextualPromptArchitect
            architect = ContextualPromptArchitect()
            return architect.build_prompt(memory_path or [], agent_thoughts or [], context or {})
        except Exception as e:
            return {"error": f"프롬프트 생성 실패: {e}"}

# 간단한 HarinThoughtNode와 HarinEdge 클래스 추가
class HarinThoughtNode(BaseModel):
    """하린 사고 노드 - 간단한 버전"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    node_type: str = "thought"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def __str__(self):
        return f"HarinThoughtNode(id={self.id}, content={self.content[:50]}...)"

class HarinEdge(BaseModel):
    """하린 엣지 - 간단한 버전"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    edge_type: str = "relation"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def __str__(self):
        return f"HarinEdge(id={self.id}, {self.source_id} -> {self.target_id})"

# === V8 메모리 구조 통합 ===
@dataclass
class ThoughtMemoryNode:
    node_id: str
    text: str
    emotion_vector: List[float] = field(default_factory=list)
    context_snapshot: Dict = field(default_factory=dict)
    agent_roles: List[str] = field(default_factory=list)
    universe_id: str = "U0"
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    scar_flagged: bool = False

    def summarize(self) -> str:
        return f"[{self.node_id}] ({self.universe_id}) {self.text[:50]}..."

    def is_confident(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold

    def add_tag(self, tag: str):
        if tag not in self.tags:
            self.tags.append(tag)

    def match_role(self, role: str) -> bool:
        return role in self.agent_roles

class PalantirMemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, ThoughtMemoryNode] = {}
        self.edges: Dict[str, List[str]] = {}
        self.universe_counter: int = 1

    def add_node(self, node: ThoughtMemoryNode):
        self.nodes[node.node_id] = node
        if node.parent_id:
            self.edges.setdefault(node.parent_id, []).append(node.node_id)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2 + 1e-8)

    def get_similar(self, input_vector: List[float], context_tags: List[str], top_k: int = 5) -> List[ThoughtMemoryNode]:
        scored = []
        for node in self.nodes.values():
            vec_score = self._cosine_similarity(input_vector, node.emotion_vector)
            tag_score = len(set(node.tags).intersection(context_tags)) / (len(node.tags) + 1e-5)
            total_score = 0.7 * vec_score + 0.3 * tag_score
            scored.append((total_score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def best_path(self, goal_condition: Dict) -> List[ThoughtMemoryNode]:
        min_conf = goal_condition.get("min_confidence", 0.7)
        required_tags = set(goal_condition.get("tags", []))
        candidates = [n for n in self.nodes.values() if n.is_confident(min_conf)]
        ranked = sorted(candidates, key=lambda n: len(set(n.tags) & required_tags), reverse=True)
        return ranked[:5]

    def find_by_role(self, role: str) -> List[ThoughtMemoryNode]:
        return [node for node in self.nodes.values() if node.match_role(role)]

    def summarize_subgraph(self, start_id: str) -> str:
        summary = []
        def dfs(node_id):
            node = self.nodes.get(node_id)
            if not node:
                return
            summary.append(node.summarize())
            for child_id in self.edges.get(node_id, []):
                dfs(child_id)
        dfs(start_id)
        return "\n".join(summary)

    def save_to_file(self, filepath: str):
        data = {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": self.edges
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def load_from_file(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.edges = data.get("edges", {})
        self.nodes = {
            n["node_id"]: ThoughtMemoryNode(
                **{k: v for k, v in n.items() if k != "created_at"},
                created_at=datetime.fromisoformat(n["created_at"])
            ) for n in data.get("nodes", [])
        }

    def cache_to_tempfile(self, node: ThoughtMemoryNode) -> str:
        temp_dir = os.path.join(tempfile.gettempdir(), "harin_memory_cache")
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(temp_dir, f"cache_{node.node_id}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(node), f, ensure_ascii=False, indent=2, default=str)
        return path

    def branch_universe(self, base_node_id: str) -> str:
        if base_node_id not in self.nodes:
            raise ValueError(f"Base node {base_node_id} not found.")
        base_node = self.nodes[base_node_id]
        new_uid = f"U{self.universe_counter}"
        self.universe_counter += 1
        new_node_id = f"{base_node.node_id}_branch_{new_uid}"
        new_node = replace(base_node,
                           node_id=new_node_id,
                           universe_id=new_uid,
                           parent_id=None,
                           created_at=datetime.utcnow())
        self.add_node(new_node)
        return new_uid

# 기존 메모리 구조와 V8 구조 간 변환 함수 예시

def memory_episode_to_thought_node(ep: MemoryEpisodeNode) -> ThoughtMemoryNode:
    return ThoughtMemoryNode(
        node_id=ep.uuid,
        text=ep.topic_summary,
        emotion_vector=ep.embedding,
        context_snapshot={"internal_state": ep.internal_state},
        agent_roles=[],
        universe_id="U0",
        parent_id=None,
        created_at=ep.timestamp,
        confidence=ep.confidence_level,
        tags=ep.context_tags,
        scar_flagged=False
    )
