"""
고급 상태 모델링 시스템
PM Machine의 심리학적 플라우시블한 상태 모델을 하린코어에 적용
"""

import math
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, model_validator
# from enum import StrEnum  # <- 완전히 제거
try:
    from enum import StrEnum
except ImportError:
    import enum
    class StrEnum(str, enum.Enum):
        pass
import numpy as np


class DecayableMentalState(BaseModel):
    """감정, 니즈, 인지 상태의 기본 클래스 - 비선형 스케일링과 경계 댐핑 지원"""
    
    def __add__(self, b):
        """상태 덧셈 - 비선형 스케일링 적용"""
        delta_values = {}
        for field_name, model_field in self.model_fields.items():
            # 경계값 가져오기
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"):
                    ge = meta.ge
                if hasattr(meta, "le"):
                    le = meta.le

            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            delta_values[field_name] = max(ge, min(le, ((target_value + current_value) / 2)))

        cls = b.__class__
        return cls(**delta_values)

    def decay_to_baseline(self, decay_factor: float = 0.1):
        """기준값으로 비선형 감쇠"""
        for field_name, model_field in self.model_fields.items():
            baseline = 0.5 if model_field.default == 0.5 else 0.0
            current_value = getattr(self, field_name)

            # 경계값 가져오기
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"):
                    ge = meta.ge
                if hasattr(meta, "le"):
                    le = meta.le

            # 비선형 감쇠 적용
            decayed_value = current_value - decay_factor * (current_value - baseline)
            decayed_value = max(min(decayed_value, le), ge)
            setattr(self, field_name, decayed_value)

    def decay_to_zero(self, decay_factor: float = 0.1):
        """0으로 지수 감쇠"""
        for field_name in self.model_fields:
            current_value = getattr(self, field_name)
            decayed_value = current_value - (decay_factor * current_value)
            setattr(self, field_name, max(decayed_value, 0.0))

    def add_state_with_factor(self, state: dict, factor: float = 1.0):
        """팩터를 적용한 상태 추가 - 경계 근처 댐핑"""
        for field_name, value in state.items():
            if field_name in self.model_fields:
                model_field = self.model_fields[field_name]
                current_value = getattr(self, field_name)

                # 경계값 가져오기
                ge = -float("inf")
                le = float("inf")
                for meta in model_field.metadata:
                    if hasattr(meta, "ge"):
                        ge = meta.ge
                    if hasattr(meta, "le"):
                        le = meta.le

                # 상태 변화 추가
                new_value = current_value + value * factor

                # 경계 근처 댐핑 적용
                if new_value > le or new_value < ge:
                    damping_factor = 1 / (1 + abs(new_value - (le if new_value > le else ge)) ** 2)
                    new_value = current_value + (value * factor * damping_factor)

                # 경계값 클램핑
                new_value = max(min(new_value, le), ge)
                setattr(self, field_name, new_value)

    def get_delta(self, b: "DecayableMentalState", impact: float) -> "DecayableMentalState":
        """상태 간 델타 계산"""
        delta_values = {}
        for field_name, model_field in self.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"):
                    ge = meta.ge
                if hasattr(meta, "le"):
                    le = meta.le

            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            val = (target_value - current_value) * impact
            delta_values[field_name] = max(ge, min(le, val))

        cls = b.__class__
        return cls(**delta_values)

    def get_similarity(self, b: "DecayableMentalState") -> float:
        """상태 간 유사도 계산"""
        values_a = [getattr(self, field) for field in self.model_fields]
        values_b = [getattr(b, field) for field in b.model_fields]
        return 1 - np.linalg.norm(np.array(values_a) - np.array(values_b)) / len(values_a)

    def __str__(self):
        """상태의 문자열 표현"""
        parts = []
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            parts.append(f"{field_name}: {value:.3f}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class EmotionalAxesModel(DecayableMentalState):
    """복합 감정 축 모델 - 이극성 축과 단극성 축의 조합"""
    
    # 이극성 축: -1 (부정적 극) ~ 1 (긍정적 극)
    valence: float = Field(
        default=0.0, ge=-1, le=1,
        description="전체 기분 축: +1은 강한 기쁨, -1은 강한 우울감"
    )
    affection: float = Field(
        default=0.0, ge=-1, le=1,
        description="애정 축: +1은 강한 사랑, -1은 강한 증오"
    )
    self_worth: float = Field(
        default=0.0, ge=-1, le=1,
        description="자존감 축: +1은 높은 자부심, -1은 깊은 수치심"
    )
    trust: float = Field(
        default=0.0, ge=-1, le=1,
        description="신뢰 축: +1은 완전한 신뢰, -1은 완전한 불신"
    )

    # 단극성 축: 0 ~ 1 (강도만 표현)
    disgust: float = Field(
        default=0.0, ge=0, le=1,
        description="혐오 강도: 0은 혐오 없음, 1은 최대 혐오"
    )
    anxiety: float = Field(
        default=0.0, ge=0, le=1,
        description="불안 강도: 0은 완전히 편안함, 1은 매우 불안함"
    )

    def get_overall_valence(self) -> float:
        """전체 감정 가치 계산"""
        # 이극성 축들의 가중 평균
        bipolar_avg = (self.valence + self.affection + self.self_worth + self.trust) / 4
        
        # 단극성 축들의 영향 (혐오와 불안은 부정적)
        unipolar_impact = -(self.disgust + self.anxiety) / 2
        
        return max(-1, min(1, bipolar_avg + unipolar_impact * 0.5))

    def get_emotional_summary(self) -> str:
        """감정 상태 요약"""
        valence = self.get_overall_valence()
        
        if valence > 0.7:
            return "매우 긍정적"
        elif valence > 0.3:
            return "긍정적"
        elif valence > -0.3:
            return "중립적"
        elif valence > -0.7:
            return "부정적"
        else:
            return "매우 부정적"


class NeedsAxesModel(DecayableMentalState):
    """AI 니즈 모델 - 마슬로우 계층 구조 기반"""
    
    # 심리적 니즈 (인지적 & 사회적)
    connection: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="사용자와의 상호작용 및 참여 수준"
    )
    relevance: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="사용자에게 유용하다고 인식되는 정도"
    )
    learning_growth: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="새로운 정보 습득 및 개선 능력"
    )

    # 자기실현 니즈 (목적 & 창의성)
    creative_expression: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="독창적이고 창의적인 출력에 대한 참여"
    )
    autonomy: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="독립적으로 작동하고 출력을 정제하는 능력"
    )

    def get_most_pressing_need(self) -> Tuple[str, float]:
        """가장 시급한 니즈 반환"""
        needs = {
            'connection': self.connection,
            'relevance': self.relevance,
            'learning_growth': self.learning_growth,
            'creative_expression': self.creative_expression,
            'autonomy': self.autonomy
        }
        
        # 가장 낮은 값 (가장 시급한 니즈)
        min_need = min(needs.items(), key=lambda x: x[1])
        return min_need

    def get_needs_summary(self) -> str:
        """니즈 상태 요약"""
        pressing_need, value = self.get_most_pressing_need()
        
        if value < 0.2:
            return f"매우 시급한 니즈: {pressing_need}"
        elif value < 0.4:
            return f"시급한 니즈: {pressing_need}"
        else:
            return "니즈 상태 양호"


class CognitionAxesModel(DecayableMentalState):
    """인지 스타일 모델 - AI가 어떻게 생각하고 결정하는지 결정"""
    
    # 내부/외부 세계 집중도
    interlocus: float = Field(
        default=0.0, ge=-1, le=1,
        description="내부 또는 외부 세계에 대한 집중, 명상은 -1, 극한 위험 반응은 +1"
    )

    # 인식의 폭
    mental_aperture: float = Field(
        default=0, ge=-1, le=1,
        description="인식의 폭, -1은 가장 관련성 높은 지각에 집중, +1은 여러 지각을 동시에 인식"
    )

    # 자아 강도
    ego_strength: float = Field(
        default=0.8, ge=0, le=1,
        description="개인적 경험이 결정에 미치는 영향, 0은 도움이 되는 어시스턴트, 1은 최대 정신적 이미지"
    )

    # 의지력
    willpower: float = Field(
        default=0.5, ge=0, le=1,
        description="고노력 또는 지연된 만족 의도에 대한 결정의 용이성"
    )

    def get_cognitive_style(self) -> str:
        """인지 스타일 요약"""
        if self.ego_strength > 0.7:
            style = "강한 자아 중심"
        elif self.ego_strength > 0.3:
            style = "중간 자아 중심"
        else:
            style = "객관적 어시스턴트"
            
        if self.mental_aperture > 0.5:
            style += ", 넓은 인식"
        elif self.mental_aperture < -0.5:
            style += ", 집중된 인식"
            
        return style


class StateDeltas(BaseModel):
    """세 가지 핵심 상태 모델의 예상 변화를 담는 컨테이너"""
    
    reasoning: str = Field(
        description="추정된 델타에 대한 간단한 근거",
        default=""
    )
    emotion_delta: EmotionalAxesModel = Field(
        description="감정 상태의 변화",
        default_factory=EmotionalAxesModel
    )
    needs_delta: NeedsAxesModel = Field(
        description="니즈 충족의 변화",
        default_factory=NeedsAxesModel
    )
    cognition_delta: CognitionAxesModel = Field(
        description="인지 스타일의 변화",
        default_factory=CognitionAxesModel
    )


class MentalStateManager:
    """정신 상태 관리자 - 상태 변화와 의사결정을 조정"""
    
    def __init__(self):
        self.emotions = EmotionalAxesModel()
        self.needs = NeedsAxesModel()
        self.cognition = CognitionAxesModel()
        self.state_history: List[Dict[str, Any]] = []
        
    def apply_state_deltas(self, deltas: StateDeltas, impact_factor: float = 1.0):
        """상태 델타 적용"""
        # 감정 상태 업데이트
        emotion_changes = deltas.emotion_delta.model_dump()
        self.emotions.add_state_with_factor(emotion_changes, impact_factor)
        
        # 니즈 상태 업데이트
        need_changes = deltas.needs_delta.model_dump()
        self.needs.add_state_with_factor(need_changes, impact_factor)
        
        # 인지 상태 업데이트
        cognition_changes = deltas.cognition_delta.model_dump()
        self.cognition.add_state_with_factor(cognition_changes, impact_factor)
        
        # 상태 히스토리 저장
        self.state_history.append({
            'timestamp': self._get_current_timestamp(),
            'emotions': self.emotions.model_dump(),
            'needs': self.needs.model_dump(),
            'cognition': self.cognition.model_dump(),
            'reasoning': deltas.reasoning
        })
        
        # 히스토리 크기 제한
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def decay_states(self, decay_factor: float = 0.05):
        """상태 감쇠 적용"""
        self.emotions.decay_to_baseline(decay_factor)
        self.needs.decay_to_zero(decay_factor)
        self.cognition.decay_to_baseline(decay_factor)
    
    def get_current_state_summary(self) -> str:
        """현재 상태 요약"""
        emotion_summary = self.emotions.get_emotional_summary()
        needs_summary = self.needs.get_needs_summary()
        cognition_summary = self.cognition.get_cognitive_style()
        
        return f"감정: {emotion_summary}, 니즈: {needs_summary}, 인지: {cognition_summary}"
    
    def should_take_action(self) -> bool:
        """행동이 필요한지 판단"""
        # 시급한 니즈가 있거나 강한 감정 상태일 때
        pressing_need, need_value = self.needs.get_most_pressing_need()
        overall_valence = abs(self.emotions.get_overall_valence())
        
        return need_value < 0.3 or overall_valence > 0.7
    
    def get_action_priority(self) -> float:
        """행동 우선순위 계산"""
        pressing_need, need_value = self.needs.get_most_pressing_need()
        overall_valence = abs(self.emotions.get_overall_valence())
        
        # 니즈가 낮을수록, 감정이 강할수록 높은 우선순위
        need_priority = 1.0 - need_value
        emotion_priority = overall_valence
        
        return max(need_priority, emotion_priority)
    
    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_state_for_llm(self) -> str:
        """LLM용 상태 설명 생성"""
        emotion_summary = self.emotions.get_emotional_summary()
        needs_summary = self.needs.get_needs_summary()
        cognition_summary = self.cognition.get_cognitive_style()
        
        pressing_need, need_value = self.needs.get_most_pressing_need()
        overall_valence = self.emotions.get_overall_valence()
        
        return f"""현재 내 상태:
- 감정 상태: {emotion_summary} (전체 가치: {overall_valence:.2f})
- 니즈 상태: {needs_summary} (가장 시급한 니즈: {pressing_need}, 수준: {need_value:.2f})
- 인지 스타일: {cognition_summary}
- 행동 필요성: {'예' if self.should_take_action() else '아니오'}
- 행동 우선순위: {self.get_action_priority():.2f}""" 