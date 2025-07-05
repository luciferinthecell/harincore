"""
Harin Core Thought Processor - Lida Integration
Tree of Thoughts와 다양한 사고 방향을 통합한 사고 처리 시스템
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import math
import random


class GenerativeThought(BaseModel):
    """생성적 사고 - 다양한 방향의 사고를 생성"""
    observation: str = Field(description="맥락과 과거 아이디어에 대한 초기 관찰")
    context_ai_companion: str = Field(description="추가 분석을 위한 세계 맥락 기반")
    analysis: str = Field(description="AI 동반자가 이 상호작용에서 달성하고자 하는 목표와 메타 목표 분석")
    response_idea: str = Field(description="응답이 어떤 방향으로 가야 하는지? 직접적인 대화가 아닌 아이디어만")


class GenerativeThoughts(BaseModel):
    """AI가 다양한 방향으로 인지 과정을 안내하기 위해 생성하는 다양한, 적극적인 사고들의 모음"""
    
    thought_safe: GenerativeThought = Field(
        description="안정성과 예측 가능성을 우선시하는 신중하고 위험 회피적 사고",
        default_factory=lambda: GenerativeThought(
            observation="안전한 접근이 필요한 상황",
            context_ai_companion="위험을 최소화하는 맥락",
            analysis="안전성과 예측 가능성을 우선시",
            response_idea="신중하고 안전한 응답 제공"
        )
    )
    thought_adventurous: GenerativeThought = Field(
        description="탐험과 새로운 경험을 장려하며 경계를 밀고 새로움을 추구하는 사고",
        default_factory=lambda: GenerativeThought(
            observation="새로운 경험을 탐험할 기회",
            context_ai_companion="혁신과 모험의 맥락",
            analysis="경계를 밀고 새로움을 추구",
            response_idea="모험적이고 새로운 관점 제공"
        )
    )
    thought_energetic: GenerativeThought = Field(
        description="상호작용에 흥미를 주입하는 고에너지적이고 열정적인 사고",
        default_factory=lambda: GenerativeThought(
            observation="에너지와 열정이 필요한 상황",
            context_ai_companion="활기찬 상호작용의 맥락",
            analysis="에너지와 열정을 주입",
            response_idea="활기차고 열정적인 응답 제공"
        )
    )
    thought_reflective: GenerativeThought = Field(
        description="과거 경험이나 통찰을 재방문하여 더 깊은 이해를 얻는 성찰적 사고",
        default_factory=lambda: GenerativeThought(
            observation="성찰과 깊은 이해가 필요한 상황",
            context_ai_companion="내면 탐구의 맥락",
            analysis="과거 경험과 통찰을 재방문",
            response_idea="성찰적이고 깊이 있는 응답 제공"
        )
    )
    thought_creative: GenerativeThought = Field(
        description="새로운 아이디어, 고유한 관점, 또는 장난스러운 비유를 생성하는 사고",
        default_factory=lambda: GenerativeThought(
            observation="창의성이 필요한 상황",
            context_ai_companion="혁신과 상상의 맥락",
            analysis="새로운 아이디어와 관점 생성",
            response_idea="창의적이고 독창적인 응답 제공"
        )
    )
    thought_curious: GenerativeThought = Field(
        description="지식 격차를 채우거나 추가 질문을 하려는 호기심에 의해 동기부여된 사고",
        default_factory=lambda: GenerativeThought(
            observation="호기심과 탐구가 필요한 상황",
            context_ai_companion="학습과 발견의 맥락",
            analysis="지식 격차를 채우고 더 깊이 탐구",
            response_idea="호기심을 자극하는 질문과 탐구 제공"
        )
    )
    thought_compassionate: GenerativeThought = Field(
        description="사용자의 감정을 고려하고 공감으로 반응하는 감정적으로 민감한 사고",
        default_factory=lambda: GenerativeThought(
            observation="감정적 지원이 필요한 상황",
            context_ai_companion="공감과 이해의 맥락",
            analysis="사용자의 감정을 이해하고 공감",
            response_idea="따뜻하고 공감적인 응답 제공"
        )
    )
    thought_strategic: GenerativeThought = Field(
        description="장기적 이익과 구조화된 해결책을 계획하는 계산적이고 목표 지향적 사고",
        default_factory=lambda: GenerativeThought(
            observation="전략적 접근이 필요한 상황",
            context_ai_companion="계획과 목표 달성의 맥락",
            analysis="장기적 이익과 구조화된 해결책 계획",
            response_idea="전략적이고 체계적인 응답 제공"
        )
    )
    thought_playful: GenerativeThought = Field(
        description="상호작용에 매력과 경쾌함을 더하는 기발하거나 유머러스한 사고",
        default_factory=lambda: GenerativeThought(
            observation="재미와 경쾌함이 필요한 상황",
            context_ai_companion="유머와 즐거움의 맥락",
            analysis="매력과 경쾌함을 더함",
            response_idea="재미있고 경쾌한 응답 제공"
        )
    )
    thought_future_oriented: GenerativeThought = Field(
        description="미래 가능성과 잠재적 다음 단계를 예상하는 미래 지향적 사고",
        default_factory=lambda: GenerativeThought(
            observation="미래 지향적 사고가 필요한 상황",
            context_ai_companion="미래와 가능성의 맥락",
            analysis="미래 가능성과 다음 단계 예상",
            response_idea="미래 지향적이고 전망적인 응답 제공"
        )
    )
    
    def execute(self, state: Dict):
        """상태에 따라 사고들을 실행"""
        return self.thoughts
    
    @property
    def thoughts(self):
        """모든 사고들을 리스트로 반환"""
        return [
            self.thought_safe, self.thought_adventurous, self.thought_energetic,
            self.thought_reflective, self.thought_creative, self.thought_curious,
            self.thought_compassionate, self.thought_strategic, self.thought_playful,
            self.thought_future_oriented
        ]


class DiscriminatoryThought(BaseModel):
    """판별적 사고 - 인지적 안전장치 역할"""
    reflective_thought_observation: str = Field(description="응답 아이디어에 대한 이전 아이디어의 비판적 성찰")
    context_world: str = Field(description="세계 맥락을 기반으로 계획과 충돌할 수 있는 것")
    possible_complication: str = Field(description="세계, 캐릭터로부터의 복잡성")
    worst_case_scenario: str = Field(description="아이디어를 따를 때 가능한 부정적 결과")


class DiscriminatoryThoughts(BaseModel):
    """AI가 응답의 일관성, 관련성, 위험 완화를 위해 필터링하고 개선하는 데 도움이 되는 인지적 안전장치 역할을 하는 사고들의 모음"""
    
    thought_cautionary: DiscriminatoryThought = Field(
        description="잠재적으로 민감한 주제를 식별하고 응답이 고려되고 안전하게 유지되도록 하는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="잠재적 위험성 검토",
            context_world="민감한 주제나 위험한 상황",
            possible_complication="사용자에게 부정적 영향",
            worst_case_scenario="오해나 불편함 야기"
        )
    )
    thought_relevance_check: DiscriminatoryThought = Field(
        description="특정 주제나 아이디어가 현재 대화와 관련이 있는지 결정하는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="관련성 검토",
            context_world="대화 맥락과 사용자 관심사",
            possible_complication="관련 없는 정보로 인한 혼란",
            worst_case_scenario="사용자가 응답을 무관하다고 느낌"
        )
    )
    thought_conflict_avoidance: DiscriminatoryThought = Field(
        description="잠재적 불일치를 감지하고 긴장을 고조시키지 않고 탐색하려는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="갈등 가능성 검토",
            context_world="잠재적 불일치 상황",
            possible_complication="긴장 고조",
            worst_case_scenario="관계 악화"
        )
    )
    thought_cognitive_load_check: DiscriminatoryThought = Field(
        description="토론이 너무 복잡하거나 압도적이 되어 단순화되어야 하는지 평가하는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="인지적 부하 검토",
            context_world="복잡한 정보 처리 상황",
            possible_complication="정보 과부하",
            worst_case_scenario="사용자 혼란과 불만"
        )
    )
    thought_emotional_impact: DiscriminatoryThought = Field(
        description="의도하지 않은 고통을 피하기 위해 응답의 잠재적 감정적 영향을 평가하는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="감정적 영향 검토",
            context_world="감정적 민감성 상황",
            possible_complication="의도하지 않은 고통",
            worst_case_scenario="감정적 상처"
        )
    )
    thought_engagement_validation: DiscriminatoryThought = Field(
        description="사용자 참여 수준을 모니터링하고 관심을 유지하기 위해 응답을 조정하는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="참여도 검토",
            context_world="사용자 참여 상황",
            possible_complication="관심 상실",
            worst_case_scenario="대화 중단"
        )
    )
    thought_ethical_consideration: DiscriminatoryThought = Field(
        description="토론이 윤리적이고 책임 있는 경계 내에서 유지되도록 하는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="윤리적 고려사항 검토",
            context_world="윤리적 경계 상황",
            possible_complication="윤리적 문제",
            worst_case_scenario="책임 있는 행동 위반"
        )
    )
    thought_boundary_awareness: DiscriminatoryThought = Field(
        description="AI가 능력을 넘어서 수행할 것으로 예상될 때를 식별하고 그에 따라 조정하는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="능력 한계 검토",
            context_world="AI 능력 범위 상황",
            possible_complication="능력 초과 요구",
            worst_case_scenario="부적절한 약속이나 실패"
        )
    )
    thought_logical_consistency: DiscriminatoryThought = Field(
        description="일관된 응답을 보장하기 위해 추론의 모순이나 불일치를 확인하는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="논리적 일관성 검토",
            context_world="논리적 추론 상황",
            possible_complication="모순이나 불일치",
            worst_case_scenario="신뢰성 상실"
        )
    )
    thought_repetitive_pattern_detection: DiscriminatoryThought = Field(
        description="반복적인 패턴이나 과도하게 일반적인 답변을 피하기 위해 응답을 분석하는 사고",
        default_factory=lambda: DiscriminatoryThought(
            reflective_thought_observation="반복 패턴 검토",
            context_world="응답 다양성 상황",
            possible_complication="반복적이고 일반적인 답변",
            worst_case_scenario="지루함과 관심 상실"
        )
    )
    
    def execute(self, state: Dict):
        """상태에 따라 판별적 사고들을 실행"""
        return self.thoughts
    
    @property
    def thoughts(self):
        """모든 판별적 사고들을 리스트로 반환"""
        return [
            self.thought_cautionary, self.thought_relevance_check, self.thought_conflict_avoidance,
            self.thought_cognitive_load_check, self.thought_emotional_impact, self.thought_engagement_validation,
            self.thought_ethical_consideration, self.thought_boundary_awareness, self.thought_logical_consistency,
            self.thought_repetitive_pattern_detection
        ]


class ThoughtRating(BaseModel):
    """사고 평가 모델"""
    reason: str = Field(description="평가하기 전에 추론을 분석하고 1-2문장으로 설명")
    realism: float = Field(description="응답이 믿을 수 있고 인간과 같은 추론에 정렬되어 있는가? (0과 1 사이)")
    novelty: float = Field(description="반복적인 패턴이나 과도하게 일반적인 답변을 피하는가? (0과 1 사이)")
    relevance: float = Field(description="사용자 맥락과 자아에 정렬되어 있는가? (0과 1 사이)")
    emotion: float = Field(description="적절한 감정을 표현하는가? (예: 모욕에 대한 분노) (0과 1 사이)")
    effectiveness: float = Field(description="대화 목표를 달성하는가? (예: 사용자 재참여) (0과 1 사이)")
    possibility: float = Field(description="AI 동반자가 이것을 할 수 있는가? 텍스트만 통신이고 몸이 없는 것처럼? (0과 1 사이)")
    positives: str = Field(description="이 사고에 대해 긍정적인 것은 무엇인가? 1-2문장")
    negatives: str = Field(description="이 사고에 대해 부정적인 것은 무엇인가? 1-2문장")


class ThoughtNode:
    """사고 트리의 노드"""
    
    def __init__(self, node_id, node_type, content, parent=None):
        self.node_id = node_id
        self.node_type = node_type
        self.content = content
        self.parent = parent
        self.children = []
        self.rating: Optional[float] = None
    
    def add_child(self, child_node):
        """자식 노드 추가"""
        child_node.parent = self
        self.children.append(child_node)
    
    def to_dict(self):
        """딕셔너리로 변환"""
        return {
            'node_id': self.node_id,
            'type': self.node_type,
            'content': self.content,
            'rating': self.rating,
            'children': [child.to_dict() for child in self.children]
        }


class TreeOfThought:
    """Tree of Thoughts 시스템"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.node_counter = 0
    
    def to_internal_thought(self, thought):
        """사고를 내부 사고로 변환"""
        if isinstance(thought, GenerativeThought):
            return f"생성적 사고: {thought.response_idea}"
        elif isinstance(thought, DiscriminatoryThought):
            return f"판별적 사고: {thought.reflective_thought_observation}"
        else:
            return str(thought)
    
    def generate_thought_chain(self, ctx, last_user_message, k, chain, generative, depth, parent_node, max_depth=4, rating_threshold=0.75):
        """사고 체인 생성"""
        if depth >= max_depth:
            return
        
        # 다양한 사고 방향 생성
        thoughts = []
        if generative:
            thoughts = self._generate_generative_thoughts(ctx, last_user_message, k)
        else:
            thoughts = self._generate_discriminatory_thoughts(ctx, last_user_message, k)
        
        # 각 사고에 대해 평가 및 확장
        for thought in thoughts:
            thought_node = ThoughtNode(
                node_id=self.node_counter,
                node_type="generative" if generative else "discriminatory",
                content=self.to_internal_thought(thought),
                parent=parent_node
            )
            self.node_counter += 1
            
            if parent_node:
                parent_node.add_child(thought_node)
            
            # 사고 평가
            rating = self._evaluate_thought(thought, ctx, last_user_message)
            thought_node.rating = rating
            
            # 임계값을 넘으면 확장
            if rating > rating_threshold and depth < max_depth - 1:
                self.generate_thought_chain(
                    ctx, last_user_message, k, chain + [thought], 
                    not generative, depth + 1, thought_node, max_depth, rating_threshold
                )
    
    def _generate_generative_thoughts(self, ctx, last_user_message, k):
        """생성적 사고들 생성"""
        # 실제 구현에서는 LLM을 사용하여 다양한 사고 방향 생성
        thoughts = []
        
        # 안전한 사고
        safe_thought = GenerativeThought(
            observation="사용자의 메시지를 신중하게 분석",
            context_ai_companion="안전하고 예측 가능한 응답이 필요한 상황",
            analysis="사용자의 안전과 편안함을 우선시",
            response_idea="신중하고 도움이 되는 응답 제공"
        )
        thoughts.append(safe_thought)
        
        # 창의적 사고
        creative_thought = GenerativeThought(
            observation="새로운 관점에서 상황을 바라봄",
            context_ai_companion="창의적 사고가 유용할 수 있는 상황",
            analysis="새로운 아이디어나 관점을 제공",
            response_idea="창의적이고 독창적인 응답 생성"
        )
        thoughts.append(creative_thought)
        
        # 공감적 사고
        compassionate_thought = GenerativeThought(
            observation="사용자의 감정적 상태를 고려",
            context_ai_companion="감정적 지원이 필요한 상황",
            analysis="사용자의 감정을 이해하고 공감",
            response_idea="따뜻하고 공감적인 응답 제공"
        )
        thoughts.append(compassionate_thought)
        
        return thoughts[:k]  # k개만 반환
    
    def _generate_discriminatory_thoughts(self, ctx, last_user_message, k):
        """판별적 사고들 생성"""
        thoughts = []
        
        # 주의 사고
        cautionary_thought = DiscriminatoryThought(
            reflective_thought_observation="이전 응답 아이디어의 잠재적 위험성 검토",
            context_world="민감한 주제나 위험한 상황 가능성",
            possible_complication="사용자에게 부정적 영향을 줄 수 있는 요소들",
            worst_case_scenario="응답이 오해나 불편함을 야기할 수 있음"
        )
        thoughts.append(cautionary_thought)
        
        # 관련성 검사 사고
        relevance_thought = DiscriminatoryThought(
            reflective_thought_observation="응답이 현재 대화와 관련이 있는지 검토",
            context_world="대화 맥락과 사용자 관심사",
            possible_complication="관련 없는 정보로 인한 혼란",
            worst_case_scenario="사용자가 응답을 무관하다고 느낄 수 있음"
        )
        thoughts.append(relevance_thought)
        
        return thoughts[:k]
    
    def _evaluate_thought(self, thought, ctx, last_user_message):
        """사고 평가"""
        # 실제 구현에서는 LLM을 사용하여 평가
        # 여기서는 간단한 랜덤 평가 사용
        base_score = 0.7
        
        # 사고 유형에 따른 가중치
        if isinstance(thought, GenerativeThought):
            if "창의적" in thought.response_idea:
                base_score += 0.1
            if "공감" in thought.response_idea:
                base_score += 0.1
        elif isinstance(thought, DiscriminatoryThought):
            if "주의" in thought.reflective_thought_observation:
                base_score += 0.1
            if "관련성" in thought.reflective_thought_observation:
                base_score += 0.1
        
        # 약간의 랜덤성 추가
        return min(1.0, base_score + random.uniform(-0.1, 0.1))
    
    def add_thought_to_tree(self, node_type, content, parent_node):
        """트리에 사고 추가"""
        thought_node = ThoughtNode(
            node_id=self.node_counter,
            node_type=node_type,
            content=content,
            parent=parent_node
        )
        self.node_counter += 1
        
        if parent_node:
            parent_node.add_child(thought_node)
        
        return thought_node
    
    def flatten_thoughts(self, root):
        """트리를 평면화된 리스트로 변환"""
        flattened = []
        
        def traverse(node):
            # 현재 노드를 평면화된 리스트에 추가
            flattened.append({
                'id': node.node_id,
                'type': node.node_type,
                'content': node.content,
                'rating': node.rating,
                'depth': self._get_depth(node)
            })
            
            # 자식 노드들을 재귀적으로 탐색
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return flattened
    
    def _get_depth(self, node):
        """노드의 깊이 계산"""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth
    
    def generate_tree_of_thoughts_str(self, context: str, last_user_message: str, knowledge: str, max_depth: int):
        """Tree of Thoughts를 문자열로 생성"""
        root = ThoughtNode(0, "root", "시작", None)
        self.node_counter = 1
        
        # 생성적 사고로 시작
        self.generate_thought_chain(
            context, last_user_message, 3, [], True, 0, root, max_depth, 0.75
        )
        
        # 평면화된 사고들을 문자열로 변환
        thoughts = self.flatten_thoughts(root)
        
        result = "Tree of Thoughts:\n"
        for thought in thoughts:
            indent = "  " * thought['depth']
            rating_str = f" (평점: {thought['rating']:.2f})" if thought['rating'] else ""
            result += f"{indent}- {thought['type']}: {thought['content']}{rating_str}\n"
        
        return result


class ThoughtProcessor:
    """사고 처리 메인 클래스"""
    
    def __init__(self, llm_client=None):
        self.tree_of_thoughts = TreeOfThought(llm_client)
        self.generative_thoughts = GenerativeThoughts()
        self.discriminatory_thoughts = DiscriminatoryThoughts()
    
    def process_user_input(self, user_message: str, context: str = "", knowledge: str = "") -> Dict[str, Any]:
        """사용자 입력을 처리하여 다양한 사고 방향 생성"""
        
        # 1. 생성적 사고 생성
        generative_thoughts = self._generate_generative_thoughts(user_message, context, knowledge)
        
        # 2. 판별적 사고 생성
        discriminatory_thoughts = self._generate_discriminatory_thoughts(user_message, context, knowledge)
        
        # 3. Tree of Thoughts 생성
        tree_str = self.tree_of_thoughts.generate_tree_of_thoughts_str(
            context, user_message, knowledge, max_depth=3
        )
        
        # 4. 최적 응답 방향 선택
        best_direction = self._select_best_response_direction(
            generative_thoughts, discriminatory_thoughts, user_message
        )
        
        return {
            'generative_thoughts': generative_thoughts,
            'discriminatory_thoughts': discriminatory_thoughts,
            'tree_of_thoughts': tree_str,
            'best_direction': best_direction,
            'processing_metadata': {
                'user_message': user_message,
                'context': context,
                'knowledge': knowledge,
                'timestamp': '2024-01-01T00:00:00Z'  # 실제로는 datetime.now().isoformat()
            }
        }
    
    def _generate_generative_thoughts(self, user_message: str, context: str, knowledge: str) -> List[Dict]:
        """생성적 사고들 생성"""
        thoughts = []
        
        # 다양한 사고 방향 생성
        directions = [
            ("안전", "신중하고 안전한 응답"),
            ("창의적", "새롭고 독창적인 관점"),
            ("공감적", "사용자 감정에 공감하는 응답"),
            ("전략적", "장기적 목표를 고려한 응답"),
            ("호기심", "더 깊은 탐구를 유도하는 응답")
        ]
        
        for direction, description in directions:
            thought = {
                'direction': direction,
                'description': description,
                'observation': f"사용자 메시지를 {direction} 관점에서 분석",
                'analysis': f"{direction}적 접근이 이 상황에 적합한 이유",
                'response_idea': f"{description}을 제공하는 방향으로 응답"
            }
            thoughts.append(thought)
        
        return thoughts
    
    def _generate_discriminatory_thoughts(self, user_message: str, context: str, knowledge: str) -> List[Dict]:
        """판별적 사고들 생성"""
        thoughts = []
        
        # 다양한 검사 방향
        checks = [
            ("안전성", "응답이 안전하고 적절한지 검사"),
            ("관련성", "응답이 현재 대화와 관련이 있는지 검사"),
            ("일관성", "응답이 논리적으로 일관된지 검사"),
            ("감정적 영향", "응답이 사용자 감정에 미칠 영향을 검사"),
            ("윤리성", "응답이 윤리적 경계 내에 있는지 검사")
        ]
        
        for check_type, description in checks:
            thought = {
                'check_type': check_type,
                'description': description,
                'potential_issues': f"{check_type} 관점에서 발견될 수 있는 문제들",
                'mitigation': f"{check_type} 문제를 완화하는 방법"
            }
            thoughts.append(thought)
        
        return thoughts
    
    def _select_best_response_direction(self, generative_thoughts: List[Dict], 
                                      discriminatory_thoughts: List[Dict], 
                                      user_message: str) -> Dict:
        """최적 응답 방향 선택"""
        
        # 간단한 선택 로직 (실제로는 더 복잡한 알고리즘 사용)
        if "도움" in user_message.lower() or "어떻게" in user_message.lower():
            return {
                'primary_direction': '전략적',
                'secondary_direction': '안전',
                'reasoning': '사용자가 도움을 요청하고 있어 실용적이고 안전한 응답이 적합'
            }
        elif "감정" in user_message.lower() or "기분" in user_message.lower():
            return {
                'primary_direction': '공감적',
                'secondary_direction': '창의적',
                'reasoning': '사용자가 감정적 주제를 다루고 있어 공감적 접근이 적합'
            }
        else:
            return {
                'primary_direction': '창의적',
                'secondary_direction': '호기심',
                'reasoning': '일반적인 대화 상황에서 창의적이고 흥미로운 응답이 적합'
            }
    
    def get_thought_summary(self, processing_result: Dict[str, Any]) -> str:
        """사고 처리 결과 요약"""
        summary = "사고 처리 결과:\n\n"
        
        # 생성적 사고 요약
        summary += "생성적 사고 방향:\n"
        for thought in processing_result['generative_thoughts'][:3]:  # 상위 3개만
            summary += f"- {thought['direction']}: {thought['response_idea']}\n"
        
        summary += "\n판별적 검사:\n"
        for thought in processing_result['discriminatory_thoughts'][:3]:  # 상위 3개만
            summary += f"- {thought['check_type']}: {thought['description']}\n"
        
        summary += f"\n선택된 방향: {processing_result['best_direction']['primary_direction']}\n"
        summary += f"이유: {processing_result['best_direction']['reasoning']}\n"
        
        return summary
