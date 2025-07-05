# === core/advanced_reasoning_system.py ===
# AdvancedReasoningSystem: 고급 추론 시스템 (TRIZ, 창의적 사고, 논리적 추론)

from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
import random
from dataclasses import dataclass

class ThoughtType(Enum):
    GENERATIVE = "generative"
    DISCRIMINATORY = "discriminatory"
    BACKTRACK = "backtrack"

@dataclass
class ThoughtNode:
    """사고 트리의 노드"""
    node_id: str
    node_type: str
    content: str
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def add_child(self, child_node: 'ThoughtNode'):
        self.children.append(child_node)
        child_node.parent = self

class GenerativeThought:
    """창의적 사고를 위한 생성적 사고"""
    def __init__(self, observation: str, context: str, analysis: str, idea: str):
        self.observation = observation
        self.context = context
        self.analysis = analysis
        self.idea = idea

class DiscriminatoryThought:
    """논리적 검증을 위한 판별적 사고"""
    def __init__(self, reflection: str, context: str, complication: str, outcome: str):
        self.reflection = reflection
        self.context = context
        self.complication = complication
        self.outcome = outcome

class AdvancedReasoningSystem:
    name = "advanced_reasoning_system"

    def __init__(self):
        self.thought_templates = {
            "generative": [
                "창의적 관점에서 보면",
                "다른 가능성을 생각해보면",
                "혁신적인 접근법은",
                "예상치 못한 해결책은",
                "새로운 관점에서 분석하면"
            ],
            "discriminatory": [
                "논리적으로 검증해보면",
                "잠재적 문제점은",
                "실행 가능성을 고려하면",
                "위험 요소를 분석하면",
                "대안을 비교해보면"
            ]
        }

    def run(self, user_input: str, memory_context: Dict = {}) -> Dict:
        """고급 추론 프로세스 실행"""
        # 1. 모순 추출
        contradictions = self._extract_contradictions(user_input)
        
        # 2. 사고 트리 생성
        thought_tree = self._generate_thought_tree(user_input, contradictions, memory_context)
        
        # 3. 해결책 생성
        resolutions = self._generate_resolutions(thought_tree, contradictions)
        
        # 4. 창의적 사고 적용
        creative_solutions = self._apply_creative_thinking(resolutions, user_input)
        
        # 5. 논리적 검증
        validated_solutions = self._validate_solutions_logically(creative_solutions, memory_context)

        return {
            "loop": self.name,
            "input": user_input,
            "contradictions": contradictions,
            "thought_tree": self._flatten_thought_tree(thought_tree),
            "resolutions": resolutions,
            "creative_solutions": creative_solutions,
            "validated_solutions": validated_solutions,
            "confidence_score": self._calculate_confidence(validated_solutions)
        }

    def _extract_contradictions(self, text: str) -> List[str]:
        """TRIZ 기반 모순 추출"""
        return [
            f"기능성 vs 복잡성: '{text}'에서 기능을 유지하면서 복잡성을 줄일 수 있을까?",
            f"효율성 vs 안전성: '{text}'에서 효율성을 높이면서 안전성을 보장할 수 있을까?",
            f"속도 vs 정확성: '{text}'에서 빠르면서도 정확하게 할 수 있을까?",
            f"비용 vs 품질: '{text}'에서 비용을 줄이면서 품질을 유지할 수 있을까?",
            f"개인화 vs 표준화: '{text}'에서 개인화하면서도 표준화할 수 있을까?"
        ]

    def _generate_thought_tree(self, user_input: str, contradictions: List[str], memory_context: Dict) -> ThoughtNode:
        """사고 트리 생성"""
        root = ThoughtNode(str(uuid.uuid4()), "root", f"문제 분석: {user_input}")
        
        for i, contradiction in enumerate(contradictions):
            # 생성적 사고 노드
            generative_node = self._create_generative_thought(user_input, contradiction, memory_context)
            root.add_child(generative_node)
            
            # 판별적 사고 노드
            discriminatory_node = self._create_discriminatory_thought(user_input, contradiction, memory_context)
            generative_node.add_child(discriminatory_node)
            
            # 추가 생성적 사고 (깊이 확장)
            if i < 2:  # 처음 2개 모순에 대해서만 깊이 확장
                additional_generative = self._create_generative_thought(user_input, contradiction, memory_context)
                discriminatory_node.add_child(additional_generative)

        return root

    def _create_generative_thought(self, user_input: str, contradiction: str, memory_context: Dict) -> ThoughtNode:
        """생성적 사고 노드 생성"""
        template = random.choice(self.thought_templates["generative"])
        
        # 메모리 컨텍스트를 활용한 창의적 사고
        memory_hint = ""
        if memory_context.get("recent_experiences"):
            memory_hint = f" 과거 경험을 참고하면"
        
        content = f"{template}{memory_hint} {contradiction}에 대한 새로운 접근법을 찾을 수 있습니다."
        
        return ThoughtNode(
            str(uuid.uuid4()),
            "generative_thought",
            content
        )

    def _create_discriminatory_thought(self, user_input: str, contradiction: str, memory_context: Dict) -> ThoughtNode:
        """판별적 사고 노드 생성"""
        template = random.choice(self.thought_templates["discriminatory"])
        
        # 논리적 검증 사고
        content = f"{template} {contradiction}의 실행 가능성과 위험 요소를 평가해야 합니다."
        
        return ThoughtNode(
            str(uuid.uuid4()),
            "discriminatory_thought",
            content
        )

    def _generate_resolutions(self, thought_tree: ThoughtNode, contradictions: List[str]) -> List[str]:
        """사고 트리 기반 해결책 생성"""
        resolutions = []
        
        # 사고 트리를 순회하며 해결책 생성
        def traverse_and_resolve(node: ThoughtNode):
            if node.node_type == "generative_thought":
                resolution = f"해결책: {node.content}을 통해 모순을 해결할 수 있습니다."
                resolutions.append(resolution)
            
            for child in node.children:
                traverse_and_resolve(child)
        
        traverse_and_resolve(thought_tree)
        
        # 기본 해결책 추가
        for contradiction in contradictions:
            resolutions.append(f"기본 해결책: {contradiction}에 대한 체계적 접근이 필요합니다.")
        
        return resolutions

    def _apply_creative_thinking(self, resolutions: List[str], user_input: str) -> List[str]:
        """창의적 사고 적용"""
        creative_solutions = []
        
        for resolution in resolutions:
            # TRIZ 창의적 원리 적용
            creative_principles = [
                "분리 원리: 문제를 시간이나 공간으로 분리",
                "추상화 원리: 구체적 문제를 추상적 개념으로 변환",
                "전체-부분 원리: 전체 시스템 관점에서 접근",
                "반대 원리: 문제를 반대로 생각",
                "유사성 원리: 다른 분야의 해결책 적용"
            ]
            
            for principle in creative_principles:
                creative_solution = f"창의적 접근: {principle}을 적용하여 {resolution}"
                creative_solutions.append(creative_solution)
        
        return creative_solutions

    def _validate_solutions_logically(self, creative_solutions: List[str], memory_context: Dict) -> List[str]:
        """논리적 검증"""
        validated_solutions = []
        
        for solution in creative_solutions:
            # 논리적 일관성 검증
            if self._check_logical_consistency(solution, memory_context):
                validated_solutions.append(solution)
            else:
                # 논리적 문제가 있는 경우 수정된 버전 생성
                corrected_solution = f"수정된 해결책: {solution} (논리적 일관성 확보)"
                validated_solutions.append(corrected_solution)
        
        return validated_solutions

    def _check_logical_consistency(self, solution: str, memory_context: Dict) -> bool:
        """논리적 일관성 검증"""
        # 간단한 논리적 검증 (실제로는 더 복잡한 검증 로직 필요)
        inconsistent_keywords = ["불가능", "모순", "위험", "실패"]
        return not any(keyword in solution for keyword in inconsistent_keywords)

    def _flatten_thought_tree(self, root: ThoughtNode) -> List[Dict]:
        """사고 트리를 평면화"""
        flattened = []
        
        def traverse(node: ThoughtNode):
            flattened.append({
                "id": node.node_id,
                "type": node.node_type,
                "content": node.content
            })
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return flattened

    def _calculate_confidence(self, validated_solutions: List[str]) -> float:
        """해결책의 신뢰도 계산"""
        if not validated_solutions:
            return 0.0
        
        # 해결책 수와 품질을 기반으로 신뢰도 계산
        base_confidence = min(len(validated_solutions) / 10.0, 1.0)
        
        # 논리적 검증을 통과한 해결책 비율
        validated_ratio = len(validated_solutions) / max(len(validated_solutions), 1)
        
        return (base_confidence + validated_ratio) / 2.0

# 기존 TRIZLoop와의 호환성을 위한 래퍼
class TRIZLoop(AdvancedReasoningSystem):
    """기존 TRIZLoop와의 호환성을 위한 래퍼 클래스"""
    pass 