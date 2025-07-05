"""
Tree of Thoughts 시스템
PM Machine의 다층 사고 구조와 생성적/판별적 사고 분리를 하린코어에 적용
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
try:
    from enum import Enum, StrEnum
except ImportError:
    import enum
    class StrEnum(str, enum.Enum):
        pass
    Enum = enum.Enum

from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)


class ThoughtType(StrEnum):
    """사고 타입"""
    GENERATIVE = "generative"  # 생성적 사고
    DISCRIMINATIVE = "discriminative"  # 판별적 사고
    EVALUATIVE = "evaluative"  # 평가적 사고
    SYNTHETIC = "synthetic"  # 종합적 사고


class ThoughtStatus(StrEnum):
    """사고 상태"""
    ACTIVE = "active"  # 활성
    EXPANDED = "expanded"  # 확장됨
    EVALUATED = "evaluated"  # 평가됨
    SELECTED = "selected"  # 선택됨
    REJECTED = "rejected"  # 거부됨
    COMPLETED = "completed"  # 완료됨


class ThoughtNode(BaseModel):
    """사고 노드"""
    id: str = Field(description="고유 식별자")
    content: str = Field(description="사고 내용")
    thought_type: ThoughtType = Field(description="사고 타입")
    status: ThoughtStatus = Field(default=ThoughtStatus.ACTIVE, description="사고 상태")
    parent_id: Optional[str] = Field(default=None, description="부모 노드 ID")
    children_ids: List[str] = Field(default_factory=list, description="자식 노드 ID들")
    depth: int = Field(default=0, description="깊이")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="신뢰도")
    evaluation_score: float = Field(default=0.0, ge=0.0, le=1.0, description="평가 점수")
    reasoning: str = Field(default="", description="추론 과정")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    
    def add_child(self, child_id: str):
        """자식 노드 추가"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def remove_child(self, child_id: str):
        """자식 노드 제거"""
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)
    
    def is_leaf(self) -> bool:
        """리프 노드인지 확인"""
        return len(self.children_ids) == 0
    
    def get_branch_size(self) -> int:
        """브랜치 크기 반환 (자식 노드 수)"""
        return len(self.children_ids)


class ThoughtTree(BaseModel):
    """사고 트리"""
    root_id: str = Field(description="루트 노드 ID")
    nodes: Dict[str, ThoughtNode] = Field(default_factory=dict, description="노드들")
    max_depth: int = Field(default=5, description="최대 깊이")
    max_branching: int = Field(default=3, description="최대 분기 수")
    current_focus_id: Optional[str] = Field(default=None, description="현재 포커스 노드 ID")
    
    def add_node(self, node: ThoughtNode) -> bool:
        """노드 추가"""
        if node.id in self.nodes:
            return False
        
        # 부모 노드 확인
        if node.parent_id and node.parent_id not in self.nodes:
            return False
        
        # 깊이 제한 확인
        if node.depth > self.max_depth:
            return False
        
        # 분기 수 제한 확인
        if node.parent_id:
            parent = self.nodes[node.parent_id]
            if len(parent.children_ids) >= self.max_branching:
                return False
        
        self.nodes[node.id] = node
        
        # 부모 노드에 자식 추가
        if node.parent_id:
            self.nodes[node.parent_id].add_child(node.id)
        
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """노드 제거"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        # 자식 노드들도 제거
        for child_id in node.children_ids[:]:
            self.remove_node(child_id)
        
        # 부모 노드에서 제거
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].remove_child(node_id)
        
        del self.nodes[node_id]
        return True
    
    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """노드 가져오기"""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[ThoughtNode]:
        """자식 노드들 가져오기"""
        node = self.get_node(node_id)
        if not node:
            return []
        
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]
    
    def get_ancestors(self, node_id: str) -> List[ThoughtNode]:
        """조상 노드들 가져오기"""
        ancestors = []
        current = self.get_node(node_id)
        
        while current and current.parent_id:
            parent = self.get_node(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def get_descendants(self, node_id: str) -> List[ThoughtNode]:
        """후손 노드들 가져오기"""
        descendants = []
        node = self.get_node(node_id)
        if not node:
            return descendants
        
        for child_id in node.children_ids:
            child = self.get_node(child_id)
            if child:
                descendants.append(child)
                descendants.extend(self.get_descendants(child_id))
        
        return descendants
    
    def get_leaves(self) -> List[ThoughtNode]:
        """리프 노드들 가져오기"""
        return [node for node in self.nodes.values() if node.is_leaf()]
    
    def get_path_to_root(self, node_id: str) -> List[ThoughtNode]:
        """루트까지의 경로 가져오기"""
        path = []
        current = self.get_node(node_id)
        
        while current:
            path.append(current)
            if current.parent_id:
                current = self.get_node(current.parent_id)
            else:
                break
        
        return list(reversed(path))
    
    def get_subtree(self, node_id: str) -> 'ThoughtTree':
        """서브트리 생성"""
        subtree = ThoughtTree(
            root_id=node_id,
            max_depth=self.max_depth,
            max_branching=self.max_branching
        )
        
        # 노드 복사
        descendants = self.get_descendants(node_id)
        root_node = self.get_node(node_id)
        if root_node:
            subtree.nodes[node_id] = root_node.model_copy()
        
        for node in descendants:
            subtree.nodes[node.id] = node.model_copy()
        
        return subtree


class ThoughtEvaluation(BaseModel):
    """사고 평가"""
    node_id: str = Field(description="평가할 노드 ID")
    score: float = Field(ge=0.0, le=1.0, description="평가 점수")
    reasoning: str = Field(description="평가 이유")
    criteria: List[str] = Field(default_factory=list, description="평가 기준")
    evaluator: str = Field(description="평가자")


class ThoughtExpansion(BaseModel):
    """사고 확장"""
    parent_id: str = Field(description="부모 노드 ID")
    new_thoughts: List[Dict[str, Any]] = Field(description="새로운 사고들")
    expansion_type: ThoughtType = Field(description="확장 타입")
    reasoning: str = Field(description="확장 이유")


class TreeOfThoughtsManager:
    """Tree of Thoughts 관리자"""
    
    def __init__(self):
        self.llm_client = self._get_llm_client()
        self.trees: Dict[str, ThoughtTree] = {}
        self.current_tree_id: Optional[str] = None
        
    def _get_llm_client(self):
        """LLM 클라이언트 가져오기"""
        try:
            from core.llm_client import LLMClient
            return LLMClient()
        except ImportError:
            # 폴백: 간단한 모의 클라이언트
            class MockLLMClient:
                def generate_text(self, prompt, max_tokens=200):
                    return "모의 응답"
            return MockLLMClient()
    
    def create_thought_tree(
        self,
        initial_thought: str,
        thought_type: ThoughtType = ThoughtType.GENERATIVE,
        max_depth: int = 5,
        max_branching: int = 3
    ) -> str:
        """사고 트리 생성"""
        import uuid
        
        tree_id = str(uuid.uuid4())
        root_node = ThoughtNode(
            id=str(uuid.uuid4()),
            content=initial_thought,
            thought_type=thought_type,
            depth=0
        )
        
        tree = ThoughtTree(
            root_id=root_node.id,
            max_depth=max_depth,
            max_branching=max_branching
        )
        
        tree.add_node(root_node)
        self.trees[tree_id] = tree
        self.current_tree_id = tree_id
        
        return tree_id
    
    def expand_thought(
        self,
        node_id: str,
        expansion_type: ThoughtType = ThoughtType.GENERATIVE,
        num_expansions: int = 3
    ) -> List[str]:
        """사고 확장"""
        if not self.current_tree_id:
            raise ValueError("활성 트리가 없습니다.")
        
        tree = self.trees[self.current_tree_id]
        parent_node = tree.get_node(node_id)
        
        if not parent_node:
            raise ValueError(f"노드 {node_id}를 찾을 수 없습니다.")
        
        if parent_node.depth >= tree.max_depth:
            raise ValueError("최대 깊이에 도달했습니다.")
        
        if len(parent_node.children_ids) >= tree.max_branching:
            raise ValueError("최대 분기 수에 도달했습니다.")
        
        # LLM을 사용하여 사고 확장
        expanded_thoughts = self._generate_expanded_thoughts(
            parent_node, expansion_type, num_expansions
        )
        
        new_node_ids = []
        import uuid
        
        for thought_content in expanded_thoughts:
            new_node = ThoughtNode(
                id=str(uuid.uuid4()),
                content=thought_content,
                thought_type=expansion_type,
                parent_id=node_id,
                depth=parent_node.depth + 1
            )
            
            if tree.add_node(new_node):
                new_node_ids.append(new_node.id)
        
        return new_node_ids
    
    def _generate_expanded_thoughts(
        self,
        parent_node: ThoughtNode,
        expansion_type: ThoughtType,
        num_expansions: int
    ) -> List[str]:
        """확장된 사고들 생성"""
        try:
            prompt = f"""
다음 사고를 기반으로 {num_expansions}개의 새로운 사고를 생성해주세요:

현재 사고: {parent_node.content}
사고 타입: {expansion_type.value}
깊이: {parent_node.depth}

사고 타입별 지침:
- generative: 새로운 아이디어나 가능성을 탐색
- discriminative: 현재 사고를 분석하고 평가
- evaluative: 가치나 효과성을 판단
- synthetic: 여러 관점을 종합

{num_expansions}개의 서로 다른 사고를 생성해주세요. 각 사고는 한 문장으로 표현해주세요.
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=300)
            
            # 응답을 개별 사고로 분리
            thoughts = []
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # 번호나 기호 제거
                    thought = line.lstrip('0123456789.-• ')
                    if thought:
                        thoughts.append(thought)
            
            return thoughts[:num_expansions]
            
        except Exception as e:
            logger.error(f"사고 확장 생성 중 오류: {e}")
            return [f"확장 사고 {i+1}" for i in range(num_expansions)]
    
    def evaluate_thought(self, node_id: str, criteria: List[str] = None) -> ThoughtEvaluation:
        """사고 평가"""
        if not self.current_tree_id:
            raise ValueError("활성 트리가 없습니다.")
        
        tree = self.trees[self.current_tree_id]
        node = tree.get_node(node_id)
        
        if not node:
            raise ValueError(f"노드 {node_id}를 찾을 수 없습니다.")
        
        if not criteria:
            criteria = ["논리성", "창의성", "실용성", "일관성"]
        
        try:
            prompt = f"""
다음 사고를 평가해주세요:

사고 내용: {node.content}
사고 타입: {node.thought_type.value}
평가 기준: {', '.join(criteria)}

각 기준을 0.0~1.0 사이로 평가하고, 전체 점수를 계산해주세요.
평가 이유도 함께 설명해주세요.

JSON 형태로 응답해주세요:
{{
    "score": 0.0~1.0,
    "reasoning": "평가 이유",
    "criteria_scores": {{
        "논리성": 0.0~1.0,
        "창의성": 0.0~1.0,
        "실용성": 0.0~1.0,
        "일관성": 0.0~1.0
    }}
}}
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=300)
            
            try:
                eval_data = json.loads(response)
                score = eval_data.get("score", 0.5)
                reasoning = eval_data.get("reasoning", "기본 평가")
                
                # 노드 업데이트
                node.evaluation_score = score
                node.reasoning = reasoning
                node.status = ThoughtStatus.EVALUATED
                
                return ThoughtEvaluation(
                    node_id=node_id,
                    score=score,
                    reasoning=reasoning,
                    criteria=criteria,
                    evaluator="LLM"
                )
                
            except:
                return ThoughtEvaluation(
                    node_id=node_id,
                    score=0.5,
                    reasoning="평가 중 오류 발생",
                    criteria=criteria,
                    evaluator="LLM"
                )
                
        except Exception as e:
            logger.error(f"사고 평가 중 오류: {e}")
            return ThoughtEvaluation(
                node_id=node_id,
                score=0.5,
                reasoning=f"평가 중 오류: {e}",
                criteria=criteria or [],
                evaluator="LLM"
            )
    
    def select_best_thoughts(self, num_selections: int = 3) -> List[ThoughtNode]:
        """최고 사고들 선택"""
        if not self.current_tree_id:
            return []
        
        tree = self.trees[self.current_tree_id]
        
        # 평가된 노드들 중에서 선택
        evaluated_nodes = [
            node for node in tree.nodes.values()
            if node.status == ThoughtStatus.EVALUATED
        ]
        
        # 평가 점수로 정렬
        evaluated_nodes.sort(key=lambda x: x.evaluation_score, reverse=True)
        
        # 선택된 노드들 상태 업데이트
        selected_nodes = evaluated_nodes[:num_selections]
        for node in selected_nodes:
            node.status = ThoughtStatus.SELECTED
        
        return selected_nodes
    
    def synthesize_thoughts(self, node_ids: List[str]) -> str:
        """사고들 종합"""
        if not self.current_tree_id:
            return ""
        
        tree = self.trees[self.current_tree_id]
        nodes = [tree.get_node(node_id) for node_id in node_ids if tree.get_node(node_id)]
        
        if not nodes:
            return ""
        
        try:
            prompt = f"""
다음 사고들을 종합하여 하나의 통합된 결론을 도출해주세요:

사고들:
{chr(10).join([f"- {node.content} (점수: {node.evaluation_score:.2f})" for node in nodes])}

각 사고의 장점을 살려서 논리적이고 일관된 결론을 생성해주세요.
"""
            
            synthesis = self.llm_client.generate_text(prompt, max_tokens=400)
            return synthesis.strip()
            
        except Exception as e:
            logger.error(f"사고 종합 중 오류: {e}")
            return "사고 종합 중 오류가 발생했습니다."
    
    def get_thought_tree_summary(self, tree_id: str = None) -> Dict[str, Any]:
        """사고 트리 요약"""
        if not tree_id:
            tree_id = self.current_tree_id
        
        if not tree_id or tree_id not in self.trees:
            return {"error": "트리를 찾을 수 없습니다."}
        
        tree = self.trees[tree_id]
        
        # 통계 계산
        total_nodes = len(tree.nodes)
        leaves = tree.get_leaves()
        evaluated_nodes = [node for node in tree.nodes.values() if node.status == ThoughtStatus.EVALUATED]
        selected_nodes = [node for node in tree.nodes.values() if node.status == ThoughtStatus.SELECTED]
        
        # 평균 평가 점수
        avg_score = 0.0
        if evaluated_nodes:
            avg_score = np.mean([node.evaluation_score for node in evaluated_nodes])
        
        # 사고 타입별 분포
        type_distribution = {}
        for node in tree.nodes.values():
            thought_type = node.thought_type.value
            type_distribution[thought_type] = type_distribution.get(thought_type, 0) + 1
        
        return {
            "tree_id": tree_id,
            "total_nodes": total_nodes,
            "max_depth": tree.max_depth,
            "current_depth": max([node.depth for node in tree.nodes.values()]) if tree.nodes else 0,
            "leaves_count": len(leaves),
            "evaluated_count": len(evaluated_nodes),
            "selected_count": len(selected_nodes),
            "average_score": avg_score,
            "type_distribution": type_distribution,
            "root_content": tree.get_node(tree.root_id).content if tree.get_node(tree.root_id) else ""
        }
    
    def visualize_tree(self, tree_id: str = None) -> str:
        """트리 시각화 (텍스트 기반)"""
        if not tree_id:
            tree_id = self.current_tree_id
        
        if not tree_id or tree_id not in self.trees:
            return "트리를 찾을 수 없습니다."
        
        tree = self.trees[tree_id]
        visualization = []
        
        def add_node_to_visualization(node: ThoughtNode, prefix: str = "", is_last: bool = True):
            # 노드 정보
            status_symbol = {
                ThoughtStatus.ACTIVE: "○",
                ThoughtStatus.EVALUATED: "●",
                ThoughtStatus.SELECTED: "★",
                ThoughtStatus.REJECTED: "✗",
                ThoughtStatus.COMPLETED: "✓"
            }.get(node.status, "○")
            
            score_info = f" ({node.evaluation_score:.2f})" if node.evaluation_score > 0 else ""
            node_line = f"{prefix}{'└── ' if is_last else '├── '}{status_symbol} {node.content}{score_info}"
            visualization.append(node_line)
            
            # 자식 노드들
            children = tree.get_children(node.id)
            for i, child in enumerate(children):
                child_prefix = prefix + ("    " if is_last else "│   ")
                add_node_to_visualization(child, child_prefix, i == len(children) - 1)
        
        # 루트 노드부터 시작
        root = tree.get_node(tree.root_id)
        if root:
            add_node_to_visualization(root)
        
        return "\n".join(visualization)
    
    def prune_tree(self, min_score: float = 0.3) -> int:
        """트리 가지치기"""
        if not self.current_tree_id:
            return 0
        
        tree = self.trees[self.current_tree_id]
        nodes_to_remove = []
        
        # 낮은 점수의 노드들 찾기
        for node in tree.nodes.values():
            if node.evaluation_score < min_score and node.id != tree.root_id:
                nodes_to_remove.append(node.id)
        
        # 노드들 제거
        for node_id in nodes_to_remove:
            tree.remove_node(node_id)
        
        return len(nodes_to_remove)


# 사용 예시
def create_tree_of_thoughts_example():
    """Tree of Thoughts 예시"""
    manager = TreeOfThoughtsManager()
    
    # 트리 생성
    tree_id = manager.create_thought_tree(
        initial_thought="새로운 AI 프로젝트를 시작하려고 합니다.",
        thought_type=ThoughtType.GENERATIVE,
        max_depth=3,
        max_branching=3
    )
    
    print("=== Tree of Thoughts 예시 ===")
    
    # 루트 노드 확장
    tree = manager.trees[tree_id]
    root_id = tree.root_id
    
    # 첫 번째 레벨 확장
    print("1단계: 루트 노드 확장")
    child_ids = manager.expand_thought(root_id, ThoughtType.GENERATIVE, 3)
    
    # 각 자식 노드 평가
    print("2단계: 자식 노드 평가")
    for child_id in child_ids:
        evaluation = manager.evaluate_thought(child_id)
        print(f"노드 평가: {evaluation.score:.2f} - {evaluation.reasoning}")
    
    # 최고 노드 선택
    print("3단계: 최고 노드 선택")
    best_thoughts = manager.select_best_thoughts(2)
    for thought in best_thoughts:
        print(f"선택된 사고: {thought.content} (점수: {thought.evaluation_score:.2f})")
    
    # 사고 종합
    print("4단계: 사고 종합")
    selected_ids = [thought.id for thought in best_thoughts]
    synthesis = manager.synthesize_thoughts(selected_ids)
    print(f"종합 결과: {synthesis}")
    
    # 트리 요약
    print("5단계: 트리 요약")
    summary = manager.get_thought_tree_summary(tree_id)
    print(f"총 노드 수: {summary['total_nodes']}")
    print(f"평균 점수: {summary['average_score']:.2f}")
    
    # 트리 시각화
    print("6단계: 트리 시각화")
    visualization = manager.visualize_tree(tree_id)
    print(visualization)
    
    return manager 