"""
사고 단위 분해기 (Thought Unit Decomposer)
복합적인 사고를 독립적인 단위로 분해하고 각각을 추적
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ThoughtUnitType(Enum):
    """사고 단위 타입"""
    CONCEPT = "concept"           # 개념
    REASONING = "reasoning"       # 추론
    JUDGMENT = "judgment"         # 판단
    EMOTION = "emotion"          # 감정
    MEMORY = "memory"            # 기억
    INTENTION = "intention"      # 의도
    CONTEXT = "context"          # 맥락
    RELATIONSHIP = "relationship" # 관계


@dataclass
class ThoughtUnit:
    """사고 단위"""
    id: str
    content: str
    unit_type: ThoughtUnitType
    confidence: float
    dependencies: Set[str] = field(default_factory=set)
    relationships: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    processed: bool = False
    trace_id: Optional[str] = None


@dataclass
class DecompositionResult:
    """분해 결과"""
    units: List[ThoughtUnit]
    relationships: List[Dict[str, Any]]
    decomposition_tree: Dict[str, Any]
    coverage_score: float
    complexity_analysis: Dict[str, Any]


class ThoughtUnitDecomposer:
    """사고 단위 분해기"""
    
    def __init__(self):
        self.unit_patterns = {
            ThoughtUnitType.CONCEPT: [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 대문자로 시작하는 명사구
                r'\b\d+[A-Za-z]+\b',  # 숫자+문자 조합
                r'\b[A-Z]{2,}\b',  # 대문자 약어
            ],
            ThoughtUnitType.REASONING: [
                r'왜냐하면', r'그래서', r'따라서', r'결과적으로',
                r'이유는', r'원인은', r'때문에', r'로 인해',
                r'분석하면', r'고려하면', r'생각해보면'
            ],
            ThoughtUnitType.JUDGMENT: [
                r'맞다', r'틀리다', r'옳다', r'그르다',
                r'좋다', r'나쁘다', r'적절하다', r'부적절하다',
                r'필요하다', r'불필요하다', r'중요하다'
            ],
            ThoughtUnitType.EMOTION: [
                r'좋아', r'싫어', r'화나', r'기뻐', r'슬퍼',
                r'짜증나', r'답답해', r'불안해', r'걱정돼',
                r'감사', r'미안', r'죄송', r'고마워'
            ],
            ThoughtUnitType.MEMORY: [
                r'기억', r'생각', r'경험', r'과거', r'이전',
                r'앞서', r'지난번', r'언젠가', r'그때'
            ],
            ThoughtUnitType.INTENTION: [
                r'하려고', r'하고 싶다', r'목표', r'의도',
                r'계획', r'예정', r'바라', r'원해'
            ],
            ThoughtUnitType.CONTEXT: [
                r'상황', r'환경', r'조건', r'배경',
                r'때문에', r'관련해서', r'대해서', r'에 대해'
            ],
            ThoughtUnitType.RELATIONSHIP: [
                r'관계', r'연결', r'비교', r'대조',
                r'유사', r'다른', r'같은', r'반대'
            ]
        }
        
        self.relationship_patterns = [
            (r'그리고', 'parallel'),
            (r'또한', 'parallel'),
            (r'하지만', 'contrast'),
            (r'그런데', 'contrast'),
            (r'따라서', 'causal'),
            (r'결과적으로', 'causal'),
            (r'예를 들어', 'exemplification'),
            (r'즉', 'clarification'),
            (r'특히', 'emphasis'),
            (r'중요한 것은', 'emphasis')
        ]
    
    def decompose_thought(self, text: str, trace_id: Optional[str] = None) -> DecompositionResult:
        """사고 분해"""
        # 1. 기본 단위 추출
        units = self._extract_basic_units(text, trace_id)
        
        # 2. 관계 분석
        relationships = self._analyze_relationships(units, text)
        
        # 3. 의존성 분석
        self._analyze_dependencies(units, relationships)
        
        # 4. 분해 트리 생성
        decomposition_tree = self._create_decomposition_tree(units, relationships)
        
        # 5. 커버리지 점수 계산
        coverage_score = self._calculate_coverage_score(units, text)
        
        # 6. 복잡성 분석
        complexity_analysis = self._analyze_complexity(units, relationships)
        
        return DecompositionResult(
            units=units,
            relationships=relationships,
            decomposition_tree=decomposition_tree,
            coverage_score=coverage_score,
            complexity_analysis=complexity_analysis
        )
    
    def _extract_basic_units(self, text: str, trace_id: Optional[str] = None) -> List[ThoughtUnit]:
        """기본 단위 추출"""
        units = []
        
        # 각 타입별로 패턴 매칭
        for unit_type, patterns in self.unit_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    unit = ThoughtUnit(
                        id=f"unit_{uuid.uuid4().hex[:8]}",
                        content=match.group(),
                        unit_type=unit_type,
                        confidence=self._calculate_unit_confidence(match, text),
                        trace_id=trace_id,
                        metadata={
                            "start_pos": match.start(),
                            "end_pos": match.end(),
                            "pattern": pattern
                        }
                    )
                    units.append(unit)
        
        # 문장 단위 분해 (패턴으로 매칭되지 않은 부분)
        sentences = self._extract_sentences(text)
        for sentence in sentences:
            if not any(unit.content in sentence for unit in units):
                # 문장이 다른 단위에 포함되지 않은 경우
                unit = ThoughtUnit(
                    id=f"unit_{uuid.uuid4().hex[:8]}",
                    content=sentence,
                    unit_type=self._classify_sentence_type(sentence),
                    confidence=0.5,
                    trace_id=trace_id,
                    metadata={"sentence_type": "unmatched"}
                )
                units.append(unit)
        
        return units
    
    def _extract_sentences(self, text: str) -> List[str]:
        """문장 추출"""
        # 한국어 문장 분리
        sentence_patterns = [
            r'[^.!?]*[.!?]',  # 일반적인 문장 종결
            r'[^.!?]*[가-힣]다[^가-힣]',  # 한국어 동사 종결
            r'[^.!?]*[가-힣]네[^가-힣]',  # 한국어 감탄사
            r'[^.!?]*[가-힣]어[^가-힣]',  # 한국어 감탄사
        ]
        
        sentences = []
        for pattern in sentence_patterns:
            matches = re.findall(pattern, text)
            sentences.extend([s.strip() for s in matches if s.strip()])
        
        return sentences
    
    def _classify_sentence_type(self, sentence: str) -> ThoughtUnitType:
        """문장 타입 분류"""
        # 간단한 키워드 기반 분류
        if any(word in sentence for word in ['왜', '어떻게', '무엇', '언제', '어디']):
            return ThoughtUnitType.REASONING
        elif any(word in sentence for word in ['좋다', '나쁘다', '맞다', '틀리다']):
            return ThoughtUnitType.JUDGMENT
        elif any(word in sentence for word in ['감사', '미안', '화나', '기뻐']):
            return ThoughtUnitType.EMOTION
        else:
            return ThoughtUnitType.CONTEXT
    
    def _calculate_unit_confidence(self, match, text: str) -> float:
        """단위 신뢰도 계산"""
        # 매칭 강도
        match_length = len(match.group())
        total_length = len(text)
        
        # 위치 가중치 (문장 시작 부분이 더 중요)
        position_weight = 1.0 - (match.start() / total_length)
        
        # 길이 가중치
        length_weight = min(1.0, match_length / 20.0)
        
        return min(1.0, (position_weight + length_weight) / 2.0)
    
    def _analyze_relationships(self, units: List[ThoughtUnit], text: str) -> List[Dict[str, Any]]:
        """관계 분석"""
        relationships = []
        
        for pattern, rel_type in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # 관계 전후의 단위들 찾기
                before_units = [u for u in units if u.metadata.get("end_pos", 0) < match.start()]
                after_units = [u for u in units if u.metadata.get("start_pos", len(text)) > match.end()]
                
                if before_units and after_units:
                    # 가장 가까운 단위들 선택
                    before_unit = max(before_units, key=lambda u: u.metadata.get("end_pos", 0))
                    after_unit = min(after_units, key=lambda u: u.metadata.get("start_pos", len(text)))
                    
                    relationship = {
                        "id": f"rel_{uuid.uuid4().hex[:8]}",
                        "type": rel_type,
                        "connector": match.group(),
                        "source_unit": before_unit.id,
                        "target_unit": after_unit.id,
                        "position": match.start()
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _analyze_dependencies(self, units: List[ThoughtUnit], relationships: List[Dict[str, Any]]):
        """의존성 분석"""
        # 관계 기반 의존성 설정
        for rel in relationships:
            source_id = rel["source_unit"]
            target_id = rel["target_unit"]
            
            # 타겟 단위가 소스 단위에 의존
            target_unit = next((u for u in units if u.id == target_id), None)
            if target_unit:
                target_unit.dependencies.add(source_id)
        
        # 타입 기반 의존성
        for unit in units:
            if unit.unit_type == ThoughtUnitType.JUDGMENT:
                # 판단은 추론이나 개념에 의존
                for other_unit in units:
                    if (other_unit.id != unit.id and 
                        other_unit.unit_type in [ThoughtUnitType.REASONING, ThoughtUnitType.CONCEPT]):
                        unit.dependencies.add(other_unit.id)
            
            elif unit.unit_type == ThoughtUnitType.REASONING:
                # 추론은 개념이나 맥락에 의존
                for other_unit in units:
                    if (other_unit.id != unit.id and 
                        other_unit.unit_type in [ThoughtUnitType.CONCEPT, ThoughtUnitType.CONTEXT]):
                        unit.dependencies.add(other_unit.id)
    
    def _create_decomposition_tree(self, units: List[ThoughtUnit], 
                                 relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """분해 트리 생성"""
        tree = {
            "root": {
                "type": "thought_decomposition",
                "children": []
            }
        }
        
        # 독립적인 단위들 (의존성이 없는)
        independent_units = [u for u in units if not u.dependencies]
        
        for unit in independent_units:
            node = self._create_tree_node(unit, units, relationships)
            tree["root"]["children"].append(node)
        
        return tree
    
    def _create_tree_node(self, unit: ThoughtUnit, all_units: List[ThoughtUnit],
                         relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """트리 노드 생성"""
        node = {
            "id": unit.id,
            "content": unit.content,
            "type": unit.unit_type.value,
            "confidence": unit.confidence,
            "children": []
        }
        
        # 의존하는 단위들 찾기
        dependent_units = [u for u in all_units if unit.id in u.dependencies]
        
        for dep_unit in dependent_units:
            child_node = self._create_tree_node(dep_unit, all_units, relationships)
            node["children"].append(child_node)
        
        return node
    
    def _calculate_coverage_score(self, units: List[ThoughtUnit], text: str) -> float:
        """커버리지 점수 계산"""
        if not units:
            return 0.0
        
        # 단위들이 텍스트를 얼마나 커버하는지 계산
        covered_positions = set()
        
        for unit in units:
            start_pos = unit.metadata.get("start_pos", 0)
            end_pos = unit.metadata.get("end_pos", len(unit.content))
            
            for pos in range(start_pos, end_pos):
                covered_positions.add(pos)
        
        total_positions = len(text)
        coverage = len(covered_positions) / total_positions if total_positions > 0 else 0.0
        
        return min(1.0, coverage)
    
    def _analyze_complexity(self, units: List[ThoughtUnit], 
                          relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """복잡성 분석"""
        return {
            "total_units": len(units),
            "unit_type_distribution": self._get_type_distribution(units),
            "relationship_count": len(relationships),
            "average_dependencies": self._calculate_average_dependencies(units),
            "complexity_level": self._determine_complexity_level(units, relationships)
        }
    
    def _get_type_distribution(self, units: List[ThoughtUnit]) -> Dict[str, int]:
        """타입 분포 계산"""
        distribution = {}
        for unit in units:
            unit_type = unit.unit_type.value
            distribution[unit_type] = distribution.get(unit_type, 0) + 1
        return distribution
    
    def _calculate_average_dependencies(self, units: List[ThoughtUnit]) -> float:
        """평균 의존성 수 계산"""
        if not units:
            return 0.0
        
        total_dependencies = sum(len(unit.dependencies) for unit in units)
        return total_dependencies / len(units)
    
    def _determine_complexity_level(self, units: List[ThoughtUnit], 
                                  relationships: List[Dict[str, Any]]) -> str:
        """복잡성 수준 결정"""
        total_elements = len(units) + len(relationships)
        
        if total_elements <= 3:
            return "simple"
        elif total_elements <= 8:
            return "moderate"
        elif total_elements <= 15:
            return "complex"
        else:
            return "very_complex"
    
    def get_processing_order(self, units: List[ThoughtUnit]) -> List[ThoughtUnit]:
        """처리 순서 결정 (위상 정렬)"""
        # 의존성 그래프 구성
        in_degree = {unit.id: 0 for unit in units}
        graph = {unit.id: [] for unit in units}
        
        for unit in units:
            for dep_id in unit.dependencies:
                if dep_id in graph:
                    graph[dep_id].append(unit.id)
                    in_degree[unit.id] += 1
        
        # 위상 정렬
        order = []
        queue = [unit_id for unit_id, degree in in_degree.items() if degree == 0]
        
        while queue:
            current_id = queue.pop(0)
            order.append(current_id)
            
            for neighbor_id in graph[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)
        
        # 순서대로 단위 반환
        unit_dict = {unit.id: unit for unit in units}
        return [unit_dict[unit_id] for unit_id in order if unit_id in unit_dict]
    
    def mark_processed(self, unit_id: str, units: List[ThoughtUnit]):
        """단위 처리 완료 표시"""
        for unit in units:
            if unit.id == unit_id:
                unit.processed = True
                break
    
    def get_unprocessed_units(self, units: List[ThoughtUnit]) -> List[ThoughtUnit]:
        """미처리 단위 반환"""
        return [unit for unit in units if not unit.processed]
    
    def get_units_by_type(self, units: List[ThoughtUnit], unit_type: ThoughtUnitType) -> List[ThoughtUnit]:
        """타입별 단위 반환"""
        return [unit for unit in units if unit.unit_type == unit_type]
    
    def get_high_confidence_units(self, units: List[ThoughtUnit], threshold: float = 0.7) -> List[ThoughtUnit]:
        """고신뢰도 단위 반환"""
        return [unit for unit in units if unit.confidence >= threshold] 
