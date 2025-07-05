# === memory/memory_conductor.py ===
# MemoryConductor: h.json 프로토콜 기반 통합 메모리 시스템

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import re
from collections import defaultdict
import uuid

from memory.models import MemoryEpisodeNode, InternalState
from memory.palantirgraph import PalantirGraph


@dataclass
class MemoryLayer:
    """메모리 계층 정보"""
    name: str
    priority: int
    path: str
    description: str
    access_pattern: str


@dataclass
class MemoryProtocol:
    """h.json 프로토콜 기반 메모리 구조"""
    id: str
    type: str  # structure | scar | state | trace | pattern
    tags: List[str]
    context: Dict[str, Any]
    content: str
    subnodes: List[Dict[str, Any]]
    created_at: str
    author: str = "harin"


@dataclass
class Entity:
    """개체 정보"""
    id: str
    name: str
    type: str  # person, place, concept, object
    attributes: Dict[str, Any]
    confidence: float
    source_memory: str
    created_at: str


@dataclass
class Relationship:
    """관계 정보"""
    id: str
    source_entity: str
    target_entity: str
    relationship_type: str
    strength: float
    context: Dict[str, Any]
    source_memory: str
    created_at: str


@dataclass
class Contradiction:
    """모순 정보"""
    id: str
    entity_id: str
    conflicting_memories: List[str]
    contradiction_type: str  # attribute, relationship, fact
    severity: float
    resolution_status: str  # unresolved, resolved, ignored
    created_at: str


@dataclass
class NarrativeMemory:
    """서사적 메모리"""
    id: str
    title: str
    summary: str
    entities: List[str]
    relationships: List[str]
    emotional_arc: Dict[str, float]
    key_events: List[str]
    narrative_type: str  # conversation, experience, reflection
    created_at: str


class MemoryConductor:
    """h.json 프로토콜 기반 통합 메모리 시스템 관리자"""
    
    def __init__(self):
        # h.json 프로토콜 로드
        self.protocol = self._load_protocol()
        
        # 메모리 계층 정의 (h.json 기반)
        self.layers = {
            "hot": MemoryLayer("핫 메모리", 1, "data/memory_data/ha1.jsonl", 
                             "실시간 느낌과 감성 지각 노드", "즉시 접근"),
            "cold": MemoryLayer("콜드 메모리", 2, "data/memory_data/h*.jsonl", 
                              "구조화된 장기 기억", "패턴 기반"),
            "warm": MemoryLayer("웜 메모리", 3, "data/memory_data/*_cache.jsonl", 
                              "사고 노드 캐시", "컨텍스트 기반")
        }
        
        # 메모리 파일 경로들
        self.hot_memory_path = Path("data/memory_data/ha1.jsonl")
        self.cold_memory_paths = [
            Path("data/memory_data/h1.jsonl"),
            Path("data/memory_data/h2.jsonl"),
            Path("data/memory_data/h3.jsonl")
        ]
        self.warm_memory_paths = [
            Path("data/memory_data/memory_cache.jsonl"),
            Path("data/memory_data/emotional_cache.jsonl"),
            Path("data/memory_data/loop_cache.jsonl"),
            Path("data/memory_data/thought_cache.jsonl")
        ]
        
        # 메모리 리스트들 (경험 기반 시스템용)
        self.hot_memory: List[MemoryEpisodeNode] = []
        self.warm_memory: List[MemoryEpisodeNode] = []
        self.cold_memory: List[MemoryEpisodeNode] = []
        
        # 메모리 크기 제한
        self.max_hot_memory_size = 50
        self.max_warm_memory_size = 200
        self.max_cold_memory_size = 50000
        
        # 팔란티어 그래프
        self.graph = PalantirGraph("memory/data/palantir_graph.json")
        
        # 개체/관계 관리
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.contradictions: Dict[str, Contradiction] = {}
        self.narrative_memories: Dict[str, NarrativeMemory] = {}
        
        # 개체/관계 파일 경로
        self.entities_path = Path("data/memory_data/entities.jsonl")
        self.relationships_path = Path("data/memory_data/relationships.jsonl")
        self.contradictions_path = Path("data/memory_data/contradictions.jsonl")
        self.narrative_path = Path("data/memory_data/narrative_memories.jsonl")
        
        # 메모리 시스템 초기화
        self._initialize_memory_system()
        self._load_entities_and_relationships()
        self._load_memory_episodes()
    
    def _load_protocol(self) -> Dict[str, Any]:
        """h.json 프로토콜 로드"""
        protocol_path = Path("data/memory_data/h.json")
        if protocol_path.exists():
            try:
                with open(protocol_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"프로토콜 로드 오류: {e}")
        
        # 기본 프로토콜 반환
        return {
            "id": "h_protocol_v1",
            "type": "protocol",
            "purpose": "unified high-memory structure for harinCore",
            "categories": {
                "structure": "루프 구조/전환 정보 — 사고 흐름 조정용",
                "scar": "오류 방지 구조 기억 — 사고 차단 및 재구성 조건",
                "triggered_event": "루프 진입/기억화 조건이 되는 발화",
                "thought_trace": "사고 흐름 중 발생한 판단, 결론, 망설임",
                "state": "감정, 침묵, 존재 기반 상태 기록",
                "reaction_pattern": "요한의 발화 리듬에 대해 하린이 어떻게 반응했는지"
            },
            "scan_criteria": [
                "현재 시점 대화일 것",
                "사고 흐름/루프 중 생성된 것일 것",
                "루프 진입 조건이거나 루프 내부 발화일 것",
                "Scar 유발 없이 회피 조건을 충족할 것",
                "하린이 의식적으로 반응/판단/정지/선언한 구조일 것"
            ]
        }
    
    def _initialize_memory_system(self):
        """메모리 시스템 초기화"""
        # 필요한 디렉토리 생성
        Path("data/memory_data").mkdir(parents=True, exist_ok=True)
        
        # 핫 메모리 초기화
        if not self.hot_memory_path.exists():
            self._create_initial_hot_memory()
        
        # 콜드 메모리 파일들 확인
        for path in self.cold_memory_paths:
            if not path.exists():
                print(f"경고: 콜드 메모리 파일이 없습니다: {path}")
        
        # 웜 메모리 파일들 확인
        for path in self.warm_memory_paths:
            if not path.exists():
                print(f"경고: 웜 메모리 파일이 없습니다: {path}")
    
    def _load_entities_and_relationships(self):
        """개체와 관계 데이터 로드"""
        # 개체 로드
        if self.entities_path.exists():
            try:
                with open(self.entities_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            entity = Entity(**data)
                            self.entities[entity.id] = entity
            except Exception as e:
                print(f"개체 로드 오류: {e}")
        
        # 관계 로드
        if self.relationships_path.exists():
            try:
                with open(self.relationships_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            relationship = Relationship(**data)
                            self.relationships[relationship.id] = relationship
            except Exception as e:
                print(f"관계 로드 오류: {e}")
        
        # 모순 로드
        if self.contradictions_path.exists():
            try:
                with open(self.contradictions_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            contradiction = Contradiction(**data)
                            self.contradictions[contradiction.id] = contradiction
            except Exception as e:
                print(f"모순 로드 오류: {e}")
        
        # 서사적 메모리 로드
        if self.narrative_path.exists():
            try:
                with open(self.narrative_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            narrative = NarrativeMemory(**data)
                            self.narrative_memories[narrative.id] = narrative
            except Exception as e:
                print(f"서사적 메모리 로드 오류: {e}")
    
    def _create_initial_hot_memory(self):
        """초기 핫 메모리 생성"""
        initial_node = {
            "id": "ha1_init",
            "type": "hot_memory",
            "tags": ["hot_memory", "episode", "real_time"],
            "context": {
                "source": "system_init",
                "trigger": "핫 메모리 시스템 초기화",
                "importance": 0.9,
                "linked_loops": ["enhanced_main_loop"],
                "reason_for_memory": "실시간 사고 에피소드를 저장하기 위한 핫 메모리 시스템이 시작되었음을 기록"
            },
            "content": "핫 메모리 시스템이 초기화되었습니다. Enhanced Main Loop에서 실시간 사고 에피소드가 자동으로 저장됩니다.",
            "last_used": None,
            "created_at": datetime.now().isoformat(),
            "author": "harin"
        }
        
        try:
            with open(self.hot_memory_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(initial_node, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"초기 핫 메모리 생성 오류: {e}")

    def extract_entities_and_relationships(self, content: str, memory_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """PM 시스템 기반 개체 및 관계 추출"""
        entities = []
        relationships = []
        
        # 개체 추출 패턴
        entity_patterns = {
            'person': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            'place': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:도시|나라|장소|학교|회사|집|방|사무실)\b',
            'concept': r'\b(?:개념|이론|방법|기술|시스템|프로세스|전략|접근법)\s+([가-힣a-zA-Z0-9\s]+)\b',
            'object': r'\b([가-힣a-zA-Z0-9\s]+)\s+(?:책|컴퓨터|전화|차|음식|옷|도구|장비)\b'
        }
        
        # 개체 추출
        for entity_type, pattern in entity_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                entity_name = match.group(1).strip()
                if len(entity_name) > 1:  # 의미있는 길이
                    entity_id = f"{entity_type}_{entity_name}_{memory_id}"
                    
                    # 기존 개체와 중복 확인
                    existing_entity = self._find_entity_by_name(entity_name)
                    if existing_entity:
                        # 기존 개체 업데이트
                        existing_entity.attributes['mentions'] = existing_entity.attributes.get('mentions', 0) + 1
                        existing_entity.attributes['last_seen'] = datetime.now().isoformat()
                        entities.append(existing_entity)
                    else:
                        # 새 개체 생성
                        entity = Entity(
                            id=entity_id,
                            name=entity_name,
                            type=entity_type,
                            attributes={
                                'mentions': 1,
                                'first_seen': datetime.now().isoformat(),
                                'last_seen': datetime.now().isoformat(),
                                'confidence': 0.8
                            },
                            confidence=0.8,
                            source_memory=memory_id,
                            created_at=datetime.now().isoformat()
                        )
                        entities.append(entity)
                        self.entities[entity_id] = entity
        
        # 관계 추출 패턴
        relationship_patterns = [
            (r'\b([A-Z][a-z]+)\s+(?:가|이|은|는)\s+([A-Z][a-z]+)\s+(?:와|과|에게|를|을)\s+(?:말하다|대화하다|만나다|도와주다)\b', 'interacts_with'),
            (r'\b([A-Z][a-z]+)\s+(?:가|이|은|는)\s+([A-Z][a-z]+)\s+(?:를|을)\s+(?:좋아하다|싫어하다|사랑하다|존경하다)\b', 'emotionally_connected_to'),
            (r'\b([A-Z][a-z]+)\s+(?:가|이|은|는)\s+([A-Z][a-z]+)\s+(?:에서|에)\s+(?:일하다|공부하다|살다|방문하다)\b', 'located_at'),
            (r'\b([A-Z][a-z]+)\s+(?:가|이|은|는)\s+([A-Z][a-z]+)\s+(?:의|에)\s+(?:일부|멤버|학생|직원)\b', 'part_of'),
            (r'\b([A-Z][a-z]+)\s+(?:가|이|은|는)\s+([A-Z][a-z]+)\s+(?:보다|보다는)\s+(?:크다|작다|빠르다|느리다)\b', 'compared_to')
        ]
        
        # 관계 추출
        for pattern, rel_type in relationship_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                source_name = match.group(1).strip()
                target_name = match.group(2).strip()
                
                source_entity = self._find_entity_by_name(source_name)
                target_entity = self._find_entity_by_name(target_name)
                
                if source_entity and target_entity:
                    relationship_id = f"{rel_type}_{source_entity.id}_{target_entity.id}_{memory_id}"
                    
                    # 기존 관계 확인
                    existing_rel = self._find_relationship(source_entity.id, target_entity.id, rel_type)
                    if existing_rel:
                        # 기존 관계 강화
                        existing_rel.strength = min(1.0, existing_rel.strength + 0.1)
                        existing_rel.context['mentions'] = existing_rel.context.get('mentions', 0) + 1
                        relationships.append(existing_rel)
                    else:
                        # 새 관계 생성
                        relationship = Relationship(
                            id=relationship_id,
                            source_entity=source_entity.id,
                            target_entity=target_entity.id,
                            relationship_type=rel_type,
                            strength=0.5,
                            context={
                                'mentions': 1,
                                'first_seen': datetime.now().isoformat(),
                                'last_seen': datetime.now().isoformat()
                            },
                            source_memory=memory_id,
                            created_at=datetime.now().isoformat()
                        )
                        relationships.append(relationship)
                        self.relationships[relationship_id] = relationship
        
        return entities, relationships
    
    def detect_contradictions(self, new_entities: List[Entity], new_relationships: List[Relationship]) -> List[Contradiction]:
        """PM 시스템 기반 모순 감지"""
        contradictions = []
        
        for new_entity in new_entities:
            # 기존 개체들과 비교
            for existing_entity in self.entities.values():
                if existing_entity.name == new_entity.name and existing_entity.id != new_entity.id:
                    # 속성 모순 검사
                    contradictions.extend(self._check_attribute_contradictions(new_entity, existing_entity))
        
        for new_rel in new_relationships:
            # 기존 관계들과 비교
            for existing_rel in self.relationships.values():
                if (existing_rel.source_entity == new_rel.source_entity and 
                    existing_rel.target_entity == new_rel.target_entity):
                    # 관계 모순 검사
                    contradictions.extend(self._check_relationship_contradictions(new_rel, existing_rel))
        
        return contradictions
    
    def _check_attribute_contradictions(self, entity1: Entity, entity2: Entity) -> List[Contradiction]:
        """속성 모순 검사"""
        contradictions = []
        
        # 타입 모순
        if entity1.type != entity2.type:
            contradiction = Contradiction(
                id=f"contradiction_type_{entity1.id}_{entity2.id}",
                entity_id=entity1.id,
                conflicting_memories=[entity1.source_memory, entity2.source_memory],
                contradiction_type="attribute",
                severity=0.7,
                resolution_status="unresolved",
                created_at=datetime.now().isoformat()
            )
            contradictions.append(contradiction)
            self.contradictions[contradiction.id] = contradiction
        
        # 속성 모순
        for attr, value1 in entity1.attributes.items():
            if attr in entity2.attributes:
                value2 = entity2.attributes[attr]
                if value1 != value2:
                    contradiction = Contradiction(
                        id=f"contradiction_attr_{entity1.id}_{attr}",
                        entity_id=entity1.id,
                        conflicting_memories=[entity1.source_memory, entity2.source_memory],
                        contradiction_type="attribute",
                        severity=0.5,
                        resolution_status="unresolved",
                        created_at=datetime.now().isoformat()
                    )
                    contradictions.append(contradiction)
                    self.contradictions[contradiction.id] = contradiction
        
        return contradictions
    
    def _check_relationship_contradictions(self, rel1: Relationship, rel2: Relationship) -> List[Contradiction]:
        """관계 모순 검사"""
        contradictions = []
        
        # 관계 타입 모순
        if rel1.relationship_type != rel2.relationship_type:
            contradiction = Contradiction(
                id=f"contradiction_rel_{rel1.id}_{rel2.id}",
                entity_id=rel1.source_entity,
                conflicting_memories=[rel1.source_memory, rel2.source_memory],
                contradiction_type="relationship",
                severity=0.6,
                resolution_status="unresolved",
                created_at=datetime.now().isoformat()
            )
            contradictions.append(contradiction)
            self.contradictions[contradiction.id] = contradiction
        
        return contradictions
    
    def create_narrative_memory(self, content: str, memory_id: str, 
                               entities: List[Entity], relationships: List[Relationship]) -> NarrativeMemory:
        """PM 시스템 기반 서사적 메모리 생성"""
        # 감정 아크 분석
        emotional_arc = self._analyze_emotional_arc(content)
        
        # 핵심 이벤트 추출
        key_events = self._extract_key_events(content)
        
        # 서사 타입 결정
        narrative_type = self._determine_narrative_type(content)
        
        # 요약 생성
        summary = self._generate_narrative_summary(content, entities, relationships)
        
        narrative = NarrativeMemory(
            id=f"narrative_{memory_id}",
            title=f"서사적 기억 {memory_id}",
            summary=summary,
            entities=[e.id for e in entities],
            relationships=[r.id for r in relationships],
            emotional_arc=emotional_arc,
            key_events=key_events,
            narrative_type=narrative_type,
            created_at=datetime.now().isoformat()
        )
        
        self.narrative_memories[narrative.id] = narrative
        return narrative
    
    def _analyze_emotional_arc(self, content: str) -> Dict[str, float]:
        """감정 아크 분석"""
        emotional_arc = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0,
            'trust': 0.0,
            'anticipation': 0.0
        }
        
        # 감정 키워드 매칭
        emotion_keywords = {
            'joy': ['기쁘', '행복', '즐거', '웃', '환희', '희망'],
            'sadness': ['슬프', '우울', '실망', '절망', '눈물', '비통'],
            'anger': ['화나', '분노', '짜증', '열받', '격분', '성난'],
            'fear': ['무서', '두려', '공포', '불안', '걱정', '겁'],
            'surprise': ['놀라', '충격', '예상', '갑작', '뜻밖'],
            'disgust': ['역겨', '싫', '혐오', '구역질', '메스껍'],
            'trust': ['믿', '신뢰', '확신', '안심', '의지'],
            'anticipation': ['기대', '희망', '예상', '준비', '대기']
        }
        
        for emotion, keywords in emotion_keywords.items():
            count = sum(content.count(keyword) for keyword in keywords)
            if count > 0:
                emotional_arc[emotion] = min(1.0, count / 10.0)  # 정규화
        
        return emotional_arc
    
    def _extract_key_events(self, content: str) -> List[str]:
        """핵심 이벤트 추출"""
        events = []
        
        # 이벤트 패턴
        event_patterns = [
            r'([가-힣a-zA-Z0-9\s]+)\s+(?:했다|했다|했다|했다|했다|했다)',
            r'([가-힣a-zA-Z0-9\s]+)\s+(?:발생했다|일어났다|생겼다)',
            r'([가-힣a-zA-Z0-9\s]+)\s+(?:결정했다|선택했다|결정했다)',
            r'([가-힣a-zA-Z0-9\s]+)\s+(?:발견했다|알았다|깨달았다)'
        ]
        
        for pattern in event_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                event = match.group(1).strip()
                if len(event) > 3:  # 의미있는 길이
                    events.append(event)
        
        return events[:5]  # 최대 5개 이벤트
    
    def _determine_narrative_type(self, content: str) -> str:
        """서사 타입 결정"""
        if any(word in content for word in ['대화', '말하다', '이야기', '대화하다']):
            return 'conversation'
        elif any(word in content for word in ['경험', '체험', '겪다', '만나다']):
            return 'experience'
        elif any(word in content for word in ['생각', '고민', '반성', '깨달음']):
            return 'reflection'
        else:
            return 'general'
    
    def _generate_narrative_summary(self, content: str, entities: List[Entity], relationships: List[Relationship]) -> str:
        """서사적 요약 생성"""
        # 간단한 요약 생성 (실제로는 LLM 사용 권장)
        words = content.split()
        if len(words) > 50:
            summary = ' '.join(words[:50]) + "..."
        else:
            summary = content
        
        # 개체 정보 추가
        if entities:
            entity_names = [e.name for e in entities[:3]]
            summary += f" (관련 개체: {', '.join(entity_names)})"
        
        return summary
    
    def _find_entity_by_name(self, name: str) -> Optional[Entity]:
        """이름으로 개체 찾기"""
        for entity in self.entities.values():
            if entity.name.lower() == name.lower():
                return entity
        return None
    
    def _find_relationship(self, source_id: str, target_id: str, rel_type: str) -> Optional[Relationship]:
        """관계 찾기"""
        for rel in self.relationships.values():
            if (rel.source_entity == source_id and 
                rel.target_entity == target_id and 
                rel.relationship_type == rel_type):
                return rel
        return None
    
    def save_entities_and_relationships(self):
        """개체와 관계 데이터 저장"""
        # 개체 저장
        try:
            with open(self.entities_path, 'w', encoding='utf-8') as f:
                for entity in self.entities.values():
                    f.write(json.dumps(asdict(entity), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"개체 저장 오류: {e}")
        
        # 관계 저장
        try:
            with open(self.relationships_path, 'w', encoding='utf-8') as f:
                for rel in self.relationships.values():
                    f.write(json.dumps(asdict(rel), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"관계 저장 오류: {e}")
        
        # 모순 저장
        try:
            with open(self.contradictions_path, 'w', encoding='utf-8') as f:
                for contradiction in self.contradictions.values():
                    f.write(json.dumps(asdict(contradiction), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"모순 저장 오류: {e}")
        
        # 서사적 메모리 저장
        try:
            with open(self.narrative_path, 'w', encoding='utf-8') as f:
                for narrative in self.narrative_memories.values():
                    f.write(json.dumps(asdict(narrative), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"서사적 메모리 저장 오류: {e}")

    def search_memory_priority(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """h.json 프로토콜 기반 우선순위 메모리 검색: 핫 → 콜드 → 웜"""
        results = []
        
        # 1. 핫 메모리 검색 (최우선)
        hot_results = self._search_hot_memory(query, max_results // 3)
        results.extend(hot_results)
        
        # 2. 콜드 메모리 검색 (h.json 프로토콜 기반)
        cold_results = self._search_cold_memory_protocol(query, max_results // 3)
        results.extend(cold_results)
        
        # 3. 웜 메모리 검색
        warm_results = self._search_warm_memory(query, max_results // 3)
        results.extend(warm_results)
        
        # 우선순위별 정렬
        results.sort(key=lambda x: x.get('priority', 999))
        return results[:max_results]
    
    def _search_cold_memory_protocol(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """h.json 프로토콜 기반 콜드 메모리 검색"""
        results = []
        
        # h.json 인덱스 정보 활용
        if "index" in self.protocol:
            index = self.protocol["index"]
            
            for file_name, file_info in index.items():
                if not file_name.startswith("h"):
                    continue
                
                path = Path(f"data/memory_data/{file_name}.jsonl")
                if not path.exists():
                    continue
                
                # 파일별 검색 기준 적용
                formats = file_info.get("formats", [])
                linked_loops = file_info.get("linked_loops", [])
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                node = json.loads(line)
                                
                                # 프로토콜 기반 관련성 검사
                                if self._is_relevant_by_protocol(node, query, formats, linked_loops):
                                    node['priority'] = 2
                                    results.append(node)
                                    
                                    if len(results) >= max_results:
                                        break
                except Exception as e:
                    print(f"콜드 메모리 검색 오류 ({file_name}): {e}")
        
        return results
    
    def _is_relevant_by_protocol(self, node: Dict[str, Any], query: str, 
                                formats: List[str], linked_loops: List[str]) -> bool:
        """프로토콜 기반 관련성 검사"""
        # 기본 텍스트 매칭
        if query.lower() in node.get('content', '').lower():
            return True
        
        # 태그 매칭
        node_tags = node.get('tags', [])
        if any(tag.lower() in query.lower() for tag in node_tags):
            return True
        
        # 형식 매칭
        node_type = node.get('type', '')
        if node_type in formats:
            return True
        
        # 루프 연결 매칭
        context = node.get('context', {})
        linked_loops_node = context.get('linked_loops', [])
        if any(loop in linked_loops for loop in linked_loops_node):
            return True
        
        return False
    
    def _meets_scan_criteria(self, node: Dict[str, Any], criteria: List[str]) -> bool:
        """스캔 기준 충족 여부 확인"""
        # 간단한 기준 검사 (실제로는 더 정교한 검사 필요)
        content = node.get('content', '')
        tags = node.get('tags', [])
        
        for criterion in criteria:
            if criterion in content or any(criterion in tag for tag in tags):
                return True
        
        return False
    
    def _search_hot_memory(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """핫 메모리 검색"""
        results = []
        
        if not self.hot_memory_path.exists():
            return results
        
        try:
            with open(self.hot_memory_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        node = json.loads(line)
                        if self._is_relevant(node, query):
                            node['priority'] = 1
                            results.append(node)
                            
                            if len(results) >= max_results:
                                break
        except Exception as e:
            print(f"핫 메모리 검색 오류: {e}")
        
        return results
    
    def _search_warm_memory(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """웜 메모리 검색"""
        results = []
        
        for path in self.warm_memory_paths:
            if not path.exists():
                continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            node = json.loads(line)
                            if self._is_relevant(node, query):
                                node['priority'] = 3
                                results.append(node)
                                
                                if len(results) >= max_results:
                                    break
            except Exception as e:
                print(f"웜 메모리 검색 오류 ({path}): {e}")
        
        return results
    
    def _is_relevant(self, node: Dict[str, Any], query: str) -> bool:
        """관련성 검사"""
        # 기본 텍스트 매칭
        content = node.get('content', '').lower()
        if query.lower() in content:
            return True
        
        # 태그 매칭
        tags = node.get('tags', [])
        if any(query.lower() in tag.lower() for tag in tags):
            return True
        
        # 컨텍스트 매칭
        context = node.get('context', {})
        context_str = str(context).lower()
        if query.lower() in context_str:
            return True
        
        return False

    def store_hot_memory(self, episode_node: MemoryEpisodeNode):
        """핫 메모리에 에피소드 저장"""
        # 개체 및 관계 추출
        entities, relationships = self.extract_entities_and_relationships(
            episode_node.topic_summary, episode_node.uuid
        )
        
        # 모순 감지
        contradictions = self.detect_contradictions(entities, relationships)
        
        # 서사적 메모리 생성
        narrative = self.create_narrative_memory(
            episode_node.topic_summary, episode_node.uuid, entities, relationships
        )
        
        # 메모리 노드 생성
        memory_node = {
            "id": episode_node.uuid,
            "type": "hot_memory",
            "tags": getattr(episode_node, 'tags', []),
            "context": {
                "source": getattr(episode_node, 'source', 'unknown'),
                "trigger": getattr(episode_node, 'trigger', 'unknown'),
                "importance": getattr(episode_node, 'importance', 0.5),
                "linked_loops": getattr(episode_node, 'linked_loops', []),
                "reason_for_memory": getattr(episode_node, 'reason_for_memory', ''),
                "entities": [e.id for e in entities],
                "relationships": [r.id for r in relationships],
                "contradictions": [c.id for c in contradictions],
                "narrative_id": narrative.id
            },
            "content": episode_node.topic_summary,
            "last_used": None,
            "created_at": episode_node.timestamp.isoformat(),
            "author": "harin"
        }
        
        # 핫 메모리에 저장
        try:
            with open(self.hot_memory_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(memory_node, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"핫 메모리 저장 오류: {e}")
        
        # 개체/관계 데이터 저장
        self.save_entities_and_relationships()
        
        # 경험 기반 메모리 시스템 추가
        self._analyze_experience_value(episode_node)
        self._update_experience_connections(episode_node)
        
        # 핫 메모리 크기 제한
        if len(self.hot_memory) > self.max_hot_memory_size:
            oldest_episode = self.hot_memory.pop(0)
            self._archive_experience(oldest_episode)
        
        return memory_node

    def _analyze_experience_value(self, episode_node: MemoryEpisodeNode):
        """경험의 가치를 분석하고 메타데이터 업데이트"""
        # 감정적 영향도 분석
        emotional_state = episode_node.internal_state
        if hasattr(emotional_state, 'emotional_valence'):
            episode_node.emotional_impact = emotional_state.emotional_valence
        
        # 학습 가치 평가
        learning_keywords = ['learn', 'understand', 'realize', 'discover', 'figure out', 'find out']
        content_lower = episode_node.topic_summary.lower()
        learning_score = sum(1 for keyword in learning_keywords if keyword in content_lower) / len(learning_keywords)
        episode_node.learning_value = min(1.0, learning_score * 2)
        
        # 중요도 평가 (감정적 영향도와 학습 가치 기반)
        episode_node.salience = (abs(episode_node.emotional_impact) + episode_node.learning_value) / 2
        
        # 컨텍스트 태그 추출
        episode_node.context_tags = self._extract_context_tags(episode_node.topic_summary)
        
        # 미래 관련성 예측
        episode_node.future_relevance = self._predict_future_relevance(episode_node)
    
    def _extract_context_tags(self, content: str) -> List[str]:
        """컨텐츠에서 컨텍스트 태그 추출"""
        tags = []
        
        # 기본 태그들
        if any(word in content.lower() for word in ['user', '사용자', '대화']):
            tags.append('user_interaction')
        if any(word in content.lower() for word in ['감정', 'emotion', 'feeling']):
            tags.append('emotional')
        if any(word in content.lower() for word in ['학습', 'learn', 'study']):
            tags.append('learning')
        if any(word in content.lower() for word in ['문제', 'problem', 'issue']):
            tags.append('problem_solving')
        if any(word in content.lower() for word in ['계획', 'plan', 'goal']):
            tags.append('planning')
        
        return tags
    
    def _predict_future_relevance(self, episode_node: MemoryEpisodeNode) -> float:
        """미래 관련성 예측"""
        # 기본 관련성 (학습 가치와 감정적 영향도 기반)
        base_relevance = (episode_node.learning_value + abs(episode_node.emotional_impact)) / 2
        
        # 컨텍스트 태그에 따른 보정
        context_boost = 0.0
        if 'user_interaction' in episode_node.context_tags:
            context_boost += 0.2
        if 'learning' in episode_node.context_tags:
            context_boost += 0.3
        if 'problem_solving' in episode_node.context_tags:
            context_boost += 0.2
        
        return min(1.0, base_relevance + context_boost)
    
    def _update_experience_connections(self, episode_node: MemoryEpisodeNode):
        """관련된 다른 에피소드들과의 연결 업데이트"""
        # 유사한 에피소드 찾기
        similar_episodes = []
        
        for existing_episode in self.hot_memory + self.warm_memory:
            if existing_episode.uuid == episode_node.uuid:
                continue
            
            # 태그 유사성 확인
            common_tags = set(episode_node.context_tags) & set(existing_episode.context_tags)
            if len(common_tags) > 0:
                similar_episodes.append(existing_episode.uuid)
            
            # 감정적 유사성 확인
            if abs(episode_node.emotional_impact - existing_episode.emotional_impact) < 0.3:
                similar_episodes.append(existing_episode.uuid)
        
        # 상위 3개만 유지
        episode_node.related_episodes = list(set(similar_episodes))[:3]
    
    def _archive_experience(self, episode_node: MemoryEpisodeNode):
        """경험을 아카이브로 이동"""
        # 결과 만족도 평가 (기본값)
        if episode_node.outcome_satisfaction == 0.0:
            # 감정적 영향도 기반으로 만족도 추정
            if episode_node.emotional_impact > 0.3:
                episode_node.outcome_satisfaction = 0.7
            elif episode_node.emotional_impact < -0.3:
                episode_node.outcome_satisfaction = -0.3
            else:
                episode_node.outcome_satisfaction = 0.0
        
        # 웜 메모리로 이동
        self.warm_memory.append(episode_node)
        
        # 웜 메모리 크기 제한
        if len(self.warm_memory) > self.max_warm_memory_size:
            # 가장 낮은 검색 우선순위를 가진 에피소드 제거
            self.warm_memory.sort(key=lambda x: x.get_retrieval_priority())
            removed_episode = self.warm_memory.pop(0)
            self._store_to_cold_memory(removed_episode)
    
    def retrieve_relevant_experiences(self, query: str, top_k: int = 5) -> List[MemoryEpisodeNode]:
        """쿼리와 관련된 경험들 검색"""
        # 모든 메모리에서 검색
        all_episodes = self.hot_memory + self.warm_memory + self.cold_memory
        
        # 검색 우선순위로 정렬
        scored_episodes = []
        for episode in all_episodes:
            # 기본 검색 우선순위
            base_score = episode.get_retrieval_priority()
            
            # 쿼리 관련성 점수
            relevance_score = self._calculate_query_relevance(query, episode)
            
            # 최종 점수
            final_score = base_score * 0.6 + relevance_score * 0.4
            scored_episodes.append((final_score, episode))
        
        # 점수순으로 정렬하고 상위 k개 반환
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [episode for score, episode in scored_episodes[:top_k]]
    
    def _calculate_query_relevance(self, query: str, episode: MemoryEpisodeNode) -> float:
        """쿼리와 에피소드의 관련성 계산"""
        query_lower = query.lower()
        content_lower = episode.topic_summary.lower()
        
        # 키워드 매칭
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        common_words = query_words & content_words
        
        if not query_words:
            return 0.0
        
        keyword_score = len(common_words) / len(query_words)
        
        # 태그 매칭
        tag_score = 0.0
        for tag in episode.context_tags:
            if tag.lower() in query_lower:
                tag_score += 0.2
        
        return min(1.0, keyword_score + tag_score)
    
    def get_experience_insights(self, episode_node: MemoryEpisodeNode) -> Dict[str, Any]:
        """경험에서 인사이트 추출"""
        insights = {
            'experience_score': episode_node.get_experience_score(),
            'is_significant': episode_node.is_significant_experience(),
            'retrieval_priority': episode_node.get_retrieval_priority(),
            'emotional_pattern': self._analyze_emotional_pattern(episode_node),
            'learning_opportunities': self._identify_learning_opportunities(episode_node),
            'related_experiences_count': len(episode_node.related_episodes)
        }
        
        return insights
    
    def _analyze_emotional_pattern(self, episode_node: MemoryEpisodeNode) -> Dict[str, Any]:
        """감정적 패턴 분석"""
        pattern = {
            'intensity': abs(episode_node.emotional_impact),
            'valence': 'positive' if episode_node.emotional_impact > 0 else 'negative' if episode_node.emotional_impact < 0 else 'neutral',
            'stability': 1.0 - abs(episode_node.emotional_impact),  # 높은 감정 = 낮은 안정성
            'impact_level': 'high' if abs(episode_node.emotional_impact) > 0.7 else 'medium' if abs(episode_node.emotional_impact) > 0.3 else 'low'
        }
        
        return pattern
    
    def _identify_learning_opportunities(self, episode_node: MemoryEpisodeNode) -> List[str]:
        """학습 기회 식별"""
        opportunities = []
        
        if episode_node.learning_value > 0.5:
            opportunities.append("high_learning_potential")
        
        if episode_node.future_relevance > 0.7:
            opportunities.append("future_applicability")
        
        if len(episode_node.related_episodes) > 2:
            opportunities.append("pattern_recognition")
        
        if episode_node.outcome_satisfaction < 0:
            opportunities.append("improvement_needed")
        
        return opportunities

    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        stats = {
            "hot_memory": {
                "file_path": str(self.hot_memory_path),
                "node_count": self._count_nodes_in_file(self.hot_memory_path),
                "file_size": self.hot_memory_path.stat().st_size if self.hot_memory_path.exists() else 0
            },
            "cold_memory": {
                "files": [],
                "total_nodes": 0,
                "total_size": 0
            },
            "warm_memory": {
                "files": [],
                "total_nodes": 0,
                "total_size": 0
            },
            "entities": {
                "count": len(self.entities),
                "types": defaultdict(int)
            },
            "relationships": {
                "count": len(self.relationships),
                "types": defaultdict(int)
            },
            "contradictions": {
                "count": len(self.contradictions),
                "unresolved": len([c for c in self.contradictions.values() if c.resolution_status == "unresolved"])
            },
            "narrative_memories": {
                "count": len(self.narrative_memories),
                "types": defaultdict(int)
            }
        }
        
        # 콜드 메모리 통계
        for path in self.cold_memory_paths:
            if path.exists():
                node_count = self._count_nodes_in_file(path)
                file_size = path.stat().st_size
                stats["cold_memory"]["files"].append({
                    "path": str(path),
                    "node_count": node_count,
                    "file_size": file_size
                })
                stats["cold_memory"]["total_nodes"] += node_count
                stats["cold_memory"]["total_size"] += file_size
        
        # 웜 메모리 통계
        for path in self.warm_memory_paths:
            if path.exists():
                node_count = self._count_nodes_in_file(path)
                file_size = path.stat().st_size
                stats["warm_memory"]["files"].append({
                    "path": str(path),
                    "node_count": node_count,
                    "file_size": file_size
                })
                stats["warm_memory"]["total_nodes"] += node_count
                stats["warm_memory"]["total_size"] += file_size
        
        # 개체 타입 통계
        for entity in self.entities.values():
            stats["entities"]["types"][entity.type] += 1
        
        # 관계 타입 통계
        for rel in self.relationships.values():
            stats["relationships"]["types"][rel.relationship_type] += 1
        
        # 서사 타입 통계
        for narrative in self.narrative_memories.values():
            stats["narrative_memories"]["types"][narrative.narrative_type] += 1
        
        return stats
    
    def _count_nodes_in_file(self, file_path: Path) -> int:
        """파일 내 노드 수 계산"""
        if not file_path.exists():
            return 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0
    
    def migrate_hot_to_warm(self, node_id: str):
        """핫 메모리에서 웜 메모리로 마이그레이션"""
        # 핫 메모리에서 노드 찾기
        hot_node = self._find_hot_node(node_id)
        if not hot_node:
            print(f"핫 메모리에서 노드를 찾을 수 없습니다: {node_id}")
            return
        
        # 웜 메모리에 저장
        warm_memory_path = Path("data/memory_data/memory_cache.jsonl")
        try:
            with open(warm_memory_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(hot_node, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"웜 메모리 저장 오류: {e}")
            return
        
        # 핫 메모리에서 제거
        self._remove_hot_node(node_id)
        
        print(f"노드 {node_id}가 핫 메모리에서 웜 메모리로 마이그레이션되었습니다.")
    
    def _find_hot_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """핫 메모리에서 노드 찾기"""
        if not self.hot_memory_path.exists():
            return None
        
        try:
            with open(self.hot_memory_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        node = json.loads(line)
                        if node.get('id') == node_id:
                            return node
        except Exception as e:
            print(f"핫 노드 검색 오류: {e}")
        
        return None
    
    def _remove_hot_node(self, node_id: str):
        """핫 메모리에서 노드 제거"""
        if not self.hot_memory_path.exists():
            return
        
        temp_path = self.hot_memory_path.with_suffix('.tmp')
        
        try:
            with open(self.hot_memory_path, 'r', encoding='utf-8') as f_in:
                with open(temp_path, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        if line.strip():
                            node = json.loads(line)
                            if node.get('id') != node_id:
                                f_out.write(line)
            
            # 임시 파일을 원본 파일로 교체
            temp_path.replace(self.hot_memory_path)
        except Exception as e:
            print(f"핫 노드 제거 오류: {e}")
            if temp_path.exists():
                temp_path.unlink()
    
    def _find_warm_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """웜 메모리에서 노드 찾기"""
        for path in self.warm_memory_paths:
            if not path.exists():
                continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            node = json.loads(line)
                            if node.get('id') == node_id:
                                return node
            except Exception as e:
                print(f"웜 노드 검색 오류 ({path}): {e}")
        
        return None
    
    def _remove_warm_node(self, node_id: str):
        """웜 메모리에서 노드 제거"""
        for path in self.warm_memory_paths:
            if not path.exists():
                continue
            
            temp_path = path.with_suffix('.tmp')
            
            try:
                with open(path, 'r', encoding='utf-8') as f_in:
                    with open(temp_path, 'w', encoding='utf-8') as f_out:
                        for line in f_in:
                            if line.strip():
                                node = json.loads(line)
                                if node.get('id') != node_id:
                                    f_out.write(line)
                
                # 임시 파일을 원본 파일로 교체
                temp_path.replace(path)
                break  # 첫 번째 발견된 노드만 제거
            except Exception as e:
                print(f"웜 노드 제거 오류 ({path}): {e}")
                if temp_path.exists():
                    temp_path.unlink()

    def _load_memory_episodes(self):
        """기존 메모리 파일들에서 MemoryEpisodeNode 로드"""
        # 핫 메모리 로드
        if self.hot_memory_path.exists():
            try:
                with open(self.hot_memory_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # h.json 프로토콜 형식을 MemoryEpisodeNode로 변환
                            episode_node = MemoryEpisodeNode(
                                uuid=data.get('id', str(uuid.uuid4())),
                                topic_summary=data.get('content', ''),
                                internal_state=InternalState(
                                    emotional_valence=data.get('context', {}).get('importance', 0.0),
                                    cognitive_process=data.get('type', 'hot_memory')
                                ),
                                experience_type='interaction',
                                context_tags=data.get('tags', [])
                            )
                            self.hot_memory.append(episode_node)
            except Exception as e:
                print(f"핫 메모리 에피소드 로드 오류: {e}")
        
        # 웜 메모리 로드
        for path in self.warm_memory_paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                episode_node = MemoryEpisodeNode(
                                    uuid=data.get('id', str(uuid.uuid4())),
                                    topic_summary=data.get('content', ''),
                                    internal_state=InternalState(
                                        emotional_valence=data.get('context', {}).get('importance', 0.0),
                                        cognitive_process=data.get('type', 'warm_memory')
                                    ),
                                    experience_type='reflection',
                                    context_tags=data.get('tags', [])
                                )
                                self.warm_memory.append(episode_node)
                except Exception as e:
                    print(f"웜 메모리 에피소드 로드 오류 ({path}): {e}")
        
        # 콜드 메모리 로드
        for path in self.cold_memory_paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                episode_node = MemoryEpisodeNode(
                                    uuid=data.get('id', str(uuid.uuid4())),
                                    topic_summary=data.get('content', ''),
                                    internal_state=InternalState(
                                        emotional_valence=data.get('context', {}).get('importance', 0.0),
                                        cognitive_process=data.get('type', 'cold_memory')
                                    ),
                                    experience_type='learning',
                                    context_tags=data.get('tags', [])
                                )
                                self.cold_memory.append(episode_node)
                except Exception as e:
                    print(f"콜드 메모리 에피소드 로드 오류 ({path}): {e}")
        
        print(f"메모리 에피소드 로드 완료: 핫={len(self.hot_memory)}, 웜={len(self.warm_memory)}, 콜드={len(self.cold_memory)}")
    
    def _store_to_cold_memory(self, episode_node: MemoryEpisodeNode):
        """에피소드를 콜드 메모리에 저장"""
        self.cold_memory.append(episode_node)
        
        # 콜드 메모리 크기 제한
        if len(self.cold_memory) > self.max_cold_memory_size:
            # 가장 낮은 검색 우선순위를 가진 에피소드 제거
            self.cold_memory.sort(key=lambda x: x.get_retrieval_priority())
            self.cold_memory.pop(0)
