"""
고급 그래프 기반 메모리 시스템
PM Machine의 3층 구조 메모리 시스템을 하린코어에 적용
"""

import json
import logging
import sqlite3
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
try:
    from enum import StrEnum
except ImportError:
    import enum
    class StrEnum(str, enum.Enum):
        pass
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import regex as re

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

# 상수 정의
EGO_NODE_UUID = "00000000-0000-0000-0000-000000000001"
EGO_NODE_NAME = "Self"

class StructuralEdgeLabel(StrEnum):
    INSTANCE_OF = "INSTANCE_OF"
    IS_A = "IS_A"
    EXPERIENCED_BY = "EXPERIENCED_BY"

# 시맨틱 축 정의
SEMANTIC_AXES_DEFINITION = {
    "animacy": ("living being, animal, person", "inanimate object, rock, tool"),
    "social_interaction": ("conversation, relationship, team", "solitude, isolation, individual work"),
    "concreteness": ("physical object, location", "abstract idea, concept, plan"),
    "temporality": ("event, moment in time, history", "enduring state, permanent characteristic"),
    "emotional_tone": ("joy, success, happiness", "sadness, failure, conflict"),
    "agency": ("action, decision, intention", "observation, passive event, occurrence")
}

EMBEDDING_DIM = 768  # 임베딩 차원

# 영어 불용어 집합
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot',
    'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few',
    'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll",
    "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll",
    "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most',
    "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should',
    "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're",
    "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
    "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're",
    "you've", 'your', 'yours', 'yourself', 'yourselves', 'tell', 'give', 'ask', 'name', 'list', 'describe', 'can', 'could'
}


class InternalState(BaseModel):
    """AI의 내부 상태를 구조화된 형태로 표현"""
    emotional_valence: float = Field(
        description="주요 감정의 가치, -1.0(매우 부정적) ~ 1.0(매우 긍정적)",
        default=0
    )
    emotional_label: str = Field(
        description="주요 감정을 나타내는 간결한 단어 (예: 'curiosity', 'satisfaction', 'confusion')",
        default=""
    )
    cognitive_process: str = Field(
        description="지배적인 인지 과정 (예: 'storing_new_fact', 'detecting_contradiction', 'forming_goal')",
        default=""
    )
    certainty: float = Field(
        default=1.0,
        description="AI가 처리된 정보에 대한 확신, 0.0 ~ 1.0"
    )
    salience_focus: List[str] = Field(
        default_factory=list,
        description="주요 관심 대상이 된 엔티티 이름이나 개념들의 리스트"
    )


class MemoryEpisodeNode(BaseModel):
    """메모리 에피소드 노드 - 상호작용의 기본 단위"""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ego_uuid: str = Field(default=EGO_NODE_UUID, description="AI의 중심 EgoNode에 대한 링크")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    topic_summary: str = Field(description="에피소드가 무엇에 관한 것인지에 대한 LLM 생성 요약")
    internal_state: InternalState = Field(
        description="이 에피소드 동안의 AI의 구조화된 내부 상태",
        default_factory=InternalState
    )
    embedding: List[float] = Field(default_factory=list)  # BLOB로 저장

    @model_validator(mode='before')
    @classmethod
    def ensure_utc_timestamp(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ts = data.get('timestamp')
            if isinstance(ts, str):
                data['timestamp'] = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if isinstance(ts, datetime) and ts.tzinfo is None:
                data['timestamp'] = ts.replace(tzinfo=timezone.utc)
        return data


class GraphNode(BaseModel):
    """그래프 노드 - 엔티티나 개념을 나타냄"""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    labels: List[str]  # 예: ["Person"], ["Location"], ["Organization", "Client"]
    source_episode_uuid: str  # 이 노드가 주로 정의되거나 마지막으로 크게 업데이트된 에피소드
    concept_uuid: Optional[str] = None  # ConceptNode에 대한 링크
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: Dict[str, Any] = Field(default_factory=dict)  # 추가적인 비관계형 데이터
    embedding: List[float] = Field(default_factory=list)  # BLOB로 저장

    @model_validator(mode='before')
    @classmethod
    def ensure_utc_created_at(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ts = data.get('created_at')
            if isinstance(ts, str):
                data['created_at'] = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if isinstance(ts, datetime) and ts.tzinfo is None:
                data['created_at'] = ts.replace(tzinfo=timezone.utc)
        return data


class GraphEdge(BaseModel):
    """그래프 엣지 - 노드 간 관계를 나타냄"""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_uuid: str  # 소스 GraphNode의 UUID
    target_uuid: str  # 타겟 GraphNode의 UUID
    label: str  # 예: "works_for", "knows", "located_in"
    fact_text: str  # 사실의 자연어 표현, 예: "Alex works for Acme Corp"
    source_episode_uuid: str  # 이 사실이 학습된 에피소드
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_at: Optional[datetime] = Field(
        default=None,
        description="이 사실이 유효해진 타임스탬프 (해당되는 경우)"
    )
    invalid_at: Optional[datetime] = Field(
        default=None,
        description="이 사실이 무효해진 타임스탬프 (모순/업데이트로 인해)"
    )
    attributes: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float] = Field(default_factory=list)  # BLOB로 저장

    @model_validator(mode='before')
    @classmethod
    def ensure_utc_timestamps(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ['created_at', 'valid_at', 'invalid_at']:
                ts = data.get(key)
                if ts:
                    if isinstance(ts, str):
                        data[key] = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    if isinstance(ts, datetime) and ts.tzinfo is None:
                        data[key] = ts.replace(tzinfo=timezone.utc)
        return data


class ConceptNode(BaseModel):
    """개념 노드 - 추상적 개념을 나타냄"""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="개념의 이름")
    description: str = Field(description="이 개념이 무엇을 나타내는지에 대한 간단한 설명")
    parent_concept_uuid: Optional[str] = Field(
        default=None,
        description="부모 개념의 UUID (IS_A 계층 구조용)"
    )
    embedding: List[float] = Field(default_factory=list)  # BLOB로 저장


# LLM 응답을 위한 스키마들
class AnalyzedInternalState(BaseModel):
    state: InternalState

class EpisodeTopicSummary(BaseModel):
    summary: str

class ExtractedEntity(BaseModel):
    name: str
    label: str  # 제안된 주요 라벨, 예: 'Person', 'Project', 'Location'

class ExtractedEntities(BaseModel):
    entities: List[ExtractedEntity]

class ExtractedRelationship(BaseModel):
    source_name: str
    target_name: str
    label: str  # 예: "works_at", "is_located_in"
    fact: str  # 사실을 나타내는 문장이나 구

class ExtractedRelationships(BaseModel):
    relationships: List[ExtractedRelationship]

class EntityClassifier(BaseModel):
    entity_name: str
    most_likely_concept_name: str
    reasoning: str

class Confirmation(BaseModel):
    is_same: bool
    reasoning: str

class ContradictionCheck(BaseModel):
    is_contradictory: bool
    is_duplicate: bool  # 새로운 사실이 기존 것과 본질적으로 동일한지
    reasoning: str
    conflicting_edge_uuid: Optional[str] = None  # 모순되는 엣지의 UUID

class NewConceptDefinition(BaseModel):
    """새로운 개념을 정의하고 기존 계층 구조 내에 배치"""
    description: str = Field(description="이 새로운 개념이 무엇을 나타내는지에 대한 간단하고 명확한 설명")
    parent_concept_name: str = Field(
        description="이 개념의 부모 개념 이름 (IS_A 관계에서)"
    )

class RephrasedQuestions(BaseModel):
    """질문의 대안적 표현을 제공하여 검색 회수를 개선"""
    questions: List[str] = Field(description="원래 질문의 3가지 다양한 재구성")

class RelevantItems(BaseModel):
    """사용자의 질문과 관련된 것으로 간주되는 항목들의 리스트"""
    relevant_item_identifiers: List[str] = Field(description="질문에 직접적으로 도움이 되는 항목들의 UUID 리스트")


class CognitiveMemoryManager:
    """인지 메모리 관리자 - 3층 구조 메모리 시스템의 핵심"""
    
    def __init__(self, db_path: str = "cognitive_memory.db"):
        self.db_path = db_path
        self.episodes: Dict[str, MemoryEpisodeNode] = {}
        self.graph_nodes: Dict[str, GraphNode] = {}
        self.graph_edges: Dict[str, GraphEdge] = {}
        self.concept_nodes: Dict[str, ConceptNode] = {}
        self.networkx_graph = nx.DiGraph()
        self.lock = threading.Lock()
        
        # LLM 매니저 (기존 하린코어의 LLM 시스템 사용)
        from core.llm_client import LLMClient
        self.llm_client = LLMClient()
        
        self._create_tables_if_not_exist()
        self._initialize_ego_node()
        self._bootstrap_core_concepts()
        self._rebuild_networkx_graph()
    
    def _map_pydantic_type_to_sql(self, pydantic_type: Any) -> str:
        """Pydantic 타입을 SQLite 타입으로 매핑"""
        if pydantic_type == str:
            return "TEXT"
        elif pydantic_type == int:
            return "INTEGER"
        elif pydantic_type == float:
            return "REAL"
        elif pydantic_type == bool:
            return "INTEGER"
        elif pydantic_type == datetime:
            return "TEXT"
        elif pydantic_type == list:
            return "BLOB"  # JSON으로 직렬화
        elif pydantic_type == dict:
            return "BLOB"  # JSON으로 직렬화
        else:
            return "TEXT"
    
    def _prepare_data_for_sql(self, model_instance: BaseModel) -> Tuple:
        """Pydantic 모델 인스턴스를 SQL 데이터로 준비"""
        data = model_instance.model_dump()
        values = []
        
        for field_name, field_info in model_instance.model_fields.items():
            value = data.get(field_name)
            
            if isinstance(value, datetime):
                values.append(value.isoformat())
            elif isinstance(value, (list, dict)):
                values.append(json.dumps(value))
            elif isinstance(value, list) and field_name == 'embedding':
                # 임베딩은 numpy 배열로 변환
                values.append(np.array(value).tobytes())
            else:
                values.append(value)
        
        return tuple(values)
    
    def _create_tables_if_not_exist(self):
        """데이터베이스 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 에피소드 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_episodes (
                    uuid TEXT PRIMARY KEY,
                    ego_uuid TEXT,
                    timestamp TEXT,
                    topic_summary TEXT,
                    internal_state TEXT,
                    embedding BLOB
                )
            """)
            
            # 그래프 노드 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    uuid TEXT PRIMARY KEY,
                    name TEXT,
                    labels TEXT,
                    source_episode_uuid TEXT,
                    concept_uuid TEXT,
                    created_at TEXT,
                    attributes TEXT,
                    embedding BLOB
                )
            """)
            
            # 그래프 엣지 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    uuid TEXT PRIMARY KEY,
                    source_uuid TEXT,
                    target_uuid TEXT,
                    label TEXT,
                    fact_text TEXT,
                    source_episode_uuid TEXT,
                    created_at TEXT,
                    valid_at TEXT,
                    invalid_at TEXT,
                    attributes TEXT,
                    embedding BLOB
                )
            """)
            
            # 개념 노드 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_nodes (
                    uuid TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    parent_concept_uuid TEXT,
                    embedding BLOB
                )
            """)
            
            conn.commit()
    
    def _save_to_db(self):
        """메모리 데이터를 데이터베이스에 저장"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 에피소드 저장
                for episode in self.episodes.values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_episodes 
                        (uuid, ego_uuid, timestamp, topic_summary, internal_state, embedding)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        episode.uuid,
                        episode.ego_uuid,
                        episode.timestamp.isoformat(),
                        episode.topic_summary,
                        json.dumps(episode.internal_state.model_dump()),
                        np.array(episode.embedding).tobytes() if episode.embedding else None
                    ))
                
                # 그래프 노드 저장
                for node in self.graph_nodes.values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO graph_nodes 
                        (uuid, name, labels, source_episode_uuid, concept_uuid, created_at, attributes, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        node.uuid,
                        node.name,
                        json.dumps(node.labels),
                        node.source_episode_uuid,
                        node.concept_uuid,
                        node.created_at.isoformat(),
                        json.dumps(node.attributes),
                        np.array(node.embedding).tobytes() if node.embedding else None
                    ))
                
                # 그래프 엣지 저장
                for edge in self.graph_edges.values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO graph_edges 
                        (uuid, source_uuid, target_uuid, label, fact_text, source_episode_uuid, 
                         created_at, valid_at, invalid_at, attributes, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        edge.uuid,
                        edge.source_uuid,
                        edge.target_uuid,
                        edge.label,
                        edge.fact_text,
                        edge.source_episode_uuid,
                        edge.created_at.isoformat(),
                        edge.valid_at.isoformat() if edge.valid_at else None,
                        edge.invalid_at.isoformat() if edge.invalid_at else None,
                        json.dumps(edge.attributes),
                        np.array(edge.embedding).tobytes() if edge.embedding else None
                    ))
                
                # 개념 노드 저장
                for concept in self.concept_nodes.values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO concept_nodes 
                        (uuid, name, description, parent_concept_uuid, embedding)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        concept.uuid,
                        concept.name,
                        concept.description,
                        concept.parent_concept_uuid,
                        np.array(concept.embedding).tobytes() if concept.embedding else None
                    ))
                
                conn.commit()
    
    def load_from_db(self):
        """데이터베이스에서 메모리 데이터 로드"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 에피소드 로드
            cursor.execute("SELECT * FROM memory_episodes")
            for row in cursor.fetchall():
                episode = MemoryEpisodeNode(
                    uuid=row[0],
                    ego_uuid=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    topic_summary=row[3],
                    internal_state=InternalState(**json.loads(row[4])),
                    embedding=np.frombuffer(row[5]).tolist() if row[5] else []
                )
                self.episodes[episode.uuid] = episode
            
            # 그래프 노드 로드
            cursor.execute("SELECT * FROM graph_nodes")
            for row in cursor.fetchall():
                node = GraphNode(
                    uuid=row[0],
                    name=row[1],
                    labels=json.loads(row[2]),
                    source_episode_uuid=row[3],
                    concept_uuid=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    attributes=json.loads(row[6]),
                    embedding=np.frombuffer(row[7]).tolist() if row[7] else []
                )
                self.graph_nodes[node.uuid] = node
            
            # 그래프 엣지 로드
            cursor.execute("SELECT * FROM graph_edges")
            for row in cursor.fetchall():
                edge = GraphEdge(
                    uuid=row[0],
                    source_uuid=row[1],
                    target_uuid=row[2],
                    label=row[3],
                    fact_text=row[4],
                    source_episode_uuid=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    valid_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    invalid_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    attributes=json.loads(row[9]),
                    embedding=np.frombuffer(row[10]).tolist() if row[10] else []
                )
                self.graph_edges[edge.uuid] = edge
            
            # 개념 노드 로드
            cursor.execute("SELECT * FROM concept_nodes")
            for row in cursor.fetchall():
                concept = ConceptNode(
                    uuid=row[0],
                    name=row[1],
                    description=row[2],
                    parent_concept_uuid=row[3],
                    embedding=np.frombuffer(row[4]).tolist() if row[4] else []
                )
                self.concept_nodes[concept.uuid] = concept
    
    def _initialize_ego_node(self):
        """자아 노드 초기화"""
        ego_node = GraphNode(
            uuid=EGO_NODE_UUID,
            name=EGO_NODE_NAME,
            labels=["AI", "Self"],
            source_episode_uuid="",
            created_at=datetime.now(timezone.utc)
        )
        self.graph_nodes[ego_node.uuid] = ego_node
    
    def _bootstrap_core_concepts(self):
        """핵심 개념들 부트스트랩"""
        core_concepts = [
            ("Person", "인간 개체", None),
            ("Location", "물리적 위치", None),
            ("Organization", "조직이나 기관", None),
            ("Event", "발생한 사건", None),
            ("Object", "물리적 객체", None),
            ("Concept", "추상적 개념", None),
            ("Action", "수행된 행동", None),
            ("Emotion", "감정 상태", None),
        ]
        
        for name, description, parent in core_concepts:
            concept = ConceptNode(
                name=name,
                description=description,
                parent_concept_uuid=parent
            )
            self.concept_nodes[concept.uuid] = concept
    
    def _rebuild_networkx_graph(self):
        """NetworkX 그래프 재구성"""
        self.networkx_graph.clear()
        
        # 노드 추가
        for node in self.graph_nodes.values():
            self.networkx_graph.add_node(node.uuid, **node.model_dump())
        
        # 엣지 추가
        for edge in self.graph_edges.values():
            if edge.invalid_at is None:  # 유효한 엣지만
                self.networkx_graph.add_edge(
                    edge.source_uuid,
                    edge.target_uuid,
                    label=edge.label,
                    fact_text=edge.fact_text,
                    **edge.model_dump()
                )
    
    def integrate_experience(self, text_from_interaction: str, ai_internal_state_text: str) -> Optional[MemoryEpisodeNode]:
        """상호작용 경험을 메모리 시스템에 통합"""
        try:
            # 에피소드 생성
            episode = self._create_memory_episode(text_from_interaction, ai_internal_state_text)
            if not episode:
                return None
            
            # 엔티티 추출
            entities = self._extract_entities_from_text(text_from_interaction)
            if not entities:
                return episode
            
            # 엔티티 해결 및 노드 생성
            resolved_nodes = {}
            for entity_data in entities.entities:
                node = self._resolve_entity_and_concept(entity_data, episode.uuid)
                if node:
                    resolved_nodes[entity_data.name] = node
            
            # 관계 추출 및 엣지 생성
            relationships = self._extract_relationships_from_text(text_from_interaction, list(resolved_nodes.values()))
            if relationships:
                for rel_data in relationships.relationships:
                    self._resolve_and_add_edge(rel_data, resolved_nodes, episode.uuid)
            
            # 그래프 업데이트
            self._rebuild_networkx_graph()
            
            # 데이터베이스 저장
            self._save_to_db()
            
            return episode
            
        except Exception as e:
            logger.error(f"경험 통합 중 오류: {e}")
            return None
    
    def _create_memory_episode(self, text: str, state_text: str) -> Optional[MemoryEpisodeNode]:
        """메모리 에피소드 생성"""
        try:
            # LLM을 사용하여 에피소드 요약 생성
            prompt = f"""
다음 상호작용을 간단히 요약해주세요 (한 문장으로):

상호작용: {text}
AI 상태: {state_text}

요약:
"""
            
            summary = self.llm_client.generate_text(prompt, max_tokens=50)
            
            # 내부 상태 분석
            state_prompt = f"""
다음 AI 상태를 분석하여 구조화된 형태로 변환해주세요:

{state_text}

다음 JSON 형태로 응답해주세요:
{{
    "emotional_valence": -1.0에서 1.0 사이의 값,
    "emotional_label": "주요 감정",
    "cognitive_process": "주요 인지 과정",
    "certainty": 0.0에서 1.0 사이의 값,
    "salience_focus": ["주요 관심 대상들"]
}}
"""
            
            state_response = self.llm_client.generate_text(state_prompt, max_tokens=200)
            try:
                state_data = json.loads(state_response)
                internal_state = InternalState(**state_data)
            except:
                internal_state = InternalState()
            
            # 임베딩 생성
            embedding = self.llm_client.get_embedding(text + " " + state_text)
            
            episode = MemoryEpisodeNode(
                topic_summary=summary.strip(),
                internal_state=internal_state,
                embedding=embedding
            )
            
            self.episodes[episode.uuid] = episode
            return episode
            
        except Exception as e:
            logger.error(f"에피소드 생성 중 오류: {e}")
            return None
    
    def _extract_entities_from_text(self, text: str) -> Optional[ExtractedEntities]:
        """텍스트에서 엔티티 추출"""
        try:
            prompt = f"""
다음 텍스트에서 중요한 엔티티들을 추출해주세요:

{text}

다음 JSON 형태로 응답해주세요:
{{
    "entities": [
        {{
            "name": "엔티티 이름",
            "label": "엔티티 타입 (Person, Location, Organization, Object, Concept 등)"
        }}
    ]
}}

중요한 엔티티만 추출하고, 일반적인 단어는 제외해주세요.
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=300)
            try:
                data = json.loads(response)
                return ExtractedEntities(**data)
            except:
                return None
                
        except Exception as e:
            logger.error(f"엔티티 추출 중 오류: {e}")
            return None
    
    def _find_existing_node(self, name: str, label: Optional[str] = None, threshold: float = 0.85) -> Optional[GraphNode]:
        """기존 노드 찾기 - 이름과 임베딩 유사도 기반"""
        if not name:
            return None
        
        # 이름 기반 정확 매칭
        for node in self.graph_nodes.values():
            if node.name.lower() == name.lower():
                return node
        
        # 임베딩 기반 유사도 매칭
        if name in self.graph_nodes:
            return self.graph_nodes[name]
        
        # 라벨 기반 필터링
        if label:
            candidates = [node for node in self.graph_nodes.values() if label in node.labels]
        else:
            candidates = list(self.graph_nodes.values())
        
        # 임베딩 유사도 계산 (간단한 구현)
        name_embedding = self.llm_client.get_embedding(name)
        best_match = None
        best_similarity = 0
        
        for node in candidates:
            if node.embedding:
                similarity = 1 - cosine(name_embedding, node.embedding)
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = node
        
        return best_match
    
    def _resolve_entity_and_concept(self, entity_data: ExtractedEntity, current_episode_uuid: str) -> Optional[GraphNode]:
        """엔티티 해결 및 개념 연결"""
        # 기존 노드 찾기
        existing_node = self._find_existing_node(entity_data.name, entity_data.label)
        
        if existing_node:
            # 기존 노드 업데이트
            existing_node.source_episode_uuid = current_episode_uuid
            return existing_node
        else:
            # 새 노드 생성
            embedding = self.llm_client.get_embedding(entity_data.name)
            
            node = GraphNode(
                name=entity_data.name,
                labels=[entity_data.label],
                source_episode_uuid=current_episode_uuid,
                embedding=embedding
            )
            
            # 개념 분류 및 연결
            self._classify_and_link_entity_to_concept(node)
            
            self.graph_nodes[node.uuid] = node
            return node
    
    def _classify_and_link_entity_to_concept(self, node_to_classify: GraphNode):
        """엔티티를 개념에 분류하고 연결"""
        try:
            prompt = f"""
다음 엔티티를 가장 적절한 개념에 분류해주세요:

엔티티: {node_to_classify.name}
라벨: {', '.join(node_to_classify.labels)}

사용 가능한 개념들:
{chr(10).join([f"- {concept.name}: {concept.description}" for concept in self.concept_nodes.values()])}

다음 JSON 형태로 응답해주세요:
{{
    "entity_name": "{node_to_classify.name}",
    "most_likely_concept_name": "가장 적절한 개념 이름",
    "reasoning": "분류 이유"
}}
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=200)
            try:
                data = json.loads(response)
                classifier = EntityClassifier(**data)
                
                # 개념 찾기
                target_concept = None
                for concept in self.concept_nodes.values():
                    if concept.name == classifier.most_likely_concept_name:
                        target_concept = concept
                        break
                
                if target_concept:
                    node_to_classify.concept_uuid = target_concept.uuid
                    
            except:
                pass
                
        except Exception as e:
            logger.error(f"엔티티 분류 중 오류: {e}")
    
    def _extract_relationships_from_text(self, text: str, resolved_nodes: List[GraphNode]) -> Optional[ExtractedRelationships]:
        """텍스트에서 관계 추출"""
        if not resolved_nodes:
            return None
        
        try:
            node_names = [node.name for node in resolved_nodes]
            
            prompt = f"""
다음 텍스트에서 엔티티들 간의 관계를 추출해주세요:

텍스트: {text}

엔티티들: {', '.join(node_names)}

다음 JSON 형태로 응답해주세요:
{{
    "relationships": [
        {{
            "source_name": "소스 엔티티 이름",
            "target_name": "타겟 엔티티 이름", 
            "label": "관계 타입 (예: works_at, is_located_in, knows)",
            "fact": "관계를 나타내는 문장"
        }}
    ]
}}

관계가 명확한 것만 추출해주세요.
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=400)
            try:
                data = json.loads(response)
                return ExtractedRelationships(**data)
            except:
                return None
                
        except Exception as e:
            logger.error(f"관계 추출 중 오류: {e}")
            return None
    
    def _resolve_and_add_edge(self, rel_data: ExtractedRelationship, resolved_nodes_map: Dict[str, GraphNode], current_episode_uuid: str):
        """관계 해결 및 엣지 추가"""
        source_node = resolved_nodes_map.get(rel_data.source_name)
        target_node = resolved_nodes_map.get(rel_data.target_name)
        
        if not source_node or not target_node:
            return
        
        # 모순 검사
        contradiction = self._check_contradiction(rel_data, source_node, target_node)
        if contradiction.is_contradictory:
            # 기존 엣지 무효화
            if contradiction.conflicting_edge_uuid:
                edge = self.graph_edges.get(contradiction.conflicting_edge_uuid)
                if edge:
                    edge.invalid_at = datetime.now(timezone.utc)
            return
        
        if contradiction.is_duplicate:
            return  # 중복이므로 추가하지 않음
        
        # 새 엣지 생성
        embedding = self.llm_client.get_embedding(rel_data.fact)
        
        edge = GraphEdge(
            source_uuid=source_node.uuid,
            target_uuid=target_node.uuid,
            label=rel_data.label,
            fact_text=rel_data.fact,
            source_episode_uuid=current_episode_uuid,
            embedding=embedding
        )
        
        self.graph_edges[edge.uuid] = edge
    
    def _check_contradiction(self, new_rel: ExtractedRelationship, source_node: GraphNode, target_node: GraphNode) -> ContradictionCheck:
        """새로운 관계가 기존 관계와 모순되는지 검사"""
        # 간단한 구현 - 실제로는 더 정교한 모순 검사 필요
        return ContradictionCheck(
            is_contradictory=False,
            is_duplicate=False,
            reasoning="기본 검사 통과"
        )
    
    def answer_question(self, question: str, top_k_facts: int = 10) -> str:
        """질문에 대한 답변 생성"""
        try:
            # 관련 항목 찾기
            relevant_items = self._find_relevant_items(question, top_k=top_k_facts)
            if not relevant_items:
                return "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다."
            
            # 사실들 추출
            facts = [item[1] for item in relevant_items]
            
            # 에피소드 추적
            episodes_map = self._trace_facts_to_episodes(facts)
            
            # 내러티브 컨텍스트 구성
            narrative_context = self._construct_narrative_context(episodes_map)
            
            # 최종 답변 생성
            final_answer = self._synthesize_final_answer(question, narrative_context)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"질문 답변 중 오류: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    def _find_relevant_items(self, question: str, top_k: int = 10, similarity_threshold: float = 0.7) -> List[Tuple[float, Union[GraphNode, GraphEdge]]]:
        """질문과 관련된 항목들 찾기"""
        question_embedding = self.llm_client.get_embedding(question)
        
        # 노드와 엣지 모두 검색
        all_items = list(self.graph_nodes.values()) + list(self.graph_edges.values())
        
        similarities = []
        for item in all_items:
            if item.embedding:
                similarity = 1 - cosine(question_embedding, item.embedding)
                if similarity > similarity_threshold:
                    similarities.append((similarity, item))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
    
    def _trace_facts_to_episodes(self, facts: List[Union[GraphNode, GraphEdge]]) -> Dict[str, List[Union[GraphNode, GraphEdge]]]:
        """사실들을 에피소드로 추적"""
        episodes_map = defaultdict(list)
        
        for fact in facts:
            if isinstance(fact, GraphNode):
                episode_uuid = fact.source_episode_uuid
            else:  # GraphEdge
                episode_uuid = fact.source_episode_uuid
            
            if episode_uuid:
                episodes_map[episode_uuid].append(fact)
        
        return dict(episodes_map)
    
    def _construct_narrative_context(self, episodes_map: Dict[str, List[Union[GraphNode, GraphEdge]]]) -> str:
        """내러티브 컨텍스트 구성"""
        context_parts = []
        
        for episode_uuid, facts in episodes_map.items():
            episode = self.episodes.get(episode_uuid)
            if episode:
                context_parts.append(f"에피소드: {episode.topic_summary}")
                context_parts.append(f"시간: {episode.timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                fact_texts = []
                for fact in facts:
                    if isinstance(fact, GraphNode):
                        fact_texts.append(f"- {fact.name} ({', '.join(fact.labels)})")
                    else:  # GraphEdge
                        fact_texts.append(f"- {fact.fact_text}")
                
                context_parts.extend(fact_texts)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _synthesize_final_answer(self, question: str, narrative_context: str) -> str:
        """최종 답변 생성"""
        prompt = f"""
다음 질문에 대해 제공된 컨텍스트를 바탕으로 답변해주세요:

질문: {question}

컨텍스트:
{narrative_context}

자연스럽고 유용한 답변을 제공해주세요.
"""
        
        return self.llm_client.generate_text(prompt, max_tokens=500)
    
    def close_db(self):
        """데이터베이스 연결 종료"""
        self._save_to_db() 