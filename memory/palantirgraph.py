"""
palantirgrapph.py  (refactored v2)
──────────────────────────────────
Graph‑based long‑term memory for Harin Agent **+ meta‑aware best‑path search**

🆕  핵심 추가
• `best_path_meta(meta_filter: dict, universe="U0", limit=10)`
  – 노드 `meta` 딕셔너리를 조건으로 필터링해 p×importance 상위 경로 반환

기존 기능(노드·엣지 CRUD, cosine, traverse, JSON persistence) 유지.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ────────────────────────────────────────────────────────────────────────────
#  Low‑level structures
# ────────────────────────────────────────────────────────────────────────────

def _now() -> float:
    return time.time()


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class ThoughtNode:
    id: str
    content: str
    node_type: str
    vectors: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=_now)

    @classmethod
    def create(
        cls,
        content: str,
        node_type: str = "thought",
        vectors: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ThoughtNode":
        return cls(
            id=_gen_id("N"),
            content=content,
            node_type=node_type,
            vectors=vectors or {},
            meta=meta or {},
        )


@dataclass
class Relationship:
    id: str
    source: str
    target: str
    predicate: str
    weight: float = 1.0
    created: float = field(default_factory=_now)

    @classmethod
    def create(
        cls,
        source: str,
        target: str,
        predicate: str,
        weight: float = 1.0,
    ) -> "Relationship":
        return cls(_gen_id("E"), source, target, predicate, weight)


# ────────────────────────────────────────────────────────────────────────────
#  Graph container
# ────────────────────────────────────────────────────────────────────────────

class PalantirGraph:
    def __init__(self, persist_path: Path | str | None = None) -> None:
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: Dict[str, Relationship] = {}
        self.persist_path = Path(persist_path or "memory/data/palantir_graph.json")
        if self.persist_path.exists():
            self.load()

    # ▸ CRUD --------------------------------------------------------------
    def add_node(self, node: ThoughtNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Relationship) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Edge references unknown node(s)")
        self.edges[edge.id] = edge

    def upsert(
        self,
        content: str,
        node_type: str = "thought",
        vectors: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ThoughtNode:
        for n in self.nodes.values():
            if n.content == content:
                return n
        node = ThoughtNode.create(content, node_type, vectors, meta)
        self.add_node(node)
        return node

    # ▸ Similarity & traversal -------------------------------------------
    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        num = sum(a[k] * b[k] for k in keys)
        den = math.sqrt(sum(a[k] ** 2 for k in keys)) * math.sqrt(sum(b[k] ** 2 for k in keys))
        return num / (den + 1e-9)

    def find_similar(
        self,
        probe: Dict[str, float] | str,
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> List[ThoughtNode]:
        scored: List[tuple[float, ThoughtNode]] = []
        if isinstance(probe, dict):
            for n in self.nodes.values():
                sim = self._cosine(n.vectors, probe)
                if sim >= min_score:
                    scored.append((sim, n))
        else:
            kw = probe.lower()
            for n in self.nodes.values():
                if kw in n.content.lower():
                    scored.append((1.0, n))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    def traverse(
        self,
        start_id: str,
        predicate_filter: Optional[str] = None,
        depth: int = 2,
    ) -> List[ThoughtNode]:
        visited, result = set(), []

        def _dfs(nid: str, d: int):
            if d < 0 or nid in visited:
                return
            visited.add(nid)
            result.append(self.nodes[nid])
            for e in self.out_edges(nid):
                if predicate_filter and e.predicate != predicate_filter:
                    continue
                _dfs(e.target, d - 1)

        _dfs(start_id, depth)
        return result

    def out_edges(self, nid: str) -> Iterable[Relationship]:
        return (e for e in self.edges.values() if e.source == nid)

    # ▸ NEW: meta‑aware best‑path ----------------------------------------
    def best_path_meta(
        self,
        meta_filter: Optional[Dict[str, Any]] = None,
        *,
        limit: int = 10,
    ) -> List[ThoughtNode]:
        """Return top‑ranked nodes whose meta matches all key/value in *meta_filter*."""
        meta_filter = meta_filter or {}

        def _match(node: ThoughtNode) -> bool:
            for k, v in meta_filter.items():
                if node.meta.get(k) != v:
                    return False
            return True

        candidates = [n for n in self.nodes.values() if _match(n)]
        ranked = sorted(candidates, key=lambda n: -(n.vectors.get("M", 0.5) * 1.0))  # simple importance proxy
        return ranked[:limit]

    # ▸ Persistence -------------------------------------------------------
    def save(self) -> None:
        data = {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }
        with self.persist_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        with self.persist_path.open(encoding="utf-8") as f:
            raw = json.load(f)
        self.nodes = {n["id"]: ThoughtNode(**n) for n in raw.get("nodes", [])}
        self.edges = {e["id"]: Relationship(**e) for e in raw.get("edges", [])}

    def load_jsonl(self, file_path: str | Path) -> None:
        """JSONL 파일에서 노드 데이터 로드"""
        file_path = Path(file_path)
        if not file_path.exists():
            return
        
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "id" in data and "content" in data:
                        # 기존 데이터 형식 호환성 처리
                        if "type" in data and "node_type" not in data:
                            data["node_type"] = data.pop("type")
                        
                        # vector -> vectors 변환
                        if "vector" in data and "vectors" not in data:
                            data["vectors"] = data.pop("vector")
                        
                        # emotion을 meta로 이동
                        if "emotion" in data:
                            if "meta" not in data:
                                data["meta"] = {}
                            data["meta"]["emotion"] = data.pop("emotion")
                        
                        node = ThoughtNode(**data)
                        self.nodes[node.id] = node
                except json.JSONDecodeError:
                    continue

    def load_edges(self, file_path: str | Path) -> None:
        """JSONL 파일에서 엣지 데이터 로드"""
        file_path = Path(file_path)
        if not file_path.exists():
            return
        
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "source" in data and "target" in data:
                        edge = Relationship(**data)
                        # 노드가 존재하는지 확인
                        if edge.source in self.nodes and edge.target in self.nodes:
                            self.edges[edge.id] = edge
                except json.JSONDecodeError:
                    continue

    def __del__(self):
        try:
            self.save()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
#  Optional external persistence helper
# ────────────────────────────────────────────────────────────────────────────

class GraphPersistence:
    @staticmethod
    def save(graph: "PalantirGraph", path: str | Path) -> None:
        path = Path(path)
        data = {
            "nodes": [asdict(n) for n in graph.nodes.values()],
            "edges": [asdict(e) for e in graph.edges.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str | Path) -> "PalantirGraph":
        return PalantirGraph(persist_path=path)


# ────────────────────────────────────────────────────────────────────────────
#  Harin Memory Integration
# ────────────────────────────────────────────────────────────────────────────

def initialize_memory():
    """Harin 메모리 시스템 초기화"""
    g = PalantirGraph("memory/data/palantir_graph.json")
    
    # 기존 메모리 데이터 로드
    memory_files = [
        "memory/data/harin_v6_formatted_memory.jsonl",
        "memory/data/harin_v6_summary_nodes.jsonl"
    ]
    
    edge_files = [
        "memory/data/harin_v6_memory_edges.jsonl",
        "memory/data/harin_v6_summary_edges.jsonl"
    ]
    
    for file_path in memory_files:
        if Path(file_path).exists():
            try:
                g.load_jsonl(file_path)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
    
    for file_path in edge_files:
        if Path(file_path).exists():
            try:
                g.load_edges(file_path)
            except Exception as e:
                print(f"Warning: Could not load edges from {file_path}: {e}")
    
    g.save()
    return g
