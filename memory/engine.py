"""
MemoryEngine 2.0  –  semantic, time-aware vector memory for Harin
══════════════════════════════════════════════════════════════════
Backend-agnostic: pass any embed-func + vector-db (Faiss, Qdrant…).
Adds:
  • time_decay(t)            – older memories fade unless high trust
  • trust_decay(score)       – low-scored items fade faster
  • recall()                 – weighted k-NN + diversity filter
  • data_memory_integration  – JSONL 파일 기반 메모리 참조
"""

from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Any
from memory.data_memory_manager import DataMemoryManager
from memory.data_memory_manager import MemoryNode

SECONDS_PER_DAY = 86_400


# ───────────────────────────────────────────────────────────────────
@dataclass
class MemoryItem:
    id: str
    text: str
    vector: List[float]
    meta: Dict
    ts: float = field(default_factory=time.time)   # unix
    trust: float = 0.7                             # 0‥1


# ───────────────────────────────────────────────────────────────────
class MemoryEngine:
    """
    embed_fn : Callable[[str], List[float]]
    vectordb : Any object with `.add(id, vector)` and `.search(vector, k)`
               (returns List[Tuple[id, score]])
    decay_cfg: dict – tweak γ (time) and β (trust)
    """

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        vectordb=None,
        *,
        decay_cfg: Dict = None,
        enable_data_memory: bool = True,
    ) -> None:
        self.embed = embed_fn
        self.db = vectordb
        self.items: Dict[str, MemoryItem] = {}

        self.γ = (decay_cfg or {}).get("time_gamma", 0.03)   # per-day
        self.β = (decay_cfg or {}).get("trust_beta", 0.4)
        
        # 데이터 메모리 매니저 초기화
        self.data_memory = None
        if enable_data_memory:
            try:
                self.data_memory = DataMemoryManager()
            except Exception as e:
                print(f"데이터 메모리 매니저 초기화 실패: {e}")

    # ========================================================= #
    #  WRITE
    # ========================================================= #
    def add(self, text: str, meta: Dict | None = None, trust: float = 0.7) -> str:
        vec = self.embed(text)
        mid = f"M{len(self.items)+1:06d}"
        item = MemoryItem(id=mid, text=text, vector=vec, meta=meta or {}, trust=trust)
        self.items[mid] = item
        if self.db:
            self.db.add(mid, vec)
        return mid

    # ========================================================= #
    #  READ
    # ========================================================= #
    def recall(self, query: str, *, k: int = 6, diversity: int = 4) -> List[MemoryItem]:
        qv = self.embed(query)
        if self.db:
            hits = self.db.search(qv, k=16)  # (id, cosine)
        else:
            hits = []  # 기본값
        
        scored: List[Tuple[float, MemoryItem]] = []

        for mid, sim in hits:
            if mid in self.items:
                itm = self.items[mid]
                weight = sim * self._time_decay(itm.ts) * self._trust_decay(itm.trust)
                scored.append((weight, itm))

        # sort and diversity pick
        picked: List[MemoryItem] = []
        for _, itm in sorted(scored, key=lambda x: x[0], reverse=True):
            if not self._redundant(itm, picked):
                picked.append(itm)
            if len(picked) >= k or len(picked) >= diversity:
                break
        return picked

    # ========================================================= #
    #  DATA MEMORY INTEGRATION
    # ========================================================= #
    def get_data_memory_context(self, query: str, loop_id: str = None) -> Dict[str, Any]:
        """데이터 메모리에서 사고 컨텍스트 조회"""
        if not self.data_memory:
            return {"error": "데이터 메모리 매니저가 초기화되지 않음"}
        
        return self.data_memory.get_memory_context_for_thinking(query, loop_id)
    
    def search_data_memories(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """데이터 메모리 검색"""
        if not self.data_memory:
            return []
        
        return self.data_memory.search_memories(query, top_k)
    
    def get_memory_by_node_id(self, node_id: str) -> Optional[Any]:
        """노드 ID로 메모리 조회"""
        if not self.data_memory:
            return None
        
        return self.data_memory.get_memory_by_id(node_id)
    
    def get_memories_by_loop(self, loop_id: str) -> List[Any]:
        """루프 ID로 관련 메모리 조회"""
        if not self.data_memory:
            return []
        
        return self.data_memory.get_memories_by_loop(loop_id)
    
    def get_memories_by_tag(self, tag: str) -> List[Any]:
        """태그로 메모리 조회"""
        if not self.data_memory:
            return []
        
        return self.data_memory.get_memories_by_tag(tag)
    
    def add_data_memory_node(self, content: str, memory_type: str = "memory", 
                           tags: List[str] = None, context: Dict[str, Any] = None,
                           source_file: str = "h2") -> Optional[str]:
        """새 데이터 메모리 노드 추가"""
        if not self.data_memory:
            return None
        
        node = MemoryNode(
            id="",
            content=content,
            type=memory_type,
            tags=tags or [],
            context=context or {},
            importance=context.get("importance", 0.5) if context else 0.5
        )
        
        self.data_memory.add_memory_node(node, source_file)
        return node.id
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        stats = {
            "vector_memories": len(self.items),
            "data_memories": 0
        }
        
        if self.data_memory:
            data_stats = self.data_memory.get_memory_stats()
            stats["data_memories"] = data_stats["total_nodes"]
            stats["data_files"] = data_stats["files"]
            stats["data_types"] = data_stats["types"]
            stats["data_loops"] = data_stats["loops"]
        
        return stats

    # ========================================================= #
    #  DECAY
    # ========================================================= #
    def _time_decay(self, ts: float) -> float:
        age_days = (time.time() - ts) / SECONDS_PER_DAY
        return math.exp(-self.γ * age_days)

    def _trust_decay(self, trust: float) -> float:
        return math.exp(self.β * (trust - 1))  # high trust → ~1.0, low trust → down-weight

    # ========================================================= #
    #  UTIL
    # ========================================================= #
    def _redundant(self, itm: MemoryItem, pool: List[MemoryItem]) -> bool:
        dup_kw = set(itm.text.lower().split()[:4])
        for p in pool:
            if dup_kw & set(p.text.lower().split()[:4]):
                return True
        return False 
