"""
harin.memory.data_memory_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

??? ??? ?? ???
- h.json, h1.jsonl, h2.jsonl ?? ??
- ?? ?? ?? ? ??
- ?? ? JSONL ?? ??
- ??? ??? ? ??
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from memory.palantirgraph import PalantirGraph, ThoughtNode


@dataclass
class MemoryNode:
    """??? ??"""
    id: str
    content: str
    type: str = "memory"
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    vectors: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    author: str = "harin"
    linked_nodes: List[str] = field(default_factory=list)
    importance: float = 0.5


@dataclass
class MemoryFile:
    """??? ?? ??"""
    name: str
    path: Path
    description: str
    node_count: int
    linked_loops: List[str]
    formats: List[str]


class DataMemoryManager:
    """??? ??? ?? ???"""
    
    def __init__(self, data_dir: str = "memory/data"):
        self.data_dir = Path(data_dir)
        self.memory_files = {}
        self.node_registry = {}
        self.loop_bindings = {}
        
        # ??? ?? ???
        self._initialize_memory_files()
        self._load_all_memory_data()
        self._build_node_registry()
        self._build_loop_bindings()
    
    def _initialize_memory_files(self):
        """??? ?? ???"""
        self.memory_files = {
            "h": MemoryFile(
                name="h",
                path=self.data_dir / "h.json",
                description="???? ? ?? ??",
                node_count=0,
                linked_loops=[],
                formats=["protocol", "structure"]
            ),
            "h1": MemoryFile(
                name="h1", 
                path=self.data_dir / "h1.jsonl",
                description="?? ?? ??/??/scar ?? ?? ?? ???",
                node_count=0,
                linked_loops=["loop_017", "simulate", "scar_eval"],
                formats=["scar", "structure", "triggered_event"]
            ),
            "h2": MemoryFile(
                name="h2",
                path=self.data_dir / "h2.jsonl", 
                description="?? ?? ?? ??/?? ?? ?? ?? ?? ???",
                node_count=0,
                linked_loops=["loop_030", "loop_027", "loop_017"],
                formats=["structure", "state", "reaction_pattern", "thought_trace"]
            )
        }
    
    def _load_all_memory_data(self):
        """?? ??? ??? ??"""
        # h.json ???? ??
        if self.memory_files["h"].path.exists():
            with open(self.memory_files["h"].path, 'r', encoding='utf-8') as f:
                protocol_data = json.load(f)
                self._process_protocol_data(protocol_data)
        
        # h1.jsonl ??
        if self.memory_files["h1"].path.exists():
            self._load_jsonl_file("h1")
        
        # h2.jsonl ??
        if self.memory_files["h2"].path.exists():
            self._load_jsonl_file("h2")
    
    def _process_protocol_data(self, protocol_data: Dict[str, Any]):
        """???? ??? ??"""
        # ?? ??? ?? ??
        if "loop_bindings" in protocol_data:
            self.loop_bindings = protocol_data["loop_bindings"]
        
        # ??? ?? ????
        if "index" in protocol_data:
            for file_name, index_info in protocol_data["index"].items():
                if file_name in self.memory_files:
                    self.memory_files[file_name].description = index_info.get("description", "")
                    self.memory_files[file_name].linked_loops = index_info.get("linked_loops", [])
                    self.memory_files[file_name].formats = index_info.get("formats", [])
                    self.memory_files[file_name].node_count = len(index_info.get("nodes", []))
    
    def _load_jsonl_file(self, file_name: str):
        """JSONL ?? ??"""
        file_path = self.memory_files[file_name].path
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    node_data = json.loads(line.strip())
                    memory_node = self._create_memory_node(node_data, file_name)
                    self.node_registry[memory_node.id] = memory_node
                except json.JSONDecodeError as e:
                    print(f"JSON ?? ?? {file_name}:{line_num}: {e}")
    
    def _create_memory_node(self, node_data: Dict[str, Any], source_file: str) -> MemoryNode:
        """??? ?? ??"""
        return MemoryNode(
            id=node_data.get("id", f"node_{uuid.uuid4().hex[:8]}"),
            content=node_data.get("content", ""),
            type=node_data.get("type", "memory"),
            tags=node_data.get("tags", []),
            context=node_data.get("context", {}),
            vectors=node_data.get("vectors", {}),
            meta=node_data.get("meta", {}),
            created_at=node_data.get("created_at", datetime.now().isoformat()),
            author=node_data.get("author", "harin"),
            linked_nodes=node_data.get("linked_nodes", []),
            importance=node_data.get("context", {}).get("importance", 0.5)
        )
    
    def _build_node_registry(self):
        """?? ????? ??"""
        # ?? ? ?? ?? ??
        for node_id, node in self.node_registry.items():
            # ?? ?? ??
            for tag in node.tags:
                if tag.startswith("loop."):
                    loop_id = tag.split(".")[1]
                    if loop_id not in self.loop_bindings:
                        self.loop_bindings[loop_id] = []
                    if node_id not in self.loop_bindings[loop_id]:
                        self.loop_bindings[loop_id].append(node_id)
    
    def _build_loop_bindings(self):
        """?? ??? ??"""
        # ?? ?? ???? ?? ????? ??
        for loop_id, node_ids in self.loop_bindings.items():
            for node_id in node_ids:
                if node_id in self.node_registry:
                    node = self.node_registry[node_id]
                    if loop_id not in node.meta.get("linked_loops", []):
                        if "linked_loops" not in node.meta:
                            node.meta["linked_loops"] = []
                        node.meta["linked_loops"].append(loop_id)
    
    def get_memory_by_id(self, node_id: str) -> Optional[MemoryNode]:
        """?? ID? ??? ??"""
        return self.node_registry.get(node_id)
    
    def get_memories_by_loop(self, loop_id: str) -> List[MemoryNode]:
        """?? ID? ?? ??? ??"""
        memories = []
        for node_id, node in self.node_registry.items():
            if loop_id in node.meta.get("linked_loops", []):
                memories.append(node)
        return memories
    
    def get_memories_by_tag(self, tag: str) -> List[MemoryNode]:
        """??? ??? ??"""
        memories = []
        for node_id, node in self.node_registry.items():
            if tag in node.tags:
                memories.append(node)
        return memories
    
    def get_memories_by_type(self, memory_type: str) -> List[MemoryNode]:
        """???? ??? ??"""
        memories = []
        for node_id, node in self.node_registry.items():
            if node.type == memory_type:
                memories.append(node)
        return memories
    
    def search_memories(self, query: str, top_k: int = 5) -> List[Tuple[MemoryNode, float]]:
        """??? ??"""
        results = []
        
        for node_id, node in self.node_registry.items():
            # ??? ??? ?? (???? ? ??? ?? ??)
            score = self._calculate_search_score(query, node)
            if score > 0.1:  # ?? ???
                results.append((node, score))
        
        # ??? ??
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _calculate_search_score(self, query: str, node: MemoryNode) -> float:
        """?? ?? ??"""
        score = 0.0
        query_lower = query.lower()
        
        # ?? ??
        if query_lower in node.content.lower():
            score += 0.5
        
        # ?? ??
        for tag in node.tags:
            if query_lower in tag.lower():
                score += 0.3
        
        # ???? ??
        context_str = str(node.context).lower()
        if query_lower in context_str:
            score += 0.2
        
        # ??? ???
        score *= node.importance
        
        return score
    
    def get_related_memories(self, node_id: str, max_related: int = 3) -> List[MemoryNode]:
        """?? ??? ??"""
        if node_id not in self.node_registry:
            return []
        
        source_node = self.node_registry[node_id]
        related = []
        
        # ?? ??? ????
        for loop_id in source_node.meta.get("linked_loops", []):
            loop_memories = self.get_memories_by_loop(loop_id)
            for memory in loop_memories:
                if memory.id != node_id and memory not in related:
                    related.append(memory)
        
        # ?? ??? ????
        for tag in source_node.tags:
            tag_memories = self.get_memories_by_tag(tag)
            for memory in tag_memories:
                if memory.id != node_id and memory not in related:
                    related.append(memory)
        
        return related[:max_related]
    
    def get_memory_context_for_thinking(self, query: str, loop_id: str = None) -> Dict[str, Any]:
        """??? ?? ??? ???? ??"""
        context = {
            "relevant_memories": [],
            "loop_memories": [],
            "scar_memories": [],
            "structure_memories": [],
            "context_summary": ""
        }
        
        # ?? ??? ??
        search_results = self.search_memories(query, top_k=5)
        context["relevant_memories"] = [
            {
                "id": node.id,
                "content": node.content,
                "type": node.type,
                "importance": node.importance,
                "tags": node.tags
            }
            for node, score in search_results
        ]
        
        # ??? ???
        if loop_id:
            loop_memories = self.get_memories_by_loop(loop_id)
            context["loop_memories"] = [
                {
                    "id": node.id,
                    "content": node.content,
                    "type": node.type,
                    "importance": node.importance
                }
                for node in loop_memories
            ]
        
        # SCAR ???? (?? ??)
        scar_memories = self.get_memories_by_tag("scar")
        context["scar_memories"] = [
            {
                "id": node.id,
                "content": node.content,
                "type": node.type
            }
            for node in scar_memories[:3]  # ?? 3??
        ]
        
        # ?? ????
        structure_memories = self.get_memories_by_type("structure")
        context["structure_memories"] = [
            {
                "id": node.id,
                "content": node.content,
                "importance": node.importance
            }
            for node in structure_memories[:3]  # ?? 3??
        ]
        
        # ???? ?? ??
        context["context_summary"] = self._generate_context_summary(context)
        
        return context
    
    def _generate_context_summary(self, context: Dict[str, Any]) -> str:
        """???? ?? ??"""
        summary_parts = []
        
        if context["relevant_memories"]:
            summary_parts.append(f"?? ??: {len(context['relevant_memories'])}?")
        
        if context["loop_memories"]:
            summary_parts.append(f"?? ??: {len(context['loop_memories'])}?")
        
        if context["scar_memories"]:
            summary_parts.append(f"SCAR ??: {len(context['scar_memories'])}?")
        
        if context["structure_memories"]:
            summary_parts.append(f"?? ??: {len(context['structure_memories'])}?")
        
        return " | ".join(summary_parts) if summary_parts else "?? ?? ??"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """??? ?? ??"""
        stats = {
            "total_nodes": len(self.node_registry),
            "files": {},
            "types": {},
            "loops": {}
        }
        
        # ??? ??
        for file_name, file_info in self.memory_files.items():
            file_nodes = [node for node in self.node_registry.values() 
                         if node.meta.get("source_file") == file_name]
            stats["files"][file_name] = {
                "description": file_info.description,
                "node_count": len(file_nodes),
                "linked_loops": file_info.linked_loops
            }
        
        # ??? ??
        for node in self.node_registry.values():
            node_type = node.type
            if node_type not in stats["types"]:
                stats["types"][node_type] = 0
            stats["types"][node_type] += 1
        
        # ??? ??
        for loop_id, node_ids in self.loop_bindings.items():
            stats["loops"][loop_id] = len(node_ids)
        
        return stats
    
    def add_memory_node(self, node: MemoryNode, source_file: str = "h2"):
        """? ??? ?? ??"""
        # ?? ID ??
        if not node.id:
            node.id = f"{source_file}_node_{uuid.uuid4().hex[:8]}"
        
        # ????? ????
        node.meta["source_file"] = source_file
        node.meta["created_at"] = datetime.now().isoformat()
        
        # ?????? ??
        self.node_registry[node.id] = node
        
        # JSONL ??? ??
        self._save_node_to_file(node, source_file)
    
    def _save_node_to_file(self, node: MemoryNode, source_file: str):
        """??? ??? ??"""
        file_path = self.memory_files[source_file].path
        
        # JSONL ???? ??
        node_data = {
            "id": node.id,
            "content": node.content,
            "type": node.type,
            "tags": node.tags,
            "context": node.context,
            "vectors": node.vectors,
            "meta": node.meta,
            "created_at": node.created_at,
            "author": node.author,
            "linked_nodes": node.linked_nodes
        }
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node_data, ensure_ascii=False) + '\n')
