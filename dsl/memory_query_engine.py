"""
harin.dsl.memory_query_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 3-B: MemoryQueryEngine
• DSLInterpreter에서 추출된 조건에 따라 Palantir memory 노드 필터링
• memory_node = { id, content, novelty, emotion, tags }
"""

class MemoryQueryEngine:
    def __init__(self, memory_nodes: list[dict]):
        self.memory = memory_nodes

    def query(self, conditions: dict) -> list[dict]:
        result = []
        for node in self.memory:
            match = True
            for k, v in conditions.items():
                if k not in node:
                    match = False
                elif isinstance(v, (int, float)) and node[k] < v:
                    match = False
                elif isinstance(v, str) and v not in node[k]:
                    match = False
            if match:
                result.append(node)
        return result
