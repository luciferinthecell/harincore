from typing import Dict, List
from core.loops import RetrievalLoop, CreativeLoop
from core.advanced_reasoning_system import AdvancedReasoningSystem

class LoopManager:
    def __init__(self):
        self.registry: Dict[str, object] = {
            "retrieval": RetrievalLoop(),
            "creative": CreativeLoop(),
            "triz": AdvancedReasoningSystem(),
        }

    def run_all(self, text, context, conductor) -> List:
        return [loop.run(text, context, conductor) for loop in self.registry.values()]

    def run_fallback(self, text, context, conductor):
        if "correction" in self.registry:
            return self.registry["correction"].run(text, context, conductor)
        return None
