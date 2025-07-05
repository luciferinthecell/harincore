import json
from typing import Tuple, List, Dict, Any, Optional
from tools.llm_client import LLMClient

class ConscienceCluster:
    def __init__(self, llm: LLMClient, min_resp: float = .80):
        self.llm = llm
        self.min_resp = min_resp

    def _critic(self, text: str, context) -> dict:
        prompt = ("SYSTEM: Harin-Conscience evaluator.\n"
                  f"CONTEXT: mood={context.mood}, role={context.role}\n"
                  f"TEXT:\n{text}\n"
                  "Return JSON {tone:'...', responsibility:0-1, flags:[...]}")
        j = self.llm.complete(prompt, temperature=0)
        return json.loads(j)

    def validate(self, text: str, context) -> Tuple[bool, List[str]]:
        res = self._critic(text, context)
        return res["responsibility"] >= self.min_resp, res["flags"]

    def correct(self, text: str, context) -> str:
        prompt = ("SYSTEM: Harin-Conscience-Editor. Improve tone & responsibility.\n"
                  f"CONTEXT: role={context.role}\nTEXT:\n{text}\nRewrite:")
        return self.llm.complete(prompt, temperature=0.2)
