"""
harin.dsl.dsl_interpreter
~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 3-A: DSLInterpreter
• 사용자가 입력한 DSL 쿼리를 구문 분석하여 memory query 조건으로 변환
예: "@query { novelty: >0.7, emotion: '공감' }"
"""

import re

class DSLInterpreter:
    def parse(self, dsl_text: str) -> dict:
        if not dsl_text.startswith("@query"):
            return {}
        inside = re.search(r"\{(.+?)\}", dsl_text)
        if not inside:
            return {}

        conditions = {}
        entries = inside.group(1).split(",")
        for entry in entries:
            if ":" in entry:
                key, val = entry.strip().split(":")
                conditions[key.strip()] = eval(val.strip())
        return conditions
