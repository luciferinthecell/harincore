"""
harin.output.response_synthesizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 5-2: 출력 생성기
• 외부 모델 응답 + 내부 메타 정보를 종합하여 구조화된 응답 반환
"""

def synthesize_response(model_output: str, path: dict, meta: dict, identity: str) -> dict:
    return {
        "output": model_output.strip(),
        "meta": {
            "path_id": path['id'],
            "decision": meta['decision'],
            "score": meta['score'],
            "identity": identity
        }
    }
