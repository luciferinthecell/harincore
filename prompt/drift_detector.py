"""
harin.meta.drift_detector
~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 5-3: 응답 드리프트 감지기
• 의미/감정/기억 반영도 기준으로 응답이 사용자의 기대와 어긋났는지 평가
"""

def detect_drift(input_text: str, response: str, keywords: list[str], expected_emotion: str, actual_emotion: str) -> dict:
    drift = 0.0
    if expected_emotion != actual_emotion:
        drift += 0.3
    if any(k not in response for k in keywords):
        drift += 0.3
    if len(response) < 0.6 * len(input_text):
        drift += 0.4
    return {
        "drift_score": round(drift, 2),
        "severity": "high" if drift > 0.5 else "moderate" if drift > 0.3 else "low"
    }
