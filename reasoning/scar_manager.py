"""
harin.meta.scar_manager
~~~~~~~~~~~~~~~~~~~~~~~~

Phase 4-3: 판단 실패 기록기
• 실패한 경로 ID와 이유를 저장하고, 이후 판단 루프에서 회피 조건으로 사용
"""

scar_log = []

def record_scar(path_id: str, reason: str):
    scar_log.append({"path": path_id, "reason": reason})

def should_avoid(path_id: str) -> bool:
    return any(s['path'] == path_id for s in scar_log)

def recent_scars(n: int = 3) -> list[dict]:
    return scar_log[-n:]
