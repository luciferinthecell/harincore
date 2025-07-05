"""
harin.session.harin_session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HarinSession: 시스템 전체 사고 흐름을 관통하는 상태 지속 객체
- 감정 흐름
- 리듬 (truth / resonance / responsibility)
- 기억 호출 결과
- scar 기록
- 사용자 피드백
- 루프 흐름
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class HarinSession:
    emotion: str = "중립"
    rhythm: Dict[str, float] = field(default_factory=lambda: {
        "truth": 0.7, "resonance": 0.7, "responsibility": 0.7
    })
    memory_trace: List[Dict[str, Any]] = field(default_factory=list)
    scar_log: List[str] = field(default_factory=list)
    feedback_log: List[str] = field(default_factory=list)
    loop_history: List[str] = field(default_factory=list)

    def update_emotion(self, new_emotion: str):
        self.emotion = new_emotion
        self.feedback_log.append(f"emotion:{new_emotion}")

    def update_rhythm(self, truth: float = None, resonance: float = None, responsibility: float = None):
        if truth is not None: self.rhythm["truth"] = truth
        if resonance is not None: self.rhythm["resonance"] = resonance
        if responsibility is not None: self.rhythm["responsibility"] = responsibility

    def log_memory(self, memory: Dict[str, Any]):
        self.memory_trace.append(memory)

    def add_scar(self, path_id: str):
        self.scar_log.append(path_id)

    def add_feedback(self, signal: str):
        self.feedback_log.append(signal)

    def record_loop(self, loop_label: str):
        self.loop_history.append(loop_label)
