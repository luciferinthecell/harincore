from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict

@dataclass
class Interaction:
    utterance: str
    timestamp: str
    meta: Dict

@dataclass
class UserModel:
    uid: str = "anonymous"
    knowledge_level: str = "unknown"
    preferred_depth: str = "balanced"
    emotional_tone: str = "neutral"
    purpose: str = "general"
    history: List[Interaction] = field(default_factory=list)

    def update_from_input(self, text: str) -> None:
        self.history.append(Interaction(text, datetime.utcnow().isoformat(), meta={}))
        if len(text.split()) > 50:
            self.preferred_depth = "deep"
        if any(w in text.lower() for w in ["prove", "theorem", "complexity"]):
            self.knowledge_level = "expert"
        elif any(w in text.lower() for w in ["easy", "beginner"]):
            self.knowledge_level = "novice"

    def snapshot(self) -> Dict:
        return {
            "knowledge_level": self.knowledge_level,
            "preferred_depth": self.preferred_depth,
            "emotional_tone": self.emotional_tone,
            "history_size": len(self.history)
        }


# === HarinMind Integration ===
# 기존 harinmind 의존성 제거 - Red Team 도구와는 다른 목적
# from harinmind.intent_anchor import IntentAnchor
# from harinmind.live_sync_monitor import LiveSyncMonitor

# 대체 구현: 현재 Harin Core System에 맞는 클래스들
class IntentAnalyzer:
    """사용자 의도 분석 및 앵커링 (AI 사고 시스템용)"""
    def __init__(self):
        self.intent_cache = {}
        self.context_history = []
    
    def analyze_intent(self, user_input: str, context: dict = None) -> dict:
        """사용자 입력에서 의도 추출 및 분석"""
        # TODO: 실제 의도 분석 로직 구현
        return {
            "primary_intent": "general_query",
            "confidence": 0.8,
            "context_aware": True
        }

class LiveSyncMonitor:
    """실시간 동기화 모니터링 (AI 시스템 상태용)"""
    def __init__(self):
        self.sync_status = "active"
        self.last_sync = None
    
    def monitor_sync(self) -> dict:
        """시스템 동기화 상태 모니터링"""
        # TODO: 실제 동기화 모니터링 로직 구현
        return {
            "status": "synced",
            "last_update": "2024-01-01T00:00:00Z"
        }

# 기존 코드와의 호환성을 위한 인스턴스 생성
anchor = IntentAnalyzer()
monitor = LiveSyncMonitor()
