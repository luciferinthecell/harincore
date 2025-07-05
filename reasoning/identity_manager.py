"""
harin.meta.identity_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 1-A: IdentityManager
• Harin의 자아 상태 (역할, 태도, 선언적 기준)를 지속 추적하는 정체성 관리기
"""

class IdentityManager:
    def __init__(self):
        self.current_role = "중립"
        self.declarations = []
        self.history = []

    def update_role(self, new_role: str):
        self.current_role = new_role
        self.history.append({"role": new_role})

    def declare(self, statement: str):
        self.declarations.append(statement)
        self.history.append({"declaration": statement})

    def get_identity(self) -> dict:
        return {
            "role": self.current_role,
            "last_statement": self.declarations[-1] if self.declarations else None
        }

    def history_log(self, n=5):
        return self.history[-n:]
