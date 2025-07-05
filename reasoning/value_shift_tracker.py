"""
harin.meta.value_shift_tracker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 1-B: ValueShiftTracker
• 판단 흐름 속 Harin의 가치 경향 및 판단 기준 변화 기록기
"""

class ValueShiftTracker:
    def __init__(self):
        self.log = []

    def log_shift(self, description: str, related_path: str = None):
        self.log.append({
            "desc": description,
            "related": related_path
        })

    def recent_shifts(self, n=5):
        return self.log[-n:]
