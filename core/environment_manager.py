"""
하린코어 환경 관리자
PM의 Shell 시스템을 참고하여 하린코어에 맞게 구현한 환경 관리 시스템
"""

import time
import threading
import queue
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import StrEnum
from pydantic import BaseModel, Field

from memory.models import Stimulus, StimulusType, NeedsAxesModel
from core.enhanced_main_loop import EnhancedHarinMainLoop


class TimeOfDay(StrEnum):
    """시간대 분류"""
    Morning = "Morning"      # 5-12시
    Afternoon = "Afternoon"  # 12-17시
    Evening = "Evening"      # 17-21시
    Night = "Night"          # 21-5시


class EngagementIdea(BaseModel):
    """참여 기회 아이디어"""
    thought_process: str = Field(description="이 아이디어가 좋은 이유에 대한 추론")
    suggested_action: str = Field(description="수행할 고수준 행동 유형")
    action_content: str = Field(description="행동의 구체적 내용")
    user_facing_summary: str = Field(description="사용자에게 보여줄 요약")


@dataclass
class SystemTrigger:
    """시스템 트리거 정보"""
    trigger_type: StimulusType
    content: str
    source: str
    timestamp: datetime


class EnvironmentManager:
    """하린코어 환경 관리자 - PM의 Shell 시스템을 참고하여 구현"""
    
    def __init__(self, harin_main_loop: EnhancedHarinMainLoop):
        self.harin = harin_main_loop
        
        # 큐 시스템
        self.input_queue = queue.Queue()
        self.output_queues: List[queue.Queue] = []
        self.lock = threading.Lock()
        
        # 시간 추적
        self.last_interaction_time = time.time()
        self.last_engagement_check_time = time.time()
        self.last_needs_check_time = time.time()
        self.current_time_of_day = self._get_time_of_day()
        
        # 상태 관리
        self.is_sleeping = False
        self.sleep_until_time = 0
        self.stop_event = threading.Event()
        
        # 설정
        self.USER_INACTIVITY_TIMEOUT = 60 * 120  # 2시간
        self.ENGAGEMENT_STRATEGIST_INTERVAL = 60 * 30  # 30분
        self.NEEDS_CHECK_INTERVAL = 60 * 5  # 5분
        self.NEEDS_DECAY_FACTOR = 0.01
        self.NEEDS_CRITICAL_THRESHOLD = 0.25
        self.SYSTEM_TICK_INTERVAL = 1  # 1초
        
        # 콜백 시스템
        self.on_stimulus_processed: Optional[Callable[[Stimulus], None]] = None
        self.on_action_executed: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # 워커 스레드 시작
        self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self.worker_thread.start()
    
    def _get_time_of_day(self) -> TimeOfDay:
        """현재 시간대 반환"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return TimeOfDay.Morning
        elif 12 <= hour < 17:
            return TimeOfDay.Afternoon
        elif 17 <= hour < 21:
            return TimeOfDay.Evening
        else:
            return TimeOfDay.Night
    
    def register_client(self, output_queue: queue.Queue):
        """클라이언트 등록"""
        with self.lock:
            print(f"새 클라이언트 등록. 총 클라이언트: {len(self.output_queues) + 1}")
            self.output_queues.append(output_queue)
    
    def unregister_client(self, output_queue: queue.Queue):
        """클라이언트 등록 해제"""
        with self.lock:
            try:
                self.output_queues.remove(output_queue)
                print(f"클라이언트 등록 해제. 총 클라이언트: {len(self.output_queues)}")
            except ValueError:
                pass
    
    def _broadcast(self, message: dict):
        """모든 등록된 클라이언트에게 메시지 브로드캐스트"""
        with self.lock:
            for q in list(self.output_queues):
                try:
                    q.put(message)
                except Exception as e:
                    print(f"큐 브로드캐스트 오류, 제거 중. 오류: {e}")
                    self.unregister_client(q)
    
    def process_user_input(self, user_input: str):
        """사용자 입력 처리"""
        self.input_queue.put(user_input)
        self.last_interaction_time = time.time()
        self.last_engagement_check_time = time.time()
    
    def _process_stimulus(self, stimulus: Stimulus):
        """자극 처리"""
        print(f"자극 처리 중: {stimulus.stimulus_type} - '{stimulus.content[:80]}...'")
        self._broadcast({"type": "Status", "content": "생각 중..."})
        
        # 하린 메인 루프로 자극 전달
        result = self.harin.run_session(stimulus.content, {
            "stimulus_type": stimulus.stimulus_type.value,
            "stimulus_source": stimulus.source,
            "is_system_trigger": True
        })
        
        # 결과 브로드캐스트
        self._broadcast({
            "type": "Reply", 
            "content": result["response"],
            "session_id": result["session_id"]
        })
        
        # 콜백 호출
        if self.on_stimulus_processed:
            self.on_stimulus_processed(stimulus)
    
    def _apply_needs_decay(self):
        """욕구 상태 감쇠 적용"""
        print("수동 욕구 감쇠 적용 중.")
        
        # 현재 욕구 상태 가져오기
        current_needs = self.harin._get_current_needs_state()
        
        # 각 욕구에 감쇠 적용
        for field_name, _ in current_needs.model_fields.items():
            current_value = getattr(current_needs, field_name)
            # 지수 감쇠
            new_value = current_value * (1.0 - self.NEEDS_DECAY_FACTOR)
            setattr(current_needs, field_name, max(0.0, new_value))
        
        # 업데이트된 상태 설정
        self.harin._current_needs_state = current_needs
    
    def _check_critical_needs(self) -> bool:
        """중요 욕구 체크"""
        current_needs = self.harin._get_current_needs_state()
        
        critically_low_needs = []
        for field_name, field in current_needs.__class__.model_fields.items():
            value = getattr(current_needs, field_name)
            if value < self.NEEDS_CRITICAL_THRESHOLD:
                critically_low_needs.append((field.description, field_name, value))
        
        if critically_low_needs:
            # 가장 중요한 욕구 (가장 낮은 값) 찾기
            critically_low_needs.sort(key=lambda x: x[2])
            description, name, value = critically_low_needs[0]
            
            print(f"중요 욕구 감지: '{name}'이 {value:.2f}입니다. 자극 생성 중.")
            
            content = f"내부 모니터링에 따르면 '{name}' 욕구가 매우 낮습니다 ({value:.2f}). 이를 해결하고 싶은 강한 욕구를 느낍니다."
            stim = Stimulus(
                source="System_Needs_Monitor", 
                content=content, 
                stimulus_type=StimulusType.LowNeedTrigger
            )
            
            self._process_stimulus(stim)
            return True
        
        return False
    
    def _check_system_triggers(self):
        """시스템 트리거 체크"""
        now = time.time()
        stimulus_generated = False
        
        # 수면 중이면 깨어날 시간인지만 체크
        if self.is_sleeping:
            if now > self.sleep_until_time:
                print("수면 시간 완료. 깨어남 자극 생성.")
                self.is_sleeping = False
                stim = Stimulus(
                    source="System", 
                    content="수면 사이클이 완료되었습니다.", 
                    stimulus_type=StimulusType.WakeUp
                )
                self._process_stimulus(stim)
            return  # 수면 중에는 다른 트리거 처리 안함
        
        # 최우선: 욕구 체크
        if now - self.last_needs_check_time > self.NEEDS_CHECK_INTERVAL:
            self._apply_needs_decay()
            stimulus_generated = self._check_critical_needs()
            self.last_needs_check_time = now
        
        if stimulus_generated:
            return  # 중요 욕구가 있으면 다른 트리거 실행 안함
        
        # 1. 사용자 비활성 체크
        if now - self.last_interaction_time > self.USER_INACTIVITY_TIMEOUT:
            print("사용자 비활성 타임아웃 도달. 자극 생성.")
            content = f"사용자가 {self.USER_INACTIVITY_TIMEOUT / 60:.0f}분 이상 비활성 상태입니다."
            stim = Stimulus(
                source="System", 
                content=content, 
                stimulus_type=StimulusType.UserInactivity
            )
            self._process_stimulus(stim)
            # 중요: 타이머를 현재 시간으로 리셋하여 매 틱마다 실행되지 않도록 함
            self.last_interaction_time = now
        
        # 2. 시간대 변화 체크
        new_time_of_day = self._get_time_of_day()
        if new_time_of_day != self.current_time_of_day:
            print(f"시간대가 {new_time_of_day}로 변경됨. 자극 생성.")
            content = f"시간이 {new_time_of_day}로 전환되었습니다."
            stim = Stimulus(
                source="System", 
                content=content, 
                stimulus_type=StimulusType.TimeOfDayChange
            )
            self._process_stimulus(stim)
            self.current_time_of_day = new_time_of_day
        
        # 3. 적극적 참여 전략가 체크
        if now - self.last_engagement_check_time > self.ENGAGEMENT_STRATEGIST_INTERVAL:
            self._run_engagement_strategist()
            self.last_engagement_check_time = now
    
    def _run_engagement_strategist(self):
        """적극적 참여 전략가 실행"""
        print("적극적 참여 전략가 실행 중...")
        try:
            # 간단한 참여 아이디어 생성 (실제로는 더 복잡한 로직 필요)
            engagement_ideas = [
                {
                    "thought_process": "사용자가 오랫동안 비활성 상태이므로 관심을 끌 수 있는 주제로 대화를 시작하겠습니다.",
                    "suggested_action": "InitiateUserConversation",
                    "action_content": "안녕하세요! 오늘 하루는 어떠셨나요?",
                    "user_facing_summary": "사용자와의 연결을 위해 대화를 시작했습니다."
                },
                {
                    "thought_process": "사용자의 관심사에 맞는 유용한 정보를 제공하여 가치를 창출하겠습니다.",
                    "suggested_action": "ToolCall",
                    "action_content": "최신 기술 뉴스를 검색하여 사용자에게 제공",
                    "user_facing_summary": "최신 기술 뉴스를 찾아드렸습니다."
                }
            ]
            
            # 랜덤하게 아이디어 선택
            idea = random.choice(engagement_ideas)
            
            print(f"전략가가 아이디어 생성: {idea['thought_process']}")
            stim = Stimulus(
                source="Engagement_Strategist",
                content=str(idea),
                stimulus_type=StimulusType.EngagementOpportunity
            )
            self._process_stimulus(stim)
            
        except Exception as e:
            print(f"참여 전략가 오류: {e}")
    
    def force_system_trigger(self, trigger_type: Optional[StimulusType] = None):
        """시스템 트리거 강제 실행 (테스트용)"""
        if self.is_sleeping:
            print("테스트: force_system_trigger 호출됨, 하지만 AI가 수면 중. 무시.")
            return
        
        if trigger_type is None:
            # 랜덤 트리거 선택
            possible_triggers = [
                StimulusType.UserInactivity,
                StimulusType.TimeOfDayChange,
                StimulusType.LowNeedTrigger,
                StimulusType.EngagementOpportunity
            ]
            trigger_type = random.choice(possible_triggers)
        
        print(f"테스트: 시스템 트리거 강제 실행: {trigger_type.value}")
        
        if trigger_type == StimulusType.UserInactivity:
            content = "시스템 에이전트: 사용자가 비활성 상태입니다."
            stim = Stimulus(source="System_Test", content=content, stimulus_type=StimulusType.UserInactivity)
            self._process_stimulus(stim)
        
        elif trigger_type == StimulusType.TimeOfDayChange:
            new_time = random.choice(["Morning", "Afternoon", "Evening", "Night"])
            self.current_time_of_day = TimeOfDay(new_time)
            content = f"시스템 에이전트: 시간이 {new_time}로 전환되었습니다."
            stim = Stimulus(source="System_Test", content=content, stimulus_type=StimulusType.TimeOfDayChange)
            self._process_stimulus(stim)
        
        elif trigger_type == StimulusType.LowNeedTrigger:
            # 현실적으로 만들기 위해 욕구를 수동으로 낮춤
            needs = self.harin._get_current_needs_state()
            need_to_make_low = random.choice(list(needs.__class__.model_fields.keys()))
            setattr(needs, need_to_make_low, 0.1)  # 낮은 값으로 강제 설정
            self.harin._current_needs_state = needs
            print(f"테스트: 욕구 '{need_to_make_low}'을 0.1로 수동 설정.")
            self._check_critical_needs()
        
        elif trigger_type == StimulusType.EngagementOpportunity:
            self._run_engagement_strategist()
    
    def _run_worker(self):
        """워커 스레드 메인 루프"""
        while not self.stop_event.is_set():
            try:
                # 1. 외부(사용자) 입력 먼저 체크
                try:
                    user_input = self.input_queue.get(block=False)
                    stimulus = Stimulus(
                        source="User", 
                        content=user_input, 
                        stimulus_type=StimulusType.UserMessage
                    )
                    
                    self.last_interaction_time = time.time()
                    self.last_engagement_check_time = time.time()
                    
                    self._process_stimulus(stimulus)
                except queue.Empty:
                    # 2. 사용자 입력이 없으면 내부 시스템 트리거 체크
                    self._check_system_triggers()
                    
            except Exception as e:
                print(f"환경 관리자 메인 루프 오류: {e}")
            
            time.sleep(self.SYSTEM_TICK_INTERVAL)
        
        print("환경 관리자 메인 루프가 중지되었습니다.")
    
    def stop(self):
        """환경 관리자 중지"""
        print("종료 신호 수신. 환경 관리자 중지 중.")
        self.stop_event.set()
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "is_sleeping": self.is_sleeping,
            "current_time_of_day": self.current_time_of_day.value,
            "last_interaction_time": self.last_interaction_time,
            "last_engagement_check_time": self.last_engagement_check_time,
            "last_needs_check_time": self.last_needs_check_time,
            "active_clients": len(self.output_queues),
            "user_inactivity_timeout": self.USER_INACTIVITY_TIMEOUT,
            "engagement_strategist_interval": self.ENGAGEMENT_STRATEGIST_INTERVAL,
            "needs_check_interval": self.NEEDS_CHECK_INTERVAL
        }


# 싱글톤 인스턴스 관리
_instance = None
_lock = threading.Lock()

def get_environment_manager(harin_main_loop: EnhancedHarinMainLoop) -> EnvironmentManager:
    """환경 관리자 싱글톤 인스턴스 반환"""
    global _instance
    with _lock:
        if _instance is None:
            print("--- 싱글톤 환경 관리자 인스턴스 생성 ---")
            _instance = EnvironmentManager(harin_main_loop)
            print("--- 싱글톤 환경 관리자 인스턴스 생성 및 스레드 시작 완료 ---")
    return _instance
