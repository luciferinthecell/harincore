"""
하린코어 자극 처리 시스템
PM 시스템의 자극 처리 기능을 참고하여 하린코어에 맞게 구현한 고급 자극 처리 시스템
"""

import time
import threading
import queue
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import StrEnum
from pydantic import BaseModel, Field

from memory.models import Stimulus, StimulusType, StimulusTriage, NeedsAxesModel, EmotionalAxesModel
from core.enhanced_main_loop import EnhancedHarinMainLoop
from core.stimulus_classifier import StimulusClassifier, StimulusAnalysis, ProcessingMode


class ProcessingStatus(StrEnum):
    """처리 상태"""
    Pending = "Pending"
    Processing = "Processing"
    Completed = "Completed"
    Failed = "Failed"
    Timeout = "Timeout"
    Cancelled = "Cancelled"


@dataclass
class ProcessingResult:
    """처리 결과"""
    stimulus: Stimulus
    analysis: StimulusAnalysis
    status: ProcessingStatus
    start_time: datetime
    end_time: Optional[datetime]
    processing_time: Optional[float]
    response: Optional[str]
    error_message: Optional[str]
    session_id: Optional[str]


class StimulusProcessor:
    """하린코어 자극 처리기 - PM 시스템의 자극 처리 기능을 참고하여 구현"""
    
    def __init__(self, harin_main_loop: EnhancedHarinMainLoop):
        self.harin = harin_main_loop
        self.classifier = StimulusClassifier(harin_main_loop)
        
        # 처리 큐들
        self.immediate_queue = queue.Queue()
        self.queued_queue = queue.Queue()
        self.background_queue = queue.Queue()
        self.deferred_queue = queue.Queue()
        
        # 처리 결과 추적
        self.processing_results: List[ProcessingResult] = []
        self.active_processes: Dict[str, ProcessingResult] = {}
        
        # 스레드 관리
        self.worker_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        
        # 설정
        self.max_immediate_workers = 2
        self.max_queued_workers = 4
        self.max_background_workers = 2
        self.max_deferred_workers = 1
        
        # 콜백 시스템
        self.on_stimulus_processed: Optional[Callable[[ProcessingResult], None]] = None
        self.on_processing_error: Optional[Callable[[ProcessingResult], None]] = None
        
        # 워커 스레드 시작
        self._start_worker_threads()
    
    def _start_worker_threads(self):
        """워커 스레드 시작"""
        # 즉시 처리 워커
        for i in range(self.max_immediate_workers):
            thread = threading.Thread(
                target=self._immediate_worker,
                name=f"ImmediateWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        # 큐 처리 워커
        for i in range(self.max_queued_workers):
            thread = threading.Thread(
                target=self._queued_worker,
                name=f"QueuedWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        # 백그라운드 워커
        for i in range(self.max_background_workers):
            thread = threading.Thread(
                target=self._background_worker,
                name=f"BackgroundWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        # 지연 처리 워커
        for i in range(self.max_deferred_workers):
            thread = threading.Thread(
                target=self._deferred_worker,
                name=f"DeferredWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        print(f"자극 처리기 워커 스레드 시작 완료: {len(self.worker_threads)}개")
    
    def process_stimulus(self, stimulus: Stimulus) -> str:
        """자극 처리 시작"""
        print(f"자극 처리 시작: {stimulus.stimulus_type.value}")
        
        # 자극 분류
        analysis = self.classifier.classify_stimulus(stimulus)
        
        # 처리 결과 생성
        result = ProcessingResult(
            stimulus=stimulus,
            analysis=analysis,
            status=ProcessingStatus.Pending,
            start_time=datetime.now(),
            end_time=None,
            processing_time=None,
            response=None,
            error_message=None,
            session_id=None
        )
        
        # 처리 모드에 따른 큐 배치
        if analysis.processing_mode == ProcessingMode.Immediate:
            self.immediate_queue.put(result)
            print(f"즉시 처리 큐에 추가: {stimulus.stimulus_type.value}")
        
        elif analysis.processing_mode == ProcessingMode.Queued:
            self.queued_queue.put(result)
            print(f"큐 처리 큐에 추가: {stimulus.stimulus_type.value}")
        
        elif analysis.processing_mode == ProcessingMode.Background:
            self.background_queue.put(result)
            print(f"백그라운드 처리 큐에 추가: {stimulus.stimulus_type.value}")
        
        elif analysis.processing_mode == ProcessingMode.Deferred:
            self.deferred_queue.put(result)
            print(f"지연 처리 큐에 추가: {stimulus.stimulus_type.value}")
        
        elif analysis.processing_mode == ProcessingMode.Ignored:
            result.status = ProcessingStatus.Cancelled
            result.end_time = datetime.now()
            result.processing_time = 0.0
            print(f"자극 무시: {stimulus.stimulus_type.value}")
        
        # 결과 저장
        self.processing_results.append(result)
        
        return f"stimulus_{len(self.processing_results)}"
    
    def _immediate_worker(self):
        """즉시 처리 워커"""
        while not self.stop_event.is_set():
            try:
                result = self.immediate_queue.get(timeout=1.0)
                self._process_stimulus_result(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"즉시 처리 워커 오류: {e}")
    
    def _queued_worker(self):
        """큐 처리 워커"""
        while not self.stop_event.is_set():
            try:
                result = self.queued_queue.get(timeout=1.0)
                self._process_stimulus_result(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"큐 처리 워커 오류: {e}")
    
    def _background_worker(self):
        """백그라운드 처리 워커"""
        while not self.stop_event.is_set():
            try:
                result = self.background_queue.get(timeout=5.0)
                self._process_stimulus_result(result, background=True)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"백그라운드 처리 워커 오류: {e}")
    
    def _deferred_worker(self):
        """지연 처리 워커"""
        while not self.stop_event.is_set():
            try:
                result = self.deferred_queue.get(timeout=10.0)
                # 지연 처리의 경우 추가 지연
                time.sleep(5.0)
                self._process_stimulus_result(result, background=True)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"지연 처리 워커 오류: {e}")
    
    def _process_stimulus_result(self, result: ProcessingResult, background: bool = False):
        """자극 처리 실행"""
        stimulus = result.stimulus
        analysis = result.analysis
        
        print(f"자극 처리 실행: {stimulus.stimulus_type.value} ({analysis.processing_mode.value})")
        
        # 처리 시작
        result.status = ProcessingStatus.Processing
        process_id = f"process_{len(self.active_processes)}"
        self.active_processes[process_id] = result
        
        try:
            # 타임아웃 설정
            timeout = analysis.processing_timeout
            
            # 하린 메인 루프로 자극 전달
            if background:
                # 백그라운드 처리는 비동기로 실행
                response = self._process_background_stimulus(stimulus, analysis)
            else:
                # 일반 처리는 동기로 실행
                response = self._process_normal_stimulus(stimulus, analysis, timeout)
            
            # 처리 완료
            result.status = ProcessingStatus.Completed
            result.response = response.get("response", "")
            result.session_id = response.get("session_id", "")
            
            print(f"자극 처리 완료: {stimulus.stimulus_type.value}")
            
        except Exception as e:
            # 처리 실패
            result.status = ProcessingStatus.Failed
            result.error_message = str(e)
            print(f"자극 처리 실패: {stimulus.stimulus_type.value} - {e}")
            
            # 오류 콜백 호출
            if self.on_processing_error:
                self.on_processing_error(result)
        
        finally:
            # 처리 시간 계산
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - result.start_time).total_seconds()
            
            # 활성 프로세스에서 제거
            if process_id in self.active_processes:
                del self.active_processes[process_id]
            
            # 완료 콜백 호출
            if self.on_stimulus_processed:
                self.on_stimulus_processed(result)
    
    def _process_normal_stimulus(self, stimulus: Stimulus, analysis: StimulusAnalysis, 
                                timeout: float) -> Dict[str, Any]:
        """일반 자극 처리"""
        # 하린 메인 루프로 자극 전달
        result = self.harin.run_session(stimulus.content, {
            "stimulus_type": stimulus.stimulus_type.value,
            "stimulus_source": stimulus.source,
            "is_system_trigger": stimulus.stimulus_type != StimulusType.UserMessage,
            "priority": analysis.priority.value,
            "triage": analysis.triage.value,
            "urgency_score": analysis.urgency_score,
            "complexity_score": analysis.complexity_score,
            "emotional_impact": analysis.emotional_impact,
            "needs_impact": analysis.needs_impact
        })
        
        return result
    
    def _process_background_stimulus(self, stimulus: Stimulus, analysis: StimulusAnalysis) -> Dict[str, Any]:
        """백그라운드 자극 처리"""
        # 백그라운드 처리는 간소화된 처리
        if stimulus.stimulus_type == StimulusType.TimeOfDayChange:
            # 시간대 변화는 간단한 로그만
            return {
                "response": f"시간대가 변경되었습니다: {stimulus.content}",
                "session_id": f"bg_{int(time.time())}"
            }
        
        elif stimulus.stimulus_type == StimulusType.UserInactivity:
            # 사용자 비활성은 간단한 체크만
            return {
                "response": f"사용자 비활성 감지: {stimulus.content}",
                "session_id": f"bg_{int(time.time())}"
            }
        
        else:
            # 기타 백그라운드 처리는 일반 처리와 동일
            return self._process_normal_stimulus(stimulus, analysis, 30.0)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        total = len(self.processing_results)
        completed = len([r for r in self.processing_results if r.status == ProcessingStatus.Completed])
        failed = len([r for r in self.processing_results if r.status == ProcessingStatus.Failed])
        processing = len([r for r in self.processing_results if r.status == ProcessingStatus.Processing])
        pending = len([r for r in self.processing_results if r.status == ProcessingStatus.Pending])
        
        avg_processing_time = 0.0
        if completed > 0:
            completed_results = [r for r in self.processing_results if r.processing_time is not None]
            if completed_results:
                avg_processing_time = sum(r.processing_time for r in completed_results) / len(completed_results)
        
        return {
            "total_processed": total,
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "pending": pending,
            "avg_processing_time": avg_processing_time,
            "active_processes": len(self.active_processes),
            "queue_sizes": {
                "immediate": self.immediate_queue.qsize(),
                "queued": self.queued_queue.qsize(),
                "background": self.background_queue.qsize(),
                "deferred": self.deferred_queue.qsize()
            }
        }
    
    def get_recent_results(self, limit: int = 10) -> List[ProcessingResult]:
        """최근 처리 결과 반환"""
        return self.processing_results[-limit:] if self.processing_results else []
    
    def cancel_pending_stimulus(self, stimulus_id: str) -> bool:
        """대기 중인 자극 취소"""
        # 구현: 큐에서 특정 자극을 찾아 제거
        # 현재는 간단한 구현으로 대체
        return False
    
    def stop(self):
        """자극 처리기 중지"""
        print("자극 처리기 중지 신호 수신")
        self.stop_event.set()
        
        # 모든 워커 스레드 종료 대기
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        print("자극 처리기 중지 완료")


# 싱글톤 인스턴스 관리
_instance = None

def get_stimulus_processor(harin_main_loop: EnhancedHarinMainLoop) -> StimulusProcessor:
    """자극 처리기 싱글톤 인스턴스 반환"""
    global _instance
    if _instance is None:
        print("--- 싱글톤 자극 처리기 인스턴스 생성 ---")
        _instance = StimulusProcessor(harin_main_loop)
    return _instance 