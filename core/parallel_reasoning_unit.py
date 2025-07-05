"""
병렬 추론 단위 (Parallel Reasoning Unit)
다중 의도를 동시에 처리하고 각각의 추론 경로를 독립적으로 실행
"""

import asyncio
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from core.multi_intent_parser_fixed import ParsedIntent, IntentType


class ReasoningPathStatus(Enum):
    """추론 경로 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ReasoningPath:
    """추론 경로"""
    id: str
    intent: ParsedIntent
    status: ReasoningPathStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ParallelReasoningResult:
    """병렬 추론 결과"""
    paths: List[ReasoningPath]
    integrated_result: Dict[str, Any]
    execution_summary: Dict[str, Any]
    missed_intents: List[ParsedIntent]


class ParallelReasoningUnit:
    """병렬 추론 단위"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.reasoning_handlers = {
            IntentType.COMMAND: self._handle_command,
            IntentType.QUESTION: self._handle_question,
            IntentType.STATEMENT: self._handle_statement,
            IntentType.EMOTION: self._handle_emotion,
            IntentType.JUDGMENT: self._handle_judgment,
            IntentType.REQUEST: self._handle_request,
            IntentType.FEEDBACK: self._handle_feedback
        }
    
    def process_parallel_reasoning(self, intents: List[ParsedIntent], 
                                 context: Dict[str, Any] = None) -> ParallelReasoningResult:
        """병렬 추론 처리"""
        if context is None:
            context = {}
        
        # 1. 추론 경로 생성
        reasoning_paths = self._create_reasoning_paths(intents)
        
        # 2. 의존성 순서 정렬
        ordered_paths = self._order_by_dependencies(reasoning_paths)
        
        # 3. 병렬 실행
        completed_paths = self._execute_parallel_paths(ordered_paths, context)
        
        # 4. 결과 통합
        integrated_result = self._integrate_results(completed_paths)
        
        # 5. 누락된 의도 확인
        missed_intents = self._identify_missed_intents(intents, completed_paths)
        
        # 6. 실행 요약 생성
        execution_summary = self._create_execution_summary(completed_paths, missed_intents)
        
        return ParallelReasoningResult(
            paths=completed_paths,
            integrated_result=integrated_result,
            execution_summary=execution_summary,
            missed_intents=missed_intents
        )
    
    def _create_reasoning_paths(self, intents: List[ParsedIntent]) -> List[ReasoningPath]:
        """추론 경로 생성"""
        paths = []
        
        for intent in intents:
            path = ReasoningPath(
                id=f"path_{intent.id}",
                intent=intent,
                status=ReasoningPathStatus.PENDING,
                dependencies=intent.dependencies.copy()
            )
            paths.append(path)
        
        return paths
    
    def _order_by_dependencies(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """의존성 순서로 정렬"""
        # 위상 정렬 알고리즘
        in_degree = {path.id: 0 for path in paths}
        graph = {path.id: [] for path in paths}
        
        # 의존성 그래프 구성
        for path in paths:
            for dep_id in path.dependencies:
                if dep_id in graph:
                    graph[dep_id].append(path.id)
                    in_degree[path.id] += 1
        
        # 위상 정렬
        ordered = []
        queue = [path_id for path_id, degree in in_degree.items() if degree == 0]
        
        while queue:
            current_id = queue.pop(0)
            ordered.append(current_id)
            
            for neighbor_id in graph[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)
        
        # 순서대로 경로 재배열
        path_dict = {path.id: path for path in paths}
        return [path_dict[path_id] for path_id in ordered if path_id in path_dict]
    
    def _execute_parallel_paths(self, paths: List[ReasoningPath], 
                              context: Dict[str, Any]) -> List[ReasoningPath]:
        """병렬 경로 실행"""
        import time
        
        # 의존성 그룹별로 실행
        execution_groups = self._group_by_dependencies(paths)
        completed_paths = []
        
        for group in execution_groups:
            # 그룹 내에서 병렬 실행
            futures = []
            path_future_map = {}
            
            for path in group:
                if path.status == ReasoningPathStatus.PENDING:
                    future = self.executor.submit(
                        self._execute_single_path, path, context, completed_paths
                    )
                    futures.append(future)
                    path_future_map[future] = path
            
            # 결과 수집
            for future in as_completed(futures):
                path = path_future_map[future]
                try:
                    result = future.result()
                    path.result = result
                    path.status = ReasoningPathStatus.COMPLETED
                except Exception as e:
                    path.error = str(e)
                    path.status = ReasoningPathStatus.FAILED
                
                completed_paths.append(path)
        
        return completed_paths
    
    def _group_by_dependencies(self, paths: List[ReasoningPath]) -> List[List[ReasoningPath]]:
        """의존성별 그룹화"""
        groups = []
        current_group = []
        
        for path in paths:
            # 의존성이 모두 완료되었는지 확인
            dependencies_met = all(
                any(p.id == dep_id and p.status == ReasoningPathStatus.COMPLETED 
                    for p in current_group)
                for dep_id in path.dependencies
            )
            
            if dependencies_met or not path.dependencies:
                current_group.append(path)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [path]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _execute_single_path(self, path: ReasoningPath, context: Dict[str, Any],
                           completed_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """단일 경로 실행"""
        import time
        
        start_time = time.time()
        path.status = ReasoningPathStatus.RUNNING
        
        try:
            # 의도 타입에 따른 핸들러 호출
            handler = self.reasoning_handlers.get(path.intent.intent_type)
            if handler:
                result = handler(path.intent, context, completed_paths)
            else:
                result = {"error": f"Unknown intent type: {path.intent.intent_type}"}
            
            path.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            path.execution_time = time.time() - start_time
            raise e
    
    def _handle_command(self, intent: ParsedIntent, context: Dict[str, Any],
                       completed_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """명령 처리"""
        return {
            "type": "command_execution",
            "command": intent.content,
            "priority": intent.priority,
            "execution_plan": self._create_execution_plan(intent, context),
            "estimated_time": "variable"
        }
    
    def _handle_question(self, intent: ParsedIntent, context: Dict[str, Any],
                        completed_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """질문 처리"""
        return {
            "type": "question_analysis",
            "question": intent.content,
            "priority": intent.priority,
            "answer_approach": self._determine_answer_approach(intent, context),
            "related_context": self._find_related_context(intent, completed_paths)
        }
    
    def _handle_statement(self, intent: ParsedIntent, context: Dict[str, Any],
                         completed_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """진술 처리"""
        return {
            "type": "statement_processing",
            "statement": intent.content,
            "priority": intent.priority,
            "analysis": self._analyze_statement(intent, context),
            "implications": self._extract_implications(intent, context)
        }
    
    def _handle_emotion(self, intent: ParsedIntent, context: Dict[str, Any],
                       completed_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """감정 처리"""
        return {
            "type": "emotion_processing",
            "emotion": intent.emotion or "unknown",
            "content": intent.content,
            "priority": intent.priority,
            "emotional_response": self._generate_emotional_response(intent, context),
            "context_impact": self._assess_emotional_impact(intent, context)
        }
    
    def _handle_judgment(self, intent: ParsedIntent, context: Dict[str, Any],
                        completed_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """판단 처리"""
        return {
            "type": "judgment_processing",
            "judgment": intent.content,
            "priority": intent.priority,
            "criteria": self._extract_judgment_criteria(intent, context),
            "validation": self._validate_judgment(intent, context)
        }
    
    def _handle_request(self, intent: ParsedIntent, context: Dict[str, Any],
                       completed_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """요청 처리"""
        return {
            "type": "request_processing",
            "request": intent.content,
            "priority": intent.priority,
            "feasibility": self._assess_request_feasibility(intent, context),
            "response_plan": self._create_response_plan(intent, context)
        }
    
    def _handle_feedback(self, intent: ParsedIntent, context: Dict[str, Any],
                        completed_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """피드백 처리"""
        return {
            "type": "feedback_processing",
            "feedback": intent.content,
            "priority": intent.priority,
            "feedback_type": self._classify_feedback(intent, context),
            "improvement_actions": self._extract_improvement_actions(intent, context)
        }
    
    def _create_execution_plan(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """실행 계획 생성"""
        return {
            "steps": ["분석", "계획", "실행", "검증"],
            "resources_needed": ["메모리", "추론_엔진"],
            "estimated_duration": "variable"
        }
    
    def _determine_answer_approach(self, intent: ParsedIntent, context: Dict[str, Any]) -> str:
        """답변 접근 방식 결정"""
        if "어떻게" in intent.content:
            return "방법론적_접근"
        elif "왜" in intent.content:
            return "원인_분석"
        elif "무엇" in intent.content:
            return "정의_및_설명"
        else:
            return "일반적_응답"
    
    def _find_related_context(self, intent: ParsedIntent, 
                            completed_paths: List[ReasoningPath]) -> List[Dict[str, Any]]:
        """관련 맥락 찾기"""
        related = []
        for path in completed_paths:
            if path.status == ReasoningPathStatus.COMPLETED:
                # 간단한 키워드 매칭
                if any(word in path.intent.content for word in intent.content.split()):
                    related.append({
                        "path_id": path.id,
                        "content": path.intent.content,
                        "result": path.result
                    })
        return related
    
    def _analyze_statement(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """진술 분석"""
        return {
            "factual_content": intent.content,
            "assumptions": [],
            "implications": []
        }
    
    def _extract_implications(self, intent: ParsedIntent, context: Dict[str, Any]) -> List[str]:
        """함의 추출"""
        return ["일반적_함의"]
    
    def _generate_emotional_response(self, intent: ParsedIntent, context: Dict[str, Any]) -> str:
        """감정적 응답 생성"""
        return "감정_인식_및_응답"
    
    def _assess_emotional_impact(self, intent: ParsedIntent, context: Dict[str, Any]) -> str:
        """감정적 영향 평가"""
        return "중간_영향"
    
    def _extract_judgment_criteria(self, intent: ParsedIntent, context: Dict[str, Any]) -> List[str]:
        """판단 기준 추출"""
        return ["일반적_기준"]
    
    def _validate_judgment(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """판단 검증"""
        return {
            "validity": "검증_필요",
            "confidence": intent.confidence
        }
    
    def _assess_request_feasibility(self, intent: ParsedIntent, context: Dict[str, Any]) -> str:
        """요청 실현 가능성 평가"""
        return "평가_중"
    
    def _create_response_plan(self, intent: ParsedIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """응답 계획 생성"""
        return {
            "approach": "단계적_응답",
            "timeline": "즉시_처리"
        }
    
    def _classify_feedback(self, intent: ParsedIntent, context: Dict[str, Any]) -> str:
        """피드백 분류"""
        if "개선" in intent.content or "수정" in intent.content:
            return "개선_요청"
        elif "좋다" in intent.content or "나쁘다" in intent.content:
            return "평가_피드백"
        else:
            return "일반_피드백"
    
    def _extract_improvement_actions(self, intent: ParsedIntent, context: Dict[str, Any]) -> List[str]:
        """개선 행동 추출"""
        return ["피드백_반영_계획"]
    
    def _integrate_results(self, completed_paths: List[ReasoningPath]) -> Dict[str, Any]:
        """결과 통합"""
        integrated = {
            "primary_response": "",
            "secondary_responses": [],
            "emotional_context": "",
            "action_items": [],
            "follow_up_questions": []
        }
        
        # 우선순위별로 결과 정리
        high_priority = [p for p in completed_paths if p.priority >= 4]
        medium_priority = [p for p in completed_paths if 2 <= p.priority < 4]
        low_priority = [p for p in completed_paths if p.priority < 2]
        
        # 주요 응답 (최고 우선순위)
        if high_priority:
            primary_path = max(high_priority, key=lambda p: p.priority)
            integrated["primary_response"] = self._format_primary_response(primary_path)
        
        # 보조 응답들
        for path in medium_priority:
            integrated["secondary_responses"].append({
                "type": path.intent.intent_type.value,
                "content": path.result,
                "priority": path.priority
            })
        
        # 감정적 맥락
        emotion_paths = [p for p in completed_paths if p.intent.intent_type == IntentType.EMOTION]
        if emotion_paths:
            integrated["emotional_context"] = self._combine_emotional_context(emotion_paths)
        
        # 행동 항목
        command_paths = [p for p in completed_paths if p.intent.intent_type == IntentType.COMMAND]
        integrated["action_items"] = [p.result for p in command_paths]
        
        # 후속 질문
        question_paths = [p for p in completed_paths if p.intent.intent_type == IntentType.QUESTION]
        integrated["follow_up_questions"] = [p.result for p in question_paths]
        
        return integrated
    
    def _format_primary_response(self, path: ReasoningPath) -> str:
        """주요 응답 포맷팅"""
        if path.result:
            return f"{path.intent.intent_type.value}: {path.result}"
        return f"{path.intent.intent_type.value}: 처리됨"
    
    def _combine_emotional_context(self, emotion_paths: List[ReasoningPath]) -> str:
        """감정적 맥락 결합"""
        emotions = []
        for path in emotion_paths:
            if path.result and "emotion" in path.result:
                emotions.append(path.result["emotion"])
        return ", ".join(emotions) if emotions else "감정_인식됨"
    
    def _identify_missed_intents(self, original_intents: List[ParsedIntent],
                               completed_paths: List[ReasoningPath]) -> List[ParsedIntent]:
        """누락된 의도 식별"""
        completed_intent_ids = {path.intent.id for path in completed_paths}
        missed = []
        
        for intent in original_intents:
            if intent.id not in completed_intent_ids:
                missed.append(intent)
        
        return missed
    
    def _create_execution_summary(self, completed_paths: List[ReasoningPath],
                                missed_intents: List[ParsedIntent]) -> Dict[str, Any]:
        """실행 요약 생성"""
        total_time = sum(path.execution_time for path in completed_paths)
        success_count = len([p for p in completed_paths if p.status == ReasoningPathStatus.COMPLETED])
        failed_count = len([p for p in completed_paths if p.status == ReasoningPathStatus.FAILED])
        
        return {
            "total_paths": len(completed_paths),
            "successful_paths": success_count,
            "failed_paths": failed_count,
            "missed_intents": len(missed_intents),
            "total_execution_time": total_time,
            "average_execution_time": total_time / len(completed_paths) if completed_paths else 0,
            "parallelization_efficiency": success_count / len(completed_paths) if completed_paths else 0
        }
    
    def shutdown(self):
        """리소스 정리"""
        self.executor.shutdown(wait=True) 
