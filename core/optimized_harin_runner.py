"""
harin.core.optimized_harin_runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nano-vllm 최적화가 적용된 Harin 러너
- Prefix Caching으로 반복 프롬프트 최적화
- Tensor Parallelism으로 대형 모델 처리
- 병렬 추론과 메모리 시스템 통합
- 성능 모니터링 및 벤치마킹
"""

from __future__ import annotations

import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from core.context import UserContext
from enhanced_main_loop import EnhancedHarinMainLoop
from parallel_reasoning_unit import ParallelReasoningUnit
from memory.palantirgraph import PalantirGraph
from memory.memory_retriever import MemoryRetriever
from tools.nano_vllm_client import OptimizedLLMClient, create_nano_vllm_client
from prompt.prompt_architect import PromptArchitect
from tools.llm_client import LLMClient
from core.multi_intent_parser_fixed import MultiIntentParser


class OptimizedHarinRunner:
    """nano-vllm 최적화가 적용된 Harin 러너"""
    
    def __init__(self, 
                 model_path: str = None,
                 memory_path: str = "palantir_graph.json",
                 tensor_parallel_size: int = 1,
                 enable_prefix_caching: bool = True,
                 enable_cuda_graph: bool = True):
        
        # 메모리 시스템 초기화
        self.memory = PalantirGraph(persist_path=memory_path)
        self.memory_retriever = MemoryRetriever(self.memory)
        
        # nano-vllm 최적화 클라이언트
        if model_path:
            self.llm_client = create_nano_vllm_client(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                enable_prefix_caching=enable_prefix_caching,
                enable_cuda_graph=enable_cuda_graph
            )
        else:
            # fallback to standard client
            self.llm_client = LLMClient.from_env()
        
        # 핵심 컴포넌트들
        self.prompt_architect = PromptArchitect()
        self.parallel_reasoner = ParallelReasoningUnit(max_workers=4)
        self.context = UserContext()
        
        # 성능 추적
        self.performance_metrics = {
            'total_sessions': 0,
            'total_inference_time': 0.0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'parallel_efficiency': 0.0
        }
        
        # 설정
        self.enable_optimization = True
        self.enable_performance_tracking = True
        
    def run_session(self, user_input: str, session_meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """최적화된 세션 실행"""
        
        if session_meta is None:
            session_meta = {}
        
        start_time = time.time()
        session_id = f"opt_session_{int(time.time())}"
        
        try:
            # 1. 관련 기억 검색 (Prefix 캐싱 활용)
            relevant_memories = self._retrieve_optimized_memories(user_input)
            
            # 2. 병렬 추론 실행
            reasoning_result = self._execute_parallel_reasoning(user_input, relevant_memories)
            
            # 3. 최적화된 프롬프트 생성
            prompt = self._create_optimized_prompt(user_input, reasoning_result, relevant_memories)
            
            # 4. LLM 호출 (Prefix 캐싱 적용)
            final_response = self.llm_client.complete(prompt, temperature=0.7)
            
            # 5. 성능 통계 업데이트
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, reasoning_result)
            
            # 6. 결과 저장
            self._store_session_result(session_id, user_input, final_response, reasoning_result, relevant_memories)
            
            return {
                "session_id": session_id,
                "input": user_input,
                "output": final_response,
                "execution_time": execution_time,
                "performance_stats": self._get_session_performance_stats(),
                "memory_usage": len(relevant_memories),
                "optimization_enabled": self.enable_optimization,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "session_id": session_id,
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _retrieve_optimized_memories(self, user_input: str) -> List[Any]:
        """최적화된 기억 검색 (Prefix 캐싱 활용)"""
        # 자주 사용되는 쿼리 패턴에 대한 캐싱
        query_pattern = self._extract_query_pattern(user_input)
        
        # 메모리 검색
        memories = self.memory_retriever.retrieve_relevant_memories(
            query=user_input,
            top_k=5,
            min_similarity=0.3
        )
        
        return memories
    
    def _extract_query_pattern(self, user_input: str) -> str:
        """쿼리 패턴 추출 (캐싱 키 생성용)"""
        # 간단한 패턴 추출 (첫 50자 + 키워드)
        words = user_input.lower().split()[:5]
        pattern = " ".join(words)
        return pattern[:50]
    
    def _execute_parallel_reasoning(self, user_input: str, memories: List[Any]) -> Dict[str, Any]:
        """?? ?? ??"""
        
        # 의도 파싱
        intent_parser = MultiIntentParser()
        intents = intent_parser.parse_intents(user_input)
        
        # 병렬 추론 실행
        reasoning_result = self.parallel_reasoner.process_parallel_reasoning(
            intents=intents,
            context={"memories": memories, "user_input": user_input}
        )
        
        return {
            "intents": intents,
            "parallel_result": reasoning_result,
            "execution_summary": reasoning_result.execution_summary
        }
    
    def _create_optimized_prompt(self, user_input: str, reasoning_result: Dict[str, Any], 
                               memories: List[Any]) -> str:
        """최적화된 프롬프트 생성"""
        
        # 기억 요약
        memory_summary = self._summarize_memories(memories)
        
        # 추론 결과 요약
        reasoning_summary = self._summarize_reasoning(reasoning_result)
        
        # 최적화된 프롬프트 구성
        prompt = f"""
[시스템] 당신은 Harin AI입니다. 최적화된 추론과 기억을 활용하여 정확하고 유용한 응답을 제공하세요.

[사용자 입력]
{user_input}

[관련 기억]
{memory_summary}

[추론 과정]
{reasoning_summary}

[응답 지침]
- 기억과 추론 결과를 종합하여 응답하세요
- 명확하고 구조화된 답변을 제공하세요
- 필요시 추가 정보를 요청하세요

응답:
"""
        
        return prompt
    
    def _summarize_memories(self, memories: List[Any]) -> str:
        """기억 요약"""
        if not memories:
            return "관련 기억이 없습니다."
        
        summaries = []
        for i, memory in enumerate(memories[:3]):  # 상위 3개만
            summaries.append(f"{i+1}. {memory.content[:100]}...")
        
        return "\n".join(summaries)
    
    def _summarize_reasoning(self, reasoning_result: Dict[str, Any]) -> str:
        """추론 결과 요약"""
        parallel_result = reasoning_result.get("parallel_result")
        if not parallel_result:
            return "추론 과정이 없습니다."
        
        summary = []
        for path in parallel_result.paths[:3]:  # 상위 3개 경로만
            if path.status.value == "completed":
                summary.append(f"- {path.intent.intent_type}: {path.intent.content[:50]}...")
        
        return "\n".join(summary) if summary else "추론 과정이 없습니다."
    
    def _update_performance_metrics(self, execution_time: float, reasoning_result: Dict[str, Any]):
        """성능 지표 업데이트"""
        self.performance_metrics['total_sessions'] += 1
        self.performance_metrics['total_inference_time'] += execution_time
        self.performance_metrics['average_response_time'] = (
            self.performance_metrics['total_inference_time'] / 
            self.performance_metrics['total_sessions']
        )
        
        # 캐시 히트율 업데이트
        if hasattr(self.llm_client, 'get_performance_stats'):
            stats = self.llm_client.get_performance_stats()
            if 'cache_stats' in stats:
                self.performance_metrics['cache_hit_rate'] = stats['cache_stats'].get('hit_rate', 0.0)
        
        # 병렬 효율성 업데이트
        parallel_result = reasoning_result.get("parallel_result")
        if parallel_result:
            self.performance_metrics['parallel_efficiency'] = (
                parallel_result.execution_summary.get('parallelization_efficiency', 0.0)
            )
    
    def _get_session_performance_stats(self) -> Dict[str, Any]:
        """세션 성능 통계 반환"""
        stats = self.performance_metrics.copy()
        
        # LLM 클라이언트 통계 추가
        if hasattr(self.llm_client, 'get_performance_stats'):
            llm_stats = self.llm_client.get_performance_stats()
            stats['llm_stats'] = llm_stats
        
        return stats
    
    def _store_session_result(self, session_id: str, user_input: str, response: str,
                            reasoning_result: Dict[str, Any], memories: List[Any]):
        """세션 결과 저장"""
        # 메모리에 세션 저장
        session_data = {
            "session_id": session_id,
            "user_input": user_input,
            "response": response,
            "reasoning_result": reasoning_result,
            "memory_count": len(memories),
            "timestamp": datetime.now().isoformat()
        }
        
        self.memory.add_node(
            content=f"Session {session_id}: {user_input[:50]}...",
            node_type="session",
            meta=session_data
        )
    
    def benchmark_performance(self, test_inputs: List[str]) -> Dict[str, Any]:
        """성능 벤치마크 실행"""
        results = []
        
        for i, test_input in enumerate(test_inputs):
            print(f"벤치마크 진행 중: {i+1}/{len(test_inputs)}")
            
            result = self.run_session(test_input)
            results.append(result)
        
        # 통계 계산
        total_time = sum(r.get('execution_time', 0) for r in results)
        avg_time = total_time / len(results) if results else 0
        success_rate = len([r for r in results if 'error' not in r]) / len(results) if results else 0
        
        return {
            'total_tests': len(test_inputs),
            'total_time': total_time,
            'average_time': avg_time,
            'success_rate': success_rate,
            'performance_metrics': self.performance_metrics,
            'detailed_results': results
        }
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """최적화 상태 반환"""
        return {
            'optimization_enabled': self.enable_optimization,
            'performance_tracking': self.enable_performance_tracking,
            'performance_metrics': self.performance_metrics,
            'llm_client_type': type(self.llm_client).__name__,
            'memory_size': len(self.memory.nodes),
            'cache_stats': self.llm_client.get_performance_stats() if hasattr(self.llm_client, 'get_performance_stats') else None
        }
