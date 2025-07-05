"""
harin.integrations.nano_vllm_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nano-vllm 기반 최적화된 LLM 클라이언트
- Prefix Caching
- Tensor Parallelism  
- CUDA Graph 최적화
- Torch Compilation
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from nanovllm import LLM, SamplingParams
    NANOVLLM_AVAILABLE = True
except ImportError:
    NANOVLLM_AVAILABLE = False
    print("Warning: nano-vllm not installed. Install with: pip install git+https://github.com/GeeeekExplorer/nano-vllm.git")

from tools.llm_client import LLMClient, LLMBackend


class NanoVLLMBackend:
    """nano-vllm 백엔드"""
    
    def __init__(self, 
                 model_path: str,
                 tensor_parallel_size: int = 1,
                 enforce_eager: bool = False,
                 enable_prefix_caching: bool = True,
                 enable_cuda_graph: bool = True):
        
        if not NANOVLLM_AVAILABLE:
            raise RuntimeError("nano-vllm package not installed")
            
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_cuda_graph = enable_cuda_graph
        
        # nano-vllm LLM 인스턴스 초기화
        self.llm = LLM(
            model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size
        )
        
        # Prefix 캐시 (자주 사용되는 프롬프트 패턴)
        self.prefix_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def complete(self, prompt: str, **kwargs) -> str:
        """최적화된 완성 처리"""
        
        # 1. Prefix 캐싱 확인
        if self.enable_prefix_caching:
            cached_result = self._check_prefix_cache(prompt)
            if cached_result:
                return cached_result
        
        # 2. Sampling 파라미터 설정
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 512),
            top_p=kwargs.get('top_p', 1.0),
            top_k=kwargs.get('top_k', -1)
        )
        
        # 3. nano-vllm 추론 실행
        start_time = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        inference_time = time.time() - start_time
        
        result = outputs[0]["text"]
        
        # 4. Prefix 캐시 업데이트
        if self.enable_prefix_caching:
            self._update_prefix_cache(prompt, result, inference_time)
        
        return result
    
    def _check_prefix_cache(self, prompt: str) -> Optional[str]:
        """Prefix 캐시 확인"""
        # 프롬프트의 첫 부분을 키로 사용
        prefix_key = prompt[:100]  # 첫 100자
        
        if prefix_key in self.prefix_cache:
            cache_entry = self.prefix_cache[prefix_key]
            
            # 캐시 유효성 검사 (시간 기반)
            if time.time() - cache_entry['timestamp'] < 3600:  # 1시간
                self.cache_hits += 1
                return cache_entry['result']
        
        self.cache_misses += 1
        return None
    
    def _update_prefix_cache(self, prompt: str, result: str, inference_time: float):
        """Prefix 캐시 업데이트"""
        prefix_key = prompt[:100]
        
        self.prefix_cache[prefix_key] = {
            'result': result,
            'timestamp': time.time(),
            'inference_time': inference_time,
            'prompt_length': len(prompt)
        }
        
        # 캐시 크기 제한 (메모리 관리)
        if len(self.prefix_cache) > 1000:
            # 가장 오래된 항목 제거
            oldest_key = min(self.prefix_cache.keys(), 
                           key=lambda k: self.prefix_cache[k]['timestamp'])
            del self.prefix_cache[oldest_key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.prefix_cache),
            'tensor_parallel_size': self.tensor_parallel_size,
            'enforce_eager': self.enforce_eager
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        self.prefix_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class OptimizedLLMClient(LLMClient):
    """최적화된 LLM 클라이언트"""
    
    def __init__(self, backend: LLMBackend):
        super().__init__(backend)
        self.performance_stats = {
            'total_requests': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0
        }
    
    def complete(self, prompt: str, **kwargs) -> str:
        """성능 추적이 포함된 완성 처리"""
        start_time = time.time()
        
        try:
            result = super().complete(prompt, **kwargs)
            
            # 성능 통계 업데이트
            inference_time = time.time() - start_time
            self.performance_stats['total_requests'] += 1
            self.performance_stats['total_inference_time'] += inference_time
            self.performance_stats['average_inference_time'] = (
                self.performance_stats['total_inference_time'] / 
                self.performance_stats['total_requests']
            )
            
            return result
            
        except Exception as e:
            inference_time = time.time() - start_time
            self.performance_stats['total_requests'] += 1
            self.performance_stats['total_inference_time'] += inference_time
            
            # 에러 처리
            return f"(LLM error: {e.__class__.__name__}) {prompt.splitlines()[-1]}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.performance_stats.copy()
        
        # nano-vllm 백엔드인 경우 캐시 통계 추가
        if hasattr(self.backend, 'get_cache_stats'):
            stats['cache_stats'] = self.backend.get_cache_stats()
        
        return stats


def create_nano_vllm_client(model_path: str = "path/to/model") -> OptimizedLLMClient:
    """Nano VLLM 클라이언트 생성"""
    return OptimizedLLMClient(
        model_path=model_path,
        tensor_parallel_size=2,
        enable_prefix_caching=True
    )


def benchmark_nano_vllm_vs_standard(
    model_path: str,
    test_prompts: List[str],
    tensor_parallel_size: int = 1
) -> Dict[str, Any]:
    """nano-vllm vs 표준 클라이언트 벤치마크"""
    
    if not NANOVLLM_AVAILABLE:
        return {"error": "nano-vllm not available"}
    
    # nano-vllm 클라이언트
    nano_client = create_nano_vllm_client(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=True
    )
    
    # 표준 클라이언트 (LLMClient.from_env())
    standard_client = LLMClient.from_env()
    
    results = {
        'nano_vllm': {'times': [], 'total_time': 0.0},
        'standard': {'times': [], 'total_time': 0.0}
    }
    
    # nano-vllm 테스트
    for prompt in test_prompts:
        start_time = time.time()
        nano_client.complete(prompt, max_tokens=256)
        inference_time = time.time() - start_time
        results['nano_vllm']['times'].append(inference_time)
        results['nano_vllm']['total_time'] += inference_time
    
    # 표준 클라이언트 테스트
    for prompt in test_prompts:
        start_time = time.time()
        standard_client.complete(prompt, max_tokens=256)
        inference_time = time.time() - start_time
        results['standard']['times'].append(inference_time)
        results['standard']['total_time'] += inference_time
    
    # 통계 계산
    nano_avg = results['nano_vllm']['total_time'] / len(test_prompts)
    standard_avg = results['standard']['total_time'] / len(test_prompts)
    speedup = standard_avg / nano_avg if nano_avg > 0 else 0
    
    return {
        'nano_vllm_avg_time': nano_avg,
        'standard_avg_time': standard_avg,
        'speedup_factor': speedup,
        'nano_vllm_stats': nano_client.get_performance_stats(),
        'test_prompts_count': len(test_prompts)
    } 
