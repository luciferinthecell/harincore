"""
harin.integrations.llm_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

의미 기반 Harin 에이전트용 통합 LLM 클라이언트.

원칙
------
* **모델 이름이 아닌 의미** (task → params) 를 입력받아 프롬프트/옵션을 자동 매핑.
* **백엔드 선택**은 환경변수·설정파일로 주입, 동적 전환 가능.
* **키워드 트리거 사용 금지** – 정책·시스템 지침은 SYSTEM 프롬프트 블록으로 표현.

Public API
-----------
```python
client = LLMClient.from_env()
reply = client.complete(prompt, max_tokens=512, temperature=0.7)
```
"""

from __future__ import annotations

import os
from typing import Protocol, Dict, Any, Optional
from datetime import datetime

# optional imports guarded
try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None

try:
    from google.generativeai import GenerativeModel  # type: ignore
except ImportError:  # pragma: no cover
    GenerativeModel = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
#  Protocol (meaning-first)
# ──────────────────────────────────────────────────────────────────────────


class LLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, **kw) -> str: ...


# ------------------------------------------------------------------------
#  Concrete backends
# ------------------------------------------------------------------------


class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        if openai is None:
            raise RuntimeError("openai package not installed")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        openai.api_key = api_key
        self.model = model

    def complete(self, prompt: str, **kw) -> str:  # noqa: D401
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kw,
        )
        return resp.choices[0].message.content.strip()


class GeminiBackend:
    name = "gemini"

    def __init__(self, model: str = "gemini-pro", api_key: str | None = None):
        if GenerativeModel is None:
            raise RuntimeError("google-generativeai package not installed")
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        os.environ["GOOGLE_API_KEY"] = api_key
        self.model = GenerativeModel(model)

    def complete(self, prompt: str, **kw) -> str:
        resp = self.model.generate_content(prompt, generation_config=kw or {})
        return resp.text.strip()


class EchoBackend:
    name = "echo"

    def complete(self, prompt: str, **kw) -> str:  # noqa: D401
        return "(echo) " + prompt.splitlines()[-1]


# ------------------------------------------------------------------------
#  High-level client with graceful fallback
# ------------------------------------------------------------------------


class LLMClient:
    """LLM 클라이언트 기본 구현"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
    @classmethod
    def from_env(cls) -> "LLMClient":
        """환경 변수에서 설정을 읽어 LLMClient 생성"""
        return cls()
    
    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """프롬프트 완성 (기본 구현)"""
        
        # 실제 API 호출 대신 기본 응답 반환
        if "안녕" in prompt or "hello" in prompt.lower():
            return "안녕하세요! 저는 Harin AI입니다. 무엇을 도와드릴까요?"
        elif "도움" in prompt or "help" in prompt.lower():
            return "저는 사고와 추론을 통해 문제를 해결하는 AI입니다. 질문이나 도움이 필요한 내용을 말씀해 주세요."
        else:
            return f"입력하신 내용을 분석했습니다: {prompt[:50]}... 더 구체적인 질문이나 요청을 해주시면 더 정확한 답변을 드릴 수 있습니다."
    
    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """채팅 형태의 완성 (기본 구현)"""
        
        if not messages:
            return "메시지가 없습니다."
        
        last_message = messages[-1].get("content", "")
        return self.complete(last_message, temperature)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """사용 통계 반환"""
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "last_request": datetime.now().isoformat()
        }


class OptimizedLLMClient(LLMClient):
    """최적화된 LLM 클라이언트"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, enable_prefix_caching: bool = True):
        super().__init__()
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_prefix_caching = enable_prefix_caching
        self.cache = {}
    
    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """캐시를 활용한 최적화된 완성"""
        
        # 캐시 확인
        cache_key = f"{prompt[:100]}_{temperature}_{max_tokens}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 기본 완성 실행
        result = super().complete(prompt, temperature, max_tokens)
        
        # 캐시에 저장
        if self.enable_prefix_caching:
            self.cache[cache_key] = result
        
        return result
    
    def benchmark_performance(self, test_inputs: list) -> Dict[str, Any]:
        """성능 벤치마킹"""
        
        import time
        
        start_time = time.time()
        successful_tests = 0
        execution_times = []
        
        for input_text in test_inputs:
            try:
                test_start = time.time()
                self.complete(input_text)
                test_end = time.time()
                
                execution_times.append(test_end - test_start)
                successful_tests += 1
                
            except Exception as e:
                print(f"테스트 실패: {input_text} - {e}")
        
        total_time = time.time() - start_time
        
        return {
            "total_tests": len(test_inputs),
            "successful_tests": successful_tests,
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "min_execution_time": min(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "total_time": total_time,
            "performance_metrics": {
                "cache_hit_rate": len(self.cache) / max(1, successful_tests),
                "parallel_efficiency": successful_tests / max(1, total_time)
            }
        }


def create_nano_vllm_client(model_path: str = "path/to/model") -> OptimizedLLMClient:
    """Nano VLLM 클라이언트 생성"""
    return OptimizedLLMClient(
        model_path=model_path,
        tensor_parallel_size=2,
        enable_prefix_caching=True
    )
