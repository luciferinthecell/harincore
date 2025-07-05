from __future__ import annotations
import uuid, logging
from typing import List, Dict, Any, Optional
from core.loop_manager import LoopManager
from core.evaluator import TrustEvaluator
from core.conscience_cluster import ConscienceCluster
from memory.memory_conductor import MemoryConductor
from memory.engine import MemoryEngine
from prompt.prompt_architect import PromptArchitect
from tools.llm_client import LLMClient

log = logging.getLogger("harin.recursive")

class RecursiveReasoner:
    def __init__(self, loop_manager: LoopManager, conductor: MemoryConductor,
                 trust: TrustEvaluator, conscience: ConscienceCluster,
                 prompt_arch: PromptArchitect, llm: LLMClient,
                 memory_engine: Optional[MemoryEngine] = None):
        self.loops = loop_manager
        self.conductor = conductor
        self.trust = trust
        self.conscience = conscience
        self.arch = prompt_arch
        self.llm = llm
        self.memory_engine = memory_engine

    def run(self, user_input: str, context) -> Dict[str, Any]:
        cycle_id = f"RR-{uuid.uuid4().hex[:8]}"
        
        # 데이터 메모리 컨텍스트 조회
        memory_context = {}
        if self.memory_engine:
            memory_context = self.memory_engine.get_data_memory_context(user_input)
            log.info(f"메모리 컨텍스트 로드: {memory_context.get('context_summary', 'N/A')}")
        
        # 컨텍스트에 메모리 정보 추가
        context.memory_context = memory_context
        
        judgments = self.loops.run_all(user_input, context, self.conductor)
        best = max(judgments, key=lambda j: j.score.overall())

        reflection = f"[Reflection {cycle_id}] {best.loop_id}, score={best.score.dict()}"
        context.memory.store(reflection, node_type="reflection", meta={"cycle": cycle_id})

        if self.trust.needs_rerun(best):
            alt = self.loops.run_fallback(user_input, context, self.conductor)
            if alt and alt.score.overall() > best.score.overall():
                best = alt
                context.memory.store(f"[Correction] Fallback loop selected ({alt.loop_id})",
                                     node_type="reflection", meta={"cycle": cycle_id, "phase": "fallback"})

        conscience_ok, flags = self.conscience.validate(best.answer, context)
        if not conscience_ok:
            best.answer = self.conscience.correct(best.answer, context)

        prompt = self.arch.build_prompt(user_input, context, prior_reasoning=best.answer, reflection=reflection)
        final = self.llm.complete(prompt, temperature=0.7)
        context.memory.store(final, node_type="assistant", meta={"cycle": cycle_id})
        
        # 새로운 메모리 노드 추가 (중요한 사고 결과)
        if self.memory_engine and memory_context.get("relevant_memories"):
            self._add_thinking_memory(user_input, final, cycle_id, best.loop_id)

        return {
            "answer": final, 
            "score": best.score.overall(), 
            "conscience_flags": flags, 
            "cycle": cycle_id,
            "memory_context": memory_context
        }
    
    def _add_thinking_memory(self, user_input: str, answer: str, cycle_id: str, loop_id: str):
        """사고 결과를 데이터 메모리에 추가"""
        try:
            # 사고 추적 메모리 추가
            thinking_content = f"사용자: {user_input}\n하린: {answer}\n루프: {loop_id}"
            
            self.memory_engine.add_data_memory_node(
                content=thinking_content,
                memory_type="thought_trace",
                tags=[f"loop.{loop_id}", "thinking.cycle", f"cycle.{cycle_id}"],
                context={
                    "source": "recursive_reasoner",
                    "trigger": user_input,
                    "linked_loops": [loop_id],
                    "importance": 0.8,
                    "reason_for_memory": "재귀 추론 결과로 생성된 사고 흐름"
                },
                source_file="h2"
            )
            
            log.info(f"사고 메모리 추가됨: cycle_id={cycle_id}, loop_id={loop_id}")
            
        except Exception as e:
            log.error(f"메모리 추가 실패: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        if self.memory_engine:
            return self.memory_engine.get_memory_stats()
        return {"error": "메모리 엔진이 초기화되지 않음"}
