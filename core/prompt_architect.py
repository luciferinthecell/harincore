"""
harin.prompt.prompt_architect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 5-1: 프롬프트 통합 생성기
• 판단 경로 + 감정 + 리듬 + 자아 상태 + 기억 요약을 통합하여 LLM 전달 프롬프트 생성
• 데이터 메모리 컨텍스트 통합
"""


class PromptArchitect:
    """프롬프트 아키텍트"""
    
    def __init__(self):
        self.prompt_history = []
    
    def build_prompt(self, user_input: str, context, prior_reasoning: str = None, 
                    reflection: str = None) -> str:
        """프롬프트 생성"""
        return build_prompt_with_context(user_input, context, prior_reasoning, reflection)


def build_prompt(path: dict, emotion: str, rhythm: dict, identity: str, 
                memory_context: dict = None, prior_reasoning: str = None, 
                reflection: str = None) -> str:
    
    # 기본 프롬프트 구성
    prompt_parts = [
        f"[사고 목적]\n{path['statement']}\n",
        f"[감정] {emotion} / [리듬] 진실 {rhythm['truth']} / 공명 {rhythm['resonance']}",
        f"[자아 상태] {identity}\n"
    ]
    
    # 메모리 컨텍스트 추가
    if memory_context:
        memory_section = _build_memory_section(memory_context)
        if memory_section:
            prompt_parts.append(memory_section)
    
    # 이전 추론 결과 추가
    if prior_reasoning:
        prompt_parts.append(f"[이전 추론]\n{prior_reasoning}\n")
    
    # 반성 정보 추가
    if reflection:
        prompt_parts.append(f"[반성]\n{reflection}\n")
    
    prompt_parts.append("[지시] 위 명제에 따라 사고를 전개하고 신중한 판단을 내려라.")
    
    return "\n".join(prompt_parts)


def _build_memory_section(memory_context: dict) -> str:
    """메모리 컨텍스트를 프롬프트 섹션으로 변환"""
    sections = []
    
    # 관련 메모리들
    if memory_context.get("relevant_memories"):
        relevant_memories = memory_context["relevant_memories"]
        if relevant_memories:
            memory_texts = []
            for memory in relevant_memories[:3]:  # 상위 3개만
                memory_texts.append(f"• {memory['content'][:200]}...")
            
            sections.append("[관련 기억]\n" + "\n".join(memory_texts))
    
    # SCAR 메모리들 (오류 방지)
    if memory_context.get("scar_memories"):
        scar_memories = memory_context["scar_memories"]
        if scar_memories:
            scar_texts = []
            for scar in scar_memories[:2]:  # 상위 2개만
                scar_texts.append(f"⚠️ {scar['content'][:150]}...")
            
            sections.append("[주의사항]\n" + "\n".join(scar_texts))
    
    # 구조 메모리들
    if memory_context.get("structure_memories"):
        structure_memories = memory_context["structure_memories"]
        if structure_memories:
            structure_texts = []
            for structure in structure_memories[:2]:  # 상위 2개만
                structure_texts.append(f"🔧 {structure['content'][:150]}...")
            
            sections.append("[사고 구조]\n" + "\n".join(structure_texts))
    
    # 컨텍스트 요약
    
    # 핫 메모리 topic_summary 자동 삽입
    if memory_context and memory_context.get("hot_memory_topics"):
        topics = memory_context["hot_memory_topics"]
        if topics:
            sections.append("[핫 사고 흐름 요약]\n" + "\n".join(f"🧠 {t}" for t in topics[:2]))

    if memory_context.get("context_summary"):
        sections.append(f"[메모리 요약] {memory_context['context_summary']}")
    
    return "\n\n".join(sections) if sections else ""


def build_prompt_with_context(user_input: str, context, prior_reasoning: str = None, 
                            reflection: str = None) -> str:
    """컨텍스트 객체를 사용한 프롬프트 생성"""
    
    # 기본 정보 추출
    path = {"statement": "사용자 입력에 대한 존재 기반 사고 응답"}
    emotion = "신중한 공감"
    rhythm = {"truth": "존재적 진실", "resonance": "감정 공명"}
    identity = "존재 기반 사고 주체"
    
    # 메모리 컨텍스트 추출
    memory_context = getattr(context, 'memory_context', None)
    
    return build_prompt(
        path=path,
        emotion=emotion,
        rhythm=rhythm,
        identity=identity,
        memory_context=memory_context,
        prior_reasoning=prior_reasoning,
        reflection=reflection
    )
