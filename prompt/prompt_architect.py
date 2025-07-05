"""
harin.prompt.prompt_architect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 5-1: 프롬프트 통합 생성기
• 판단 경로 + 감정 + 리듬 + 자아 상태 + 기억 요약을 통합하여 LLM 전달 프롬프트 생성
• 데이터 메모리 컨텍스트 통합
"""

from typing import List, Dict, Optional

class PromptArchitect:
    """프롬프트 아키텍트"""
    
    def __init__(self):
        self.prompt_history = []
    
    def build_prompt(self, user_input: str = None, context=None, prior_reasoning: str = None, 
                    reflection: str = None, v8_mode: bool = False, contextual_mode: bool = False, **kwargs):
        """프롬프트 생성 (V8/Contextual 옵션 지원)"""
        if contextual_mode:
            architect = ContextualPromptArchitect()
            return architect.build_prompt(kwargs.get('memory_path'), kwargs.get('agent_thoughts'), context or {}, **kwargs)
        elif v8_mode:
            architect = PromptArchitectV8()
            return architect.build_prompt(kwargs.get('memory_path'), kwargs.get('agent_thoughts'), context or {}, **kwargs)
        # 기존 방식
        return build_prompt_with_context(user_input, context, prior_reasoning, reflection)

    def build_prompt_v8(self, memory_path=None, agent_thoughts=None, context=None, **kwargs):
        architect = PromptArchitectV8()
        return architect.build_prompt(memory_path, agent_thoughts, context or {}, **kwargs)

    def build_prompt_contextual(self, memory_path=None, agent_thoughts=None, context=None, **kwargs):
        architect = ContextualPromptArchitect()
        return architect.build_prompt(memory_path, agent_thoughts, context or {}, **kwargs)


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

# === V8 프롬프트 구조 통합 ===
class PromptArchitectV8:
    def __init__(self):
        pass

    def build_prompt(self,
                     memory_path: List = None,
                     agent_thoughts: List[Dict] = None,
                     context: Dict = None,
                     emotion: Optional[List[float]] = None,
                     responsibility_score: float = 1.0) -> Dict[str, str]:
        memory_path = memory_path or []
        agent_thoughts = agent_thoughts or []
        context = context or {}
        system_prompt = self._build_system(context, responsibility_score)
        user_prompt = self._build_user(memory_path)
        assistant_prompt = self._build_assistant(agent_thoughts)
        return {
            "system": system_prompt,
            "user": user_prompt,
            "assistant": assistant_prompt
        }

    def _build_system(self, context: Dict, responsibility_score: float) -> str:
        tone = "neutral"
        if responsibility_score < 0.7:
            tone = "reflective"
        elif responsibility_score < 0.9:
            tone = "considerate"
        return f"You are Harin, responding with tone: {tone}. Context: {context.get('topic', 'general')}"

    def _build_user(self, memory_path: List) -> str:
        summary = [f"- {getattr(node, 'text', str(node))}" for node in memory_path[-3:]]
        return "Here are recent related thoughts:\n" + "\n".join(summary)

    def _build_assistant(self, agent_thoughts: List[Dict]) -> str:
        lines = [f"{d.get('agent_name', '')}: {d.get('content', '')}" for d in agent_thoughts]
        return "Insights from internal dialogue:\n" + "\n".join(lines)

# === 문맥/상상력 기반 프롬프트 생성 확장 ===
class ContextualPromptArchitect(PromptArchitectV8):
    def build_prompt(self, memory_path=None, agent_thoughts=None, context=None, master_prompt_mode: bool = False, identity_mode: bool = False, **kwargs):
        memory_path = memory_path or []
        agent_thoughts = agent_thoughts or []
        context = context or {}
        # master_prompt_mode가 True면 architect.PromptArchitect의 마스터 프롬프트 생성 호출
        if master_prompt_mode:
            try:
                from prompt.architect import PromptArchitect as MasterPromptArchitect
                identity_mgr = kwargs.get('identity_mgr')
                memory_engine = kwargs.get('memory_engine')
                flow = kwargs.get('flow')
                user_profile = kwargs.get('user_profile', {})
                master = MasterPromptArchitect(identity_mgr=identity_mgr, memory_engine=memory_engine)
                return {'system': master.build_master_prompt(flow, user_profile)}
            except Exception as e:
                return {'system': f'[마스터 프롬프트 생성 실패: {e}]'}
        # identity_mode가 True면 prompt_identity_architect의 SYSTEM 프롬프트 생성 호출
        system_identity = None
        if identity_mode:
            try:
                from prompt.prompt_identity_architect import PromptArchitect as IdentityPromptArchitect
                state = kwargs.get('state')
                identity = kwargs.get('identity')
                topic = kwargs.get('topic')
                if state and identity:
                    identity_architect = IdentityPromptArchitect(state, identity)
                    system_identity = identity_architect.build_system_prompt(topic=topic)
            except Exception as e:
                system_identity = f'[아이덴티티 프롬프트 생성 실패: {e}]'
        # 1. 키워드/상황 해석
        prompt_types = self._infer_prompt_types(memory_path, context)
        # 2. 각 유형별 프롬프트 생성
        prompts = []
        for ptype in prompt_types:
            if ptype == "memory_verification":
                prompts.append("이 응답이 과거 기억과 일치하는지 검토해줘.")
            elif ptype == "montecarlo":
                prompts.append("가능한 여러 시나리오를 상상해보고 각각의 결과를 예측해줘.")
            elif ptype == "reflection":
                prompts.append("이 응답에 논리적 오류나 왜곡이 있는지 반성해봐.")
            elif ptype == "summary":
                prompts.append("최근 기억과 대화 흐름을 요약해줘.")
            elif ptype == "scenario_generation":
                prompts.append("새로운 해결책이나 시나리오를 상상해서 제안해줘.")
        # 3. 조합
        base_prompts = super().build_prompt(memory_path, agent_thoughts, context, **kwargs)
        if prompts:
            base_prompts['system'] += '\n' + '\n'.join(prompts)
        # 4. plan/metacog 정보가 있으면 PromptSynth로 합성
        plan = kwargs.get('plan')
        metacog = kwargs.get('metacog')
        if plan and metacog:
            try:
                from prompt.generator import PromptSynth
                synth = PromptSynth()
                synth_prompt = synth.assemble(plan, metacog)
                base_prompts['system'] += '\n' + synth_prompt
            except Exception as e:
                base_prompts['system'] += f'\n[프롬프트 합성 실패: {e}]'
        # 5. identity/rhythm 기반 SYSTEM 프롬프트 추가
        if system_identity:
            base_prompts['system'] = system_identity + '\n' + base_prompts['system']
        return base_prompts

    def _infer_prompt_types(self, memory_path, context):
        types = []
        tags = set()
        for node in memory_path:
            tags.update(getattr(node, 'tags', []))
        # scar 기반 반성
        if any('scar' in t for t in tags):
            types.append("reflection")
        # 몬테카를로 시뮬레이션
        if "montecarlo" in tags:
            types.append("montecarlo")
        # 기억 검증
        if "memory_check" in tags or context.get("need_memory_verification"):
            types.append("memory_verification")
        # 요약 필요
        if "summary" in tags or context.get("need_summary"):
            types.append("summary")
        # 시나리오 생성
        if "scenario" in tags or context.get("need_scenario"):
            types.append("scenario_generation")
        return types
