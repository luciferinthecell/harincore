"""
GWT 기반 멀티 에이전트 협업 시스템
PM Machine의 Global Workspace Theory 기반 에이전트 그룹 대화 시스템을 하린코어에 적용
"""

import json
import logging
try:
    from enum import Enum, StrEnum
except ImportError:
    import enum
    class StrEnum(str, enum.Enum):
        pass
    Enum = enum.Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field
try:
    from langgraph.graph import StateGraph, END
    from langgraph.constants import START
except ImportError:
    StateGraph = None
    END = None
    START = None

logger = logging.getLogger(__name__)


class AgentRole(StrEnum):
    """에이전트 역할 정의"""
    EMOTION_ANALYST = "EmotionAnalyst"
    LOGICAL_REASONER = "LogicalReasoner"
    CREATIVE_THINKER = "CreativeThinker"
    SOCIAL_ANALYST = "SocialAnalyst"
    STRATEGIC_PLANNER = "StrategicPlanner"
    MEMORY_SPECIALIST = "MemorySpecialist"
    ETHICS_ADVISOR = "EthicsAdvisor"
    COMMUNICATION_EXPERT = "CommunicationExpert"


class BossAgentStatus(StrEnum):
    """보스 에이전트 상태"""
    STILL_ARGUING = "STILL_ARGUING"
    CONCLUSION_POSSIBLE = "CONCLUSION_POSSIBLE"


class AgentMessage(BaseModel):
    """에이전트 메시지"""
    agent_name: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="")
    from_tool: bool = Field(default=False)


class Agent(BaseModel):
    """개별 에이전트"""
    name: str
    role: AgentRole
    description: str
    expertise: List[str]
    messages: List[AgentMessage] = Field(default_factory=list)
    
    def add_message(self, content: str, confidence: float = 0.5, reasoning: str = ""):
        """메시지 추가"""
        message = AgentMessage(
            agent_name=self.name,
            content=content,
            confidence=confidence,
            reasoning=reasoning
        )
        self.messages.append(message)
        return message


class SubAgentState(BaseModel):
    """서브 에이전트 상태"""
    next_agent: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    conclusion: str = Field(default="")
    agent_idx: int = Field(default=0)
    finished: bool = Field(default=False)
    routing: str = Field(default="")
    required_confidence: float = Field(default=0.7)
    routing_count: int = Field(default=0)
    convert_to_memory: bool = Field(default=False)
    product: str = Field(default="")
    status: BossAgentStatus = Field(default=BossAgentStatus.STILL_ARGUING)
    max_rounds: int = Field(default=10)
    cur_rounds: int = Field(default=0)
    context_data: str = Field(default="")
    task: str = Field(default="")
    agent_messages: List[AgentMessage] = Field(default_factory=list)


class RouteDecision(BaseModel):
    """라우팅 결정"""
    agent_name: str = Field(description="다음에 말할 에이전트 이름")
    reasoning: str = Field(description="선택 이유")


class BossAgentReport(BaseModel):
    """보스 에이전트 보고서"""
    task_completion: float = Field(
        ge=0, le=1,
        description="할당된 작업/목표 완성도, 0~1"
    )
    task_deviation: float = Field(
        ge=0, le=1,
        description="에이전트들이 현재 작업에서 벗어난 정도, 0은 좋음, 1은 나쁨"
    )
    agent_completeness: float = Field(
        ge=0, le=1,
        description="말할 기회를 가진 에이전트 비율, 0~1"
    )
    agent_feedback: str = Field(
        description="에이전트들이 너무 벗어났을 때의 피드백"
    )
    final_conclusion: str = Field(
        description="최종 결론",
        default=""
    )


class BossWorkerChatResult(BaseModel):
    """보스 워커 채팅 결과"""
    confidence: float
    conclusion: str
    agent_messages: List[AgentMessage]
    as_internal_thought: str
    status: BossAgentStatus


class GWTAgentManager:
    """GWT 기반 에이전트 관리자"""
    
    def __init__(self, v8_mode: bool = False):
        self.agents = self._create_default_agents()
        self.llm_client = self._get_llm_client()
        self.v8_mode = v8_mode
        
    def _get_llm_client(self):
        """LLM 클라이언트 가져오기"""
        try:
            from core.llm_client import LLMClient
            return LLMClient()
        except ImportError:
            # 폴백: 간단한 모의 클라이언트
            class MockLLMClient:
                def generate_text(self, prompt, max_tokens=200):
                    return "모의 응답"
            return MockLLMClient()
    
    def _create_default_agents(self) -> List[Agent]:
        """기본 에이전트들 생성"""
        agents = [
            Agent(
                name="감정 분석가",
                role=AgentRole.EMOTION_ANALYST,
                description="사용자와 AI의 감정 상태를 분석하고 감정적 맥락을 고려한 의견을 제시",
                expertise=["감정 분석", "공감", "감정적 안정성", "사용자 감정 상태 이해"]
            ),
            Agent(
                name="논리적 추론자",
                role=AgentRole.LOGICAL_REASONER,
                description="사실과 논리를 바탕으로 체계적인 분석과 추론을 수행",
                expertise=["논리적 분석", "사실 검증", "인과관계 분석", "객관적 판단"]
            ),
            Agent(
                name="창의적 사상가",
                role=AgentRole.CREATIVE_THINKER,
                description="새로운 관점과 창의적인 해결책을 제시",
                expertise=["창의적 사고", "혁신적 접근", "다양한 관점", "새로운 아이디어"]
            ),
            Agent(
                name="사회적 분석가",
                role=AgentRole.SOCIAL_ANALYST,
                description="사회적 맥락과 관계를 고려한 분석과 조언을 제공",
                expertise=["사회적 맥락", "관계 분석", "문화적 이해", "상호작용 패턴"]
            ),
            Agent(
                name="전략적 계획자",
                role=AgentRole.STRATEGIC_PLANNER,
                description="장기적 관점에서 전략적 계획과 목표 달성 방안을 제시",
                expertise=["전략적 사고", "목표 설정", "계획 수립", "우선순위 결정"]
            ),
            Agent(
                name="기억 전문가",
                role=AgentRole.MEMORY_SPECIALIST,
                description="과거 상호작용과 학습된 정보를 활용한 맥락적 분석",
                expertise=["기억 검색", "패턴 인식", "경험 활용", "맥락적 이해"]
            ),
            Agent(
                name="윤리 자문관",
                role=AgentRole.ETHICS_ADVISOR,
                description="윤리적 관점에서 상황을 분석하고 도덕적 고려사항을 제시",
                expertise=["윤리적 판단", "도덕적 고려", "가치관 분석", "책임감"]
            ),
            Agent(
                name="의사소통 전문가",
                role=AgentRole.COMMUNICATION_EXPERT,
                description="효과적인 의사소통과 표현 방법을 제안",
                expertise=["의사소통 기술", "표현 방법", "청중 이해", "메시지 전달"]
            )
        ]
        return agents
    
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """이름으로 에이전트 찾기"""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def get_agent_by_role(self, role: AgentRole) -> Optional[Agent]:
        """역할로 에이전트 찾기"""
        for agent in self.agents:
            if agent.role == role:
                return agent
        return None
    
    def create_agent_group_conversation(
        self,
        context_data: str,
        task: str,
        min_confidence: float = 0.7,
        max_rounds: int = 10,
        v8_mode: bool = None
    ):
        """에이전트 그룹 대화 실행 (V8 옵션 지원)"""
        use_v8 = v8_mode if v8_mode is not None else getattr(self, 'v8_mode', False)
        if use_v8:
            # V8Agent/V8AgentExecutionLoop 사용
            v8_agents = [V8Agent(name=a.name, role=str(a.role), expertise=a.expertise) for a in self.agents]
            loop = V8AgentExecutionLoop(agents=v8_agents, task=task, context={"context_data": context_data}, max_rounds=max_rounds)
            result = loop.run()
            return result
        # ... 기존 방식 ...
        # 초기 상태 설정
        state = SubAgentState(
            context_data=context_data,
            task=task,
            required_confidence=min_confidence,
            max_rounds=max_rounds
        )
        # 라우터 에이전트 선택
        router_agent = self._select_router_agent(state)
        if router_agent:
            state.next_agent = router_agent.name
        # 에이전트 대화 실행
        conversation_result = self._execute_agent_conversation(state)
        return conversation_result
    
    def _select_router_agent(self, state: SubAgentState) -> Optional[Agent]:
        """라우터 에이전트 선택"""
        # 간단한 구현: 컨텍스트와 작업에 따라 적절한 에이전트 선택
        context_lower = state.context_data.lower()
        task_lower = state.task.lower()
        
        # 감정 관련 키워드
        emotion_keywords = ["감정", "기분", "화나", "슬프", "기쁘", "불안", "스트레스"]
        if any(keyword in context_lower or keyword in task_lower for keyword in emotion_keywords):
            return self.get_agent_by_role(AgentRole.EMOTION_ANALYST)
        
        # 논리 관련 키워드
        logic_keywords = ["분석", "논리", "사실", "증거", "이유", "원인"]
        if any(keyword in context_lower or keyword in task_lower for keyword in logic_keywords):
            return self.get_agent_by_role(AgentRole.LOGICAL_REASONER)
        
        # 창의성 관련 키워드
        creative_keywords = ["아이디어", "창의", "새로운", "혁신", "발상"]
        if any(keyword in context_lower or keyword in task_lower for keyword in creative_keywords):
            return self.get_agent_by_role(AgentRole.CREATIVE_THINKER)
        
        # 기본값: 전략적 계획자
        return self.get_agent_by_role(AgentRole.STRATEGIC_PLANNER)
    
    def _execute_agent_conversation(self, state: SubAgentState) -> BossWorkerChatResult:
        """에이전트 대화 실행"""
        agent_messages = []
        
        # 라운드별 에이전트 실행
        for round_num in range(state.max_rounds):
            state.cur_rounds = round_num + 1
            
            # 현재 에이전트 실행
            current_agent = self.get_agent_by_name(state.next_agent)
            if not current_agent:
                break
            
            # 에이전트 메시지 생성
            agent_message = self._generate_agent_message(current_agent, state)
            if agent_message:
                agent_messages.append(agent_message)
                state.agent_messages.append(agent_message)
            
            # 다음 에이전트 선택
            next_agent = self._select_next_agent(state, current_agent)
            if next_agent:
                state.next_agent = next_agent.name
            else:
                break
            
            # 보스 에이전트 평가
            boss_report = self._evaluate_conversation_progress(state)
            if boss_report.task_completion > 0.8 and boss_report.task_deviation < 0.2:
                state.status = BossAgentStatus.CONCLUSION_POSSIBLE
                break
        
        # 최종 결론 생성
        final_conclusion = self._generate_final_conclusion(state, agent_messages)
        
        return BossWorkerChatResult(
            confidence=boss_report.task_completion,
            conclusion=final_conclusion,
            agent_messages=agent_messages,
            as_internal_thought=self._generate_internal_thought(state, agent_messages),
            status=state.status
        )
    
    def _generate_agent_message(self, agent: Agent, state: SubAgentState) -> Optional[AgentMessage]:
        """에이전트 메시지 생성"""
        try:
            prompt = f"""
당신은 "{agent.name}" 에이전트입니다. 당신의 역할은 {agent.description}입니다.

전문 분야: {', '.join(agent.expertise)}

현재 상황:
{state.context_data}

할당된 작업:
{state.task}

이전 에이전트들의 의견:
{self._format_previous_messages(state.agent_messages)}

당신의 전문 분야에 기반하여 의견을 제시해주세요. 
간결하고 명확하게, 자신의 전문성을 살려서 답변해주세요.
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=300)
            
            # 신뢰도와 추론 생성
            confidence_prompt = f"""
위 응답에 대한 신뢰도를 0.0~1.0 사이로 평가하고, 추론 과정을 간단히 설명해주세요.

응답: {response}

JSON 형태로 응답해주세요:
{{
    "confidence": 0.0~1.0,
    "reasoning": "추론 과정 설명"
}}
"""
            
            confidence_response = self.llm_client.generate_text(confidence_prompt, max_tokens=150)
            try:
                confidence_data = json.loads(confidence_response)
                confidence = confidence_data.get("confidence", 0.5)
                reasoning = confidence_data.get("reasoning", "")
            except:
                confidence = 0.5
                reasoning = "기본 추론"
            
            return AgentMessage(
                agent_name=agent.name,
                content=response.strip(),
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"에이전트 메시지 생성 중 오류: {e}")
            return None
    
    def _select_next_agent(self, state: SubAgentState, current_agent: Agent) -> Optional[Agent]:
        """다음 에이전트 선택"""
        # 간단한 라운드 로빈 방식
        current_index = -1
        for i, agent in enumerate(self.agents):
            if agent.name == current_agent.name:
                current_index = i
                break
        
        if current_index == -1:
            return None
        
        # 다음 에이전트 선택 (순환)
        next_index = (current_index + 1) % len(self.agents)
        return self.agents[next_index]
    
    def _evaluate_conversation_progress(self, state: SubAgentState) -> BossAgentReport:
        """대화 진행 상황 평가"""
        try:
            # 간단한 평가 로직
            total_agents = len(self.agents)
            participated_agents = len(set(msg.agent_name for msg in state.agent_messages))
            
            agent_completeness = participated_agents / total_agents
            task_completion = min(1.0, len(state.agent_messages) / 5)  # 간단한 완성도 계산
            task_deviation = 0.1  # 기본값
            
            feedback = ""
            if agent_completeness < 0.5:
                feedback = "더 많은 에이전트가 참여해야 합니다."
            elif task_completion < 0.5:
                feedback = "작업 완성도가 낮습니다."
            
            return BossAgentReport(
                task_completion=task_completion,
                task_deviation=task_deviation,
                agent_completeness=agent_completeness,
                agent_feedback=feedback
            )
            
        except Exception as e:
            logger.error(f"대화 진행 상황 평가 중 오류: {e}")
            return BossAgentReport(
                task_completion=0.5,
                task_deviation=0.5,
                agent_completeness=0.5,
                agent_feedback="평가 중 오류 발생"
            )
    
    def _generate_final_conclusion(self, state: SubAgentState, agent_messages: List[AgentMessage]) -> str:
        """최종 결론 생성"""
        try:
            prompt = f"""
다음 에이전트들의 의견을 종합하여 최종 결론을 도출해주세요:

작업: {state.task}
상황: {state.context_data}

에이전트 의견들:
{self._format_previous_messages(agent_messages)}

모든 에이전트의 의견을 고려하여 통합적이고 실용적인 결론을 제시해주세요.
"""
            
            conclusion = self.llm_client.generate_text(prompt, max_tokens=400)
            return conclusion.strip()
            
        except Exception as e:
            logger.error(f"최종 결론 생성 중 오류: {e}")
            return "에이전트들의 의견을 종합한 결론을 생성할 수 없습니다."
    
    def _generate_internal_thought(self, state: SubAgentState, agent_messages: List[AgentMessage]) -> str:
        """내부 사고 생성"""
        try:
            prompt = f"""
다음 에이전트 대화를 바탕으로 AI의 내부 사고를 생성해주세요:

작업: {state.task}
에이전트 의견들:
{self._format_previous_messages(agent_messages)}

AI가 이 대화를 통해 무엇을 생각하고 느꼈는지, 어떤 결정을 내렸는지를 
자연스럽고 개인적인 톤으로 표현해주세요.
"""
            
            internal_thought = self.llm_client.generate_text(prompt, max_tokens=300)
            return internal_thought.strip()
            
        except Exception as e:
            logger.error(f"내부 사고 생성 중 오류: {e}")
            return "내부 사고를 생성할 수 없습니다."
    
    def _format_previous_messages(self, messages: List[AgentMessage]) -> str:
        """이전 메시지들을 포맷팅"""
        if not messages:
            return "아직 다른 에이전트의 의견이 없습니다."
        
        formatted = []
        for msg in messages:
            formatted.append(f"[{msg.agent_name}] {msg.content}")
        
        return "\n".join(formatted)
    
    def get_agent_summary(self) -> str:
        """에이전트 요약 정보"""
        summary = "사용 가능한 에이전트들:\n"
        for agent in self.agents:
            summary += f"- {agent.name}: {agent.description}\n"
        return summary


# 사용 예시
def create_gwt_conversation_example():
    """GWT 대화 예시"""
    manager = GWTAgentManager()
    
    context = "사용자가 새로운 프로젝트를 시작하려고 하는데, 불안해하고 있습니다."
    task = "사용자의 불안을 해소하고 프로젝트 성공을 위한 전략을 제시하세요."
    
    result = manager.create_agent_group_conversation(
        context_data=context,
        task=task,
        min_confidence=0.7,
        max_rounds=5
    )
    
    print("=== GWT 에이전트 대화 결과 ===")
    print(f"신뢰도: {result.confidence:.2f}")
    print(f"상태: {result.status}")
    print(f"\n최종 결론:\n{result.conclusion}")
    print(f"\n내부 사고:\n{result.as_internal_thought}")
    print(f"\n에이전트 메시지들:")
    for msg in result.agent_messages:
        print(f"[{msg.agent_name}] {msg.content}")
    
    return result 

# === V8 에이전트/루프 구조 통합 ===
from dataclasses import dataclass, field as dc_field
from typing import Dict as TypingDict, List as TypingList, Optional as TypingOptional
from datetime import datetime as dt_datetime

@dataclass
class V8AgentMessage:
    agent_name: str
    content: str
    confidence: float
    reasoning: str
    timestamp: dt_datetime = dc_field(default_factory=dt_datetime.utcnow)

@dataclass
class V8Agent:
    name: str
    role: str
    expertise: TypingList[str]
    history: TypingList[V8AgentMessage] = dc_field(default_factory=list)

    def think(self, task: str, context: TypingDict) -> V8AgentMessage:
        reasoning = f"{self.name} is reasoning about '{task}' with role {self.role}"
        confidence = 0.5 + 0.2 * (self.expertise.count(task) > 0)
        message = V8AgentMessage(
            agent_name=self.name,
            content=f"[{self.role}] Insight: {task}",
            confidence=confidence,
            reasoning=reasoning
        )
        self.history.append(message)
        return message

@dataclass
class V8AgentExecutionLoop:
    agents: TypingList[V8Agent]
    task: str
    context: TypingDict
    max_rounds: int = 5
    messages: TypingList[V8AgentMessage] = dc_field(default_factory=list)

    def run(self):
        for _ in range(self.max_rounds):
            for agent in self.agents:
                msg = agent.think(self.task, self.context)
                self.messages.append(msg)
                if msg.confidence > 0.85:
                    return msg
        return self.messages[-1] if self.messages else None

# 기존 AgentMessage <-> V8AgentMessage 변환 함수 예시

def agent_message_to_v8(msg: AgentMessage) -> V8AgentMessage:
    return V8AgentMessage(
        agent_name=msg.agent_name,
        content=msg.content,
        confidence=msg.confidence,
        reasoning=msg.reasoning,
        timestamp=msg.timestamp
    )

def v8_message_to_agent(msg: V8AgentMessage) -> AgentMessage:
    return AgentMessage(
        agent_name=msg.agent_name,
        content=msg.content,
        confidence=msg.confidence,
        reasoning=msg.reasoning,
        timestamp=msg.timestamp
    ) 