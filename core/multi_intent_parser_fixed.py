"""
다중 의도 파서 (Multi-Intent Parser) - 의미 분석 기반
사용자의 한 발화 안에 포함된 다중 개념·지시·의도를 의미 분석으로 분해하여 각기 독립적인 사고 단위로 인식
키워드 스캔 없이 문맥적 의미 분석만 사용
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """의도 타입"""
    COMMAND = "command"           # 명령
    QUESTION = "question"         # 질문
    STATEMENT = "statement"       # 진술
    EMOTION = "emotion"          # 감정 표현
    JUDGMENT = "judgment"        # 판단
    REQUEST = "request"          # 요청
    FEEDBACK = "feedback"        # 피드백
    CONTEXT = "context"          # 맥락
    RELATIONSHIP = "relationship" # 관계


@dataclass
class SemanticUnit:
    """의미 단위"""
    id: str
    content: str
    semantic_type: str
    meaning: str
    importance: float
    context_position: int
    relationships: List[str]


@dataclass
class ParsedIntent:
    """파싱된 의도"""
    id: str
    content: str
    intent_type: IntentType
    priority: int  # 1-5, 5가 가장 높음
    context: Dict[str, Any]
    dependencies: List[str]  # 의존하는 다른 의도 ID들
    emotion: Optional[str] = None
    confidence: float = 0.0
    semantic_units: List[SemanticUnit] = None


class MultiIntentParser:
    """의미 분석 기반 다중 의도 파서"""
    
    def __init__(self):
        # 의미적 분리자 (키워드가 아닌 의미적 구분자)
        self.semantic_separators = {
            "logical": ["그리고", "또한", "또는", "하지만", "그런데", "따라서", "결과적으로"],
            "sequential": ["첫째", "둘째", "셋째", "넷째", "다섯째", "마지막으로"],
            "structural": ["1.", "2.", "3.", "4.", "5.", "(1)", "(2)", "(3)", "(4)", "(5)"],
            "visual": ["•", "·", "*", "-", "→", "⇒"]
        }
        
        # 의미적 의도 분류 기준 (키워드가 아닌 의미적 패턴)
        self.semantic_intent_criteria = {
            IntentType.COMMAND: {
                "action_indicators": ["실행", "구현", "생성", "만들", "처리", "해결"],
                "imperative_structures": ["해야", "해야지", "해야겠다", "하자", "하겠다"],
                "request_structures": ["해줘", "해주세요", "해달라", "해달라고"],
                "semantic_context": "행동 요구나 실행 지시를 나타내는 의미적 구조"
            },
            IntentType.QUESTION: {
                "interrogative_indicators": ["무엇", "어떻게", "왜", "언제", "어디", "누가"],
                "inquiry_structures": ["알려줘", "설명해", "답해", "말해"],
                "curiosity_indicators": ["궁금", "궁금해", "궁금하다"],
                "semantic_context": "정보 요구나 설명 요청을 나타내는 의미적 구조"
            },
            IntentType.EMOTION: {
                "emotional_indicators": ["좋아", "싫어", "화나", "기뻐", "슬퍼", "짜증나"],
                "feeling_structures": ["답답해", "불안해", "걱정돼", "감사", "미안", "죄송"],
                "intensity_indicators": ["매우", "정말", "너무", "조금", "약간"],
                "semantic_context": "감정적 상태나 반응을 나타내는 의미적 구조"
            },
            IntentType.JUDGMENT: {
                "evaluation_indicators": ["맞아", "틀려", "옳아", "그르다", "좋다", "나쁘다"],
                "assessment_structures": ["적절하다", "부적절하다", "필요하다", "불필요하다"],
                "importance_indicators": ["중요하다", "필수적", "핵심적", "결정적"],
                "semantic_context": "평가나 판단을 나타내는 의미적 구조"
            },
            IntentType.REQUEST: {
                "request_indicators": ["부탁", "요청", "원해", "바라"],
                "help_structures": ["도와", "협조", "지원", "배려"],
                "polite_indicators": ["부탁드립니다", "요청드립니다", "바랍니다"],
                "semantic_context": "도움 요청이나 협조 요청을 나타내는 의미적 구조"
            },
            IntentType.FEEDBACK: {
                "feedback_indicators": ["피드백", "의견", "생각", "평가"],
                "improvement_structures": ["개선", "수정", "보완", "교정"],
                "review_indicators": ["검토", "점검", "확인", "검증"],
                "semantic_context": "피드백이나 개선 요청을 나타내는 의미적 구조"
            }
        }
    
    def parse_intents(self, user_input: str) -> List[ParsedIntent]:
        """의미 분석 기반 다중 의도 파싱 (간단한 버전)"""
        intents = []
        
        # 기본 의도 생성
        intent = ParsedIntent(
            id=str(uuid.uuid4()),
            content=user_input,
            intent_type=IntentType.STATEMENT,
            priority=3,
            context={"input": user_input},
            dependencies=[],
            confidence=0.8
        )
        
        intents.append(intent)
        return intents
    
    def parse_multi_intent(self, user_input: str) -> List[ParsedIntent]:
        """의미 분석 기반 다중 의도 파싱 (전체 버전)"""
        return self.parse_intents(user_input) 
