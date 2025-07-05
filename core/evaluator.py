"""
core.evaluator
~~~~~~~~~~~~~

신뢰도 평가 및 검증 시스템
"""

from typing import Dict, Any, List
from datetime import datetime


class TrustEvaluator:
    """신뢰도 평가기"""
    
    def __init__(self):
        self.evaluation_history = []
        self.trust_threshold = 0.7
    
    def evaluate_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """응답 신뢰도 평가"""
        
        # 기본 평가 메트릭
        evaluation = {
            "coherence": self._evaluate_coherence(response),
            "relevance": self._evaluate_relevance(response, context),
            "completeness": self._evaluate_completeness(response, context),
            "confidence": self._evaluate_confidence(response),
            "timestamp": datetime.now().isoformat()
        }
        
        # 종합 신뢰도 점수 계산
        evaluation["overall_trust"] = (
            evaluation["coherence"] * 0.3 +
            evaluation["relevance"] * 0.3 +
            evaluation["completeness"] * 0.2 +
            evaluation["confidence"] * 0.2
        )
        
        # 평가 결과 저장
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def _evaluate_coherence(self, response: str) -> float:
        """응답의 일관성 평가"""
        
        if not response or len(response.strip()) < 10:
            return 0.1
        
        # 기본 일관성 체크
        coherence_score = 0.7  # 기본 점수
        
        # 문장 구조 체크
        sentences = response.split('.')
        if len(sentences) > 1:
            coherence_score += 0.1
        
        # 논리적 연결어 체크
        logical_connectors = ['하지만', '그러나', '또한', '따라서', '결론적으로', '예를 들어']
        if any(connector in response for connector in logical_connectors):
            coherence_score += 0.1
        
        return min(1.0, coherence_score)
    
    def _evaluate_relevance(self, response: str, context: Dict[str, Any]) -> float:
        """응답의 관련성 평가"""
        
        user_input = context.get("user_input", "")
        if not user_input or not response:
            return 0.5
        
        # 키워드 매칭 체크
        input_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        
        if input_words:
            keyword_match = len(input_words.intersection(response_words)) / len(input_words)
            return min(1.0, keyword_match + 0.3)  # 기본 점수 0.3
        
        return 0.5
    
    def _evaluate_completeness(self, response: str, context: Dict[str, Any]) -> float:
        """응답의 완성도 평가"""
        
        if not response:
            return 0.0
        
        # 길이 기반 완성도
        length_score = min(1.0, len(response) / 200)  # 200자 기준
        
        # 질문 유형별 완성도
        user_input = context.get("user_input", "").lower()
        
        if "어떻게" in user_input or "방법" in user_input:
            # 방법 설명 요청
            if "단계" in response or "먼저" in response or "다음" in response:
                return min(1.0, length_score + 0.2)
        
        elif "왜" in user_input or "이유" in user_input:
            # 이유 설명 요청
            if "때문에" in response or "이유" in response or "원인" in response:
                return min(1.0, length_score + 0.2)
        
        return length_score
    
    def _evaluate_confidence(self, response: str) -> float:
        """응답의 확신도 평가"""
        
        if not response:
            return 0.0
        
        # 확신 표현 체크
        confidence_indicators = [
            "확실히", "분명히", "틀림없이", "당연히", "물론",
            "I'm sure", "definitely", "certainly", "absolutely"
        ]
        
        uncertainty_indicators = [
            "아마", "어쩌면", "아마도", "모르겠지만", "추측",
            "maybe", "perhaps", "possibly", "I think", "I guess"
        ]
        
        confidence_score = 0.5  # 기본 점수
        
        # 확신 표현 카운트
        confidence_count = sum(1 for indicator in confidence_indicators if indicator in response)
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in response)
        
        if confidence_count > uncertainty_count:
            confidence_score += 0.3
        elif uncertainty_count > confidence_count:
            confidence_score -= 0.2
        
        return max(0.0, min(1.0, confidence_score))
    
    def get_trust_summary(self) -> Dict[str, Any]:
        """신뢰도 평가 요약"""
        
        if not self.evaluation_history:
            return {"average_trust": 0.0, "total_evaluations": 0}
        
        recent_evaluations = self.evaluation_history[-10:]  # 최근 10개
        
        avg_trust = sum(eval["overall_trust"] for eval in recent_evaluations) / len(recent_evaluations)
        
        return {
            "average_trust": avg_trust,
            "total_evaluations": len(self.evaluation_history),
            "recent_evaluations": len(recent_evaluations),
            "trust_level": "high" if avg_trust > 0.8 else "medium" if avg_trust > 0.6 else "low"
        }
    
    def should_retry(self, evaluation: Dict[str, Any]) -> bool:
        """재시도 여부 결정"""
        return evaluation.get("overall_trust", 0) < self.trust_threshold
    
    def needs_rerun(self, judgment) -> bool:
        """재실행 필요 여부 확인 (기존 코드 호환성)"""
        # judgment 객체에서 점수 추출
        if hasattr(judgment, 'score') and hasattr(judgment.score, 'overall'):
            score = judgment.score.overall()
        else:
            score = 0.5  # 기본값
        
        return score < self.trust_threshold 
