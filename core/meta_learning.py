"""
고급 메타러닝 시스템
PM Machine의 응답 품질 분석과 지속적 학습 메커니즘을 하린코어에 적용
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)


class ResponseQualityMetrics(BaseModel):
    """응답 품질 메트릭"""
    clarity: float = Field(ge=0.0, le=1.0, description="명확성")
    relevance: float = Field(ge=0.0, le=1.0, description="관련성")
    helpfulness: float = Field(ge=0.0, le=1.0, description="도움됨")
    engagement: float = Field(ge=0.0, le=1.0, description="참여도")
    empathy: float = Field(ge=0.0, le=1.0, description="공감")
    creativity: float = Field(ge=0.0, le=1.0, description="창의성")
    accuracy: float = Field(ge=0.0, le=1.0, description="정확성")
    conciseness: float = Field(ge=0.0, le=1.0, description="간결성")
    
    def get_overall_score(self) -> float:
        """전체 점수 계산"""
        scores = [
            self.clarity, self.relevance, self.helpfulness, 
            self.engagement, self.empathy, self.creativity,
            self.accuracy, self.conciseness
        ]
        return np.mean(scores)
    
    def get_primary_strengths(self) -> List[str]:
        """주요 강점 반환"""
        strengths = []
        if self.clarity > 0.8:
            strengths.append("명확성")
        if self.relevance > 0.8:
            strengths.append("관련성")
        if self.helpfulness > 0.8:
            strengths.append("도움됨")
        if self.engagement > 0.8:
            strengths.append("참여도")
        if self.empathy > 0.8:
            strengths.append("공감")
        if self.creativity > 0.8:
            strengths.append("창의성")
        if self.accuracy > 0.8:
            strengths.append("정확성")
        if self.conciseness > 0.8:
            strengths.append("간결성")
        return strengths
    
    def get_primary_weaknesses(self) -> List[str]:
        """주요 약점 반환"""
        weaknesses = []
        if self.clarity < 0.4:
            weaknesses.append("명확성")
        if self.relevance < 0.4:
            weaknesses.append("관련성")
        if self.helpfulness < 0.4:
            weaknesses.append("도움됨")
        if self.engagement < 0.4:
            weaknesses.append("참여도")
        if self.empathy < 0.4:
            weaknesses.append("공감")
        if self.creativity < 0.4:
            weaknesses.append("창의성")
        if self.accuracy < 0.4:
            weaknesses.append("정확성")
        if self.conciseness < 0.4:
            weaknesses.append("간결성")
        return weaknesses


class LearningInsight(BaseModel):
    """학습 인사이트"""
    insight_type: str = Field(description="인사이트 타입")
    description: str = Field(description="인사이트 설명")
    confidence: float = Field(ge=0.0, le=1.0, description="신뢰도")
    actionable: bool = Field(description="실행 가능한지 여부")
    priority: float = Field(ge=0.0, le=1.0, description="우선순위")


class ResponseAnalysis(BaseModel):
    """응답 분석 결과"""
    user_input: str = Field(description="사용자 입력")
    original_response: str = Field(description="원본 응답")
    emotion_block: str = Field(description="감정 블록")
    thought_block: str = Field(description="사고 블록")
    quality_metrics: ResponseQualityMetrics = Field(description="품질 메트릭")
    insights: List[LearningInsight] = Field(default_factory=list, description="학습 인사이트")
    improvement_suggestions: List[str] = Field(default_factory=list, description="개선 제안")
    overall_rating: float = Field(ge=0.0, le=1.0, description="전체 평가")
    drift: Dict[str, Any] = Field(default_factory=dict, description="드리프트 결과")


class MetaLearningManager:
    """메타러닝 관리자"""
    
    def __init__(self):
        self.llm_client = self._get_llm_client()
        self.learning_history: List[ResponseAnalysis] = []
        self.improvement_patterns: Dict[str, List[float]] = {}
        
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
    
    def analyze_response_quality(
        self,
        user_input: str,
        original_response: str,
        emotion_block: str = "",
        thought_block: str = "",
        learned_facts: str = ""
    ) -> ResponseAnalysis:
        """응답 품질 분석"""
        try:
            # 품질 메트릭 분석
            quality_metrics = self._analyze_quality_metrics(
                user_input, original_response, emotion_block, thought_block
            )
            
            # 학습 인사이트 생성
            insights = self._generate_learning_insights(
                user_input, original_response, quality_metrics, learned_facts
            )
            
            # 개선 제안 생성
            improvement_suggestions = self._generate_improvement_suggestions(
                user_input, original_response, quality_metrics, insights
            )
            
            # 전체 평가 계산
            overall_rating = quality_metrics.get_overall_score()
            
            analysis = ResponseAnalysis(
                user_input=user_input,
                original_response=original_response,
                emotion_block=emotion_block,
                thought_block=thought_block,
                quality_metrics=quality_metrics,
                insights=insights,
                improvement_suggestions=improvement_suggestions,
                overall_rating=overall_rating
            )
            
            # 학습 히스토리에 추가
            self.learning_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"응답 품질 분석 중 오류: {e}")
            return self._create_default_analysis(user_input, original_response)
    
    def _analyze_quality_metrics(
        self,
        user_input: str,
        original_response: str,
        emotion_block: str,
        thought_block: str
    ) -> ResponseQualityMetrics:
        """품질 메트릭 분석"""
        try:
            prompt = f"""
다음 응답의 품질을 분석해주세요:

사용자 입력: {user_input}
AI 응답: {original_response}
감정 블록: {emotion_block}
사고 블록: {thought_block}

다음 8개 항목을 0.0~1.0 사이로 평가해주세요:
1. 명확성 (clarity): 응답이 얼마나 명확하고 이해하기 쉬운가?
2. 관련성 (relevance): 사용자 질문과 얼마나 관련이 있는가?
3. 도움됨 (helpfulness): 사용자에게 얼마나 도움이 되는가?
4. 참여도 (engagement): 사용자의 관심을 얼마나 끄는가?
5. 공감 (empathy): 사용자의 감정에 얼마나 공감하는가?
6. 창의성 (creativity): 얼마나 창의적이고 독창적인가?
7. 정확성 (accuracy): 정보가 얼마나 정확한가?
8. 간결성 (conciseness): 얼마나 간결하고 핵심적인가?

JSON 형태로 응답해주세요:
{{
    "clarity": 0.0~1.0,
    "relevance": 0.0~1.0,
    "helpfulness": 0.0~1.0,
    "engagement": 0.0~1.0,
    "empathy": 0.0~1.0,
    "creativity": 0.0~1.0,
    "accuracy": 0.0~1.0,
    "conciseness": 0.0~1.0
}}
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=300)
            try:
                metrics_data = json.loads(response)
                return ResponseQualityMetrics(**metrics_data)
            except:
                return ResponseQualityMetrics()
                
        except Exception as e:
            logger.error(f"품질 메트릭 분석 중 오류: {e}")
            return ResponseQualityMetrics()
    
    def _generate_learning_insights(
        self,
        user_input: str,
        original_response: str,
        quality_metrics: ResponseQualityMetrics,
        learned_facts: str
    ) -> List[LearningInsight]:
        """학습 인사이트 생성"""
        try:
            prompt = f"""
다음 응답을 분석하여 학습 인사이트를 생성해주세요:

사용자 입력: {user_input}
AI 응답: {original_response}
품질 메트릭: {quality_metrics.model_dump()}
학습된 사실들: {learned_facts}

다음과 같은 인사이트를 생성해주세요:
1. 응답의 강점과 약점
2. 사용자와의 상호작용 패턴
3. 개선이 필요한 영역
4. 효과적인 전략

각 인사이트에 대해 다음 정보를 포함해주세요:
- 인사이트 타입 (강점/약점/패턴/전략)
- 설명
- 신뢰도 (0.0~1.0)
- 실행 가능 여부 (true/false)
- 우선순위 (0.0~1.0)

JSON 형태로 응답해주세요:
{{
    "insights": [
        {{
            "insight_type": "강점/약점/패턴/전략",
            "description": "인사이트 설명",
            "confidence": 0.0~1.0,
            "actionable": true/false,
            "priority": 0.0~1.0
        }}
    ]
}}
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=500)
            try:
                data = json.loads(response)
                insights_data = data.get("insights", [])
                return [LearningInsight(**insight) for insight in insights_data]
            except:
                return []
                
        except Exception as e:
            logger.error(f"학습 인사이트 생성 중 오류: {e}")
            return []
    
    def _generate_improvement_suggestions(
        self,
        user_input: str,
        original_response: str,
        quality_metrics: ResponseQualityMetrics,
        insights: List[LearningInsight]
    ) -> List[str]:
        """개선 제안 생성"""
        try:
            weaknesses = quality_metrics.get_primary_weaknesses()
            high_priority_insights = [insight for insight in insights if insight.priority > 0.7]
            
            prompt = f"""
다음 응답을 개선하기 위한 구체적인 제안을 생성해주세요:

사용자 입력: {user_input}
AI 응답: {original_response}
주요 약점: {', '.join(weaknesses)}
고우선순위 인사이트: {[insight.description for insight in high_priority_insights]}

다음과 같은 구체적이고 실행 가능한 개선 제안을 3-5개 생성해주세요:
1. 응답 구조 개선
2. 감정 블록과 사고 블록의 통합 방법
3. 사용자 참여도 향상 전략
4. 명확성과 간결성 개선

각 제안은 구체적이고 실행 가능해야 합니다.
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=400)
            
            # 제안들을 리스트로 분리
            suggestions = []
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    # 불필요한 기호 제거
                    suggestion = line.lstrip('-•0123456789. ')
                    if suggestion:
                        suggestions.append(suggestion)
            
            return suggestions[:5]  # 최대 5개 제안
            
        except Exception as e:
            logger.error(f"개선 제안 생성 중 오류: {e}")
            return ["응답 품질을 지속적으로 모니터링하고 개선하세요."]
    
    def _create_default_analysis(self, user_input: str, original_response: str) -> ResponseAnalysis:
        """기본 분석 결과 생성"""
        return ResponseAnalysis(
            user_input=user_input,
            original_response=original_response,
            quality_metrics=ResponseQualityMetrics(),
            overall_rating=0.5
        )
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """학습 요약 정보"""
        if not self.learning_history:
            return {"message": "아직 학습 데이터가 없습니다."}
        
        # 전체 평균 품질 점수
        overall_scores = [analysis.overall_rating for analysis in self.learning_history]
        avg_score = np.mean(overall_scores)
        
        # 품질 메트릭 평균
        avg_metrics = {}
        metric_names = ['clarity', 'relevance', 'helpfulness', 'engagement', 
                       'empathy', 'creativity', 'accuracy', 'conciseness']
        
        for metric_name in metric_names:
            values = [getattr(analysis.quality_metrics, metric_name) 
                     for analysis in self.learning_history]
            avg_metrics[metric_name] = np.mean(values)
        
        # 최근 트렌드 (마지막 5개 응답)
        recent_scores = overall_scores[-5:] if len(overall_scores) >= 5 else overall_scores
        trend = "개선" if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else "유지"
        
        # 주요 인사이트
        all_insights = []
        for analysis in self.learning_history:
            all_insights.extend(analysis.insights)
        
        high_priority_insights = [insight for insight in all_insights if insight.priority > 0.7]
        top_insights = sorted(high_priority_insights, key=lambda x: x.priority, reverse=True)[:3]
        
        return {
            "total_responses_analyzed": len(self.learning_history),
            "average_overall_score": avg_score,
            "average_metrics": avg_metrics,
            "recent_trend": trend,
            "top_insights": [insight.description for insight in top_insights],
            "improvement_areas": self._identify_improvement_areas(avg_metrics)
        }
    
    def _identify_improvement_areas(self, avg_metrics: Dict[str, float]) -> List[str]:
        """개선 영역 식별"""
        improvement_areas = []
        for metric_name, score in avg_metrics.items():
            if score < 0.6:
                improvement_areas.append(metric_name)
        return improvement_areas
    
    def generate_improved_response(
        self,
        user_input: str,
        original_response: str,
        analysis: ResponseAnalysis
    ) -> str:
        """개선된 응답 생성"""
        try:
            prompt = f"""
다음 응답을 개선해주세요:

사용자 입력: {user_input}
원본 응답: {original_response}

개선 제안:
{chr(10).join(analysis.improvement_suggestions)}

주요 약점: {', '.join(analysis.quality_metrics.get_primary_weaknesses())}

다음 지침에 따라 응답을 개선해주세요:
1. 원본 응답의 핵심 내용은 유지하되 구조와 표현을 개선
2. 감정 블록과 사고 블록을 자연스럽게 통합
3. 사용자 참여도를 높이는 요소 추가
4. 명확성과 간결성 향상
5. 공감과 창의성 요소 강화

개선된 응답:
"""
            
            improved_response = self.llm_client.generate_text(prompt, max_tokens=500)
            return improved_response.strip()
            
        except Exception as e:
            logger.error(f"개선된 응답 생성 중 오류: {e}")
            return original_response
    
    def track_improvement_patterns(self):
        """개선 패턴 추적"""
        if len(self.learning_history) < 2:
            return
        
        # 최근 10개 응답의 품질 변화 추적
        recent_analyses = self.learning_history[-10:]
        
        for i in range(1, len(recent_analyses)):
            current = recent_analyses[i]
            previous = recent_analyses[i-1]
            
            improvement = current.overall_rating - previous.overall_rating
            
            # 개선 패턴 저장
            pattern_key = f"response_{i}"
            if pattern_key not in self.improvement_patterns:
                self.improvement_patterns[pattern_key] = []
            self.improvement_patterns[pattern_key].append(improvement)
    
    def get_improvement_recommendations(self) -> List[str]:
        """개선 추천사항 생성"""
        try:
            summary = self.get_learning_summary()
            
            prompt = f"""
다음 학습 데이터를 바탕으로 구체적인 개선 추천사항을 생성해주세요:

학습 요약:
- 총 분석된 응답 수: {summary.get('total_responses_analyzed', 0)}
- 평균 전체 점수: {summary.get('average_overall_score', 0):.2f}
- 최근 트렌드: {summary.get('recent_trend', '알 수 없음')}
- 개선 영역: {', '.join(summary.get('improvement_areas', []))}
- 주요 인사이트: {', '.join(summary.get('top_insights', []))}

다음과 같은 구체적이고 실행 가능한 추천사항을 3-5개 생성해주세요:
1. 응답 전략 개선
2. 감정 및 사고 블록 통합 방법
3. 사용자 참여도 향상 전략
4. 품질 모니터링 방법

각 추천사항은 구체적이고 실행 가능해야 합니다.
"""
            
            response = self.llm_client.generate_text(prompt, max_tokens=400)
            
            # 추천사항들을 리스트로 분리
            recommendations = []
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    recommendation = line.lstrip('-•0123456789. ')
                    if recommendation:
                        recommendations.append(recommendation)
            
            return recommendations[:5]
            
        except Exception as e:
            logger.error(f"개선 추천사항 생성 중 오류: {e}")
            return ["지속적인 품질 모니터링과 개선을 수행하세요."]

    # === Drift 감지기 연동 ===
    def analyze_response_quality_with_drift(self, user_input: str, original_response: str, emotion_block: str = "", thought_block: str = "", learned_facts: str = "", keywords: list = None, expected_emotion: str = "", actual_emotion: str = "") -> ResponseAnalysis:
        """응답 품질 분석 + drift 감지 자동 호출"""
        analysis = self.analyze_response_quality(user_input, original_response, emotion_block, thought_block, learned_facts)
        try:
            from prompt.drift_detector import detect_drift
            drift_result = detect_drift(user_input, original_response, keywords or [], expected_emotion, actual_emotion)
            analysis.drift = drift_result
        except Exception as e:
            analysis.drift = {"error": str(e)}
        return analysis


# 사용 예시
def create_meta_learning_example():
    """메타러닝 예시"""
    manager = MetaLearningManager()
    
    user_input = "새로운 프로젝트를 시작하려고 하는데, 불안해요."
    original_response = "프로젝트를 시작하는 것은 항상 도전적입니다. 계획을 세우고 단계별로 진행하면 됩니다."
    emotion_block = "사용자가 불안해하는 것을 느낌. 안정감을 제공하고 싶음."
    thought_block = "프로젝트 시작 시 불안은 자연스러운 반응. 구체적인 계획과 단계별 접근이 도움될 것."
    learned_facts = "사용자는 새로운 도전에 불안해함. 구체적인 계획을 선호함."
    
    analysis = manager.analyze_response_quality(
        user_input=user_input,
        original_response=original_response,
        emotion_block=emotion_block,
        thought_block=thought_block,
        learned_facts=learned_facts
    )
    
    print("=== 메타러닝 분석 결과 ===")
    print(f"전체 평가: {analysis.overall_rating:.2f}")
    print(f"주요 강점: {', '.join(analysis.quality_metrics.get_primary_strengths())}")
    print(f"주요 약점: {', '.join(analysis.quality_metrics.get_primary_weaknesses())}")
    print(f"\n개선 제안:")
    for i, suggestion in enumerate(analysis.improvement_suggestions, 1):
        print(f"{i}. {suggestion}")
    
    # 개선된 응답 생성
    improved_response = manager.generate_improved_response(
        user_input, original_response, analysis
    )
    
    print(f"\n개선된 응답:\n{improved_response}")
    
    return analysis 

# === V8 Scar/Meta Learning 구조 통합 ===
from typing import List as TypingList, Dict as TypingDict
from datetime import datetime as dt_datetime

class ScarTriggerEngine:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.scar_keywords = {
            "scar_3": ["illogical", "inconsistent", "missing logic"],
            "scar_4": ["avoid", "refuse", "skip"],
            "scar_5": ["comforting", "emotional appeal", "overly soft"],
            "scar_9": ["misremember", "distorted", "false recall"],
            "scar_11": ["reactive", "impulsive", "non-reasoned"],
            "scar_13": ["overconfident", "assumed", "unfounded"]
        }

    def detect(self, response: str, meta: TypingDict) -> TypingList[str]:
        detected = []
        lower_resp = response.lower()
        for scar, keywords in self.scar_keywords.items():
            score = sum(1 for k in keywords if k in lower_resp)
            if score / len(keywords) >= self.threshold:
                detected.append(scar)
        return detected

    def should_trigger_meta_loop(self, scars: TypingList[str], confidence: float) -> bool:
        high_scar = {"scar_9", "scar_11", "scar_13"}
        return bool(set(scars) & high_scar) or confidence < 0.5

class MetaLearningLoop:
    def __init__(self, scars: TypingList[str], previous_response: str, context: TypingDict):
        self.scars = scars
        self.previous = previous_response
        self.context = context

    def reflect(self) -> TypingDict:
        scar_note = ", ".join(self.scars)
        new_thought = f"Upon reflection, I noticed issues: {scar_note}. Revising the response..."
        corrected = self.revise_response()
        return {
            "node_id": f"meta_{int(dt_datetime.utcnow().timestamp())}",
            "text": corrected,
            "emotion_vector": self.context.get("emotion_vector", []),
            "context_snapshot": self.context,
            "agent_roles": ["SelfReflector"],
            "universe_id": self.context.get("universe_id", "U0"),
            "created_at": dt_datetime.utcnow(),
            "confidence": 0.65,
            "tags": ["meta_loop", "scar_correction"] + self.scars,
            "scar_flagged": False
        }

    def revise_response(self) -> str:
        # 실제 사용 시 LLM 기반 reflection prompting 필요
        return f"Revised response (v2): {self.previous} [with more clarity and responsibility]" 