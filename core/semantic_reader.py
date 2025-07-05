"""
의미 분석 기반 정교한 읽기 시스템 (Semantic Reader)
캐시 파일을 읽고 사고 루프를 거쳐 판단하는 시스템
키워드 스캔 대신 의미 분석 기반으로 정교하게 읽기
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from intent_analysis_cache import IntentAnalysis, IntentAnalysisCache


@dataclass
class ReadingContext:
    """읽기 맥락"""
    file_path: str
    reading_depth: str  # 'surface', 'semantic', 'deep'
    focus_areas: List[str]
    analysis_goals: List[str]
    reading_timestamp: datetime


@dataclass
class SemanticUnderstanding:
    """의미적 이해"""
    content_summary: str
    key_concepts: List[Dict[str, Any]]
    emotional_tone: str
    underlying_intents: List[str]
    context_relationships: Dict[str, Any]
    confidence_level: float


@dataclass
class ThoughtProcess:
    """사고 과정"""
    step_id: str
    thought_type: str  # 'analysis', 'synthesis', 'evaluation', 'inference'
    content: str
    reasoning: str
    confidence: float
    dependencies: List[str]


@dataclass
class ReadingResult:
    """읽기 결과"""
    file_path: str
    semantic_understanding: SemanticUnderstanding
    thought_processes: List[ThoughtProcess]
    judgments: List[Dict[str, Any]]
    insights: List[str]
    reading_quality: Dict[str, float]
    timestamp: datetime


class SemanticReader:
    """의미 분석 기반 정교한 읽기 시스템"""
    
    def __init__(self, intent_cache: IntentAnalysisCache):
        self.intent_cache = intent_cache
        
        # 읽기 전략
        self.reading_strategies = {
            "surface": self._surface_reading,
            "semantic": self._semantic_reading,
            "deep": self._deep_reading
        }
        
        # 사고 루프 단계
        self.thought_loop_stages = [
            "initial_comprehension",
            "semantic_analysis", 
            "context_integration",
            "intent_extraction",
            "relationship_mapping",
            "synthesis",
            "evaluation",
            "insight_generation"
        ]
    
    def read_cache_file(self, cache_file_path: str, reading_context: ReadingContext) -> ReadingResult:
        """캐시 파일 읽기"""
        # 1. 파일 로드
        intent_analysis = self.intent_cache.load_from_cache(cache_file_path)
        
        # 2. 읽기 전략 적용
        reading_strategy = self.reading_strategies.get(reading_context.reading_depth, self._semantic_reading)
        initial_understanding = reading_strategy(intent_analysis, reading_context)
        
        # 3. 사고 루프 실행
        thought_processes = self._execute_thought_loop(intent_analysis, initial_understanding, reading_context)
        
        # 4. 판단 생성
        judgments = self._generate_judgments(intent_analysis, thought_processes, reading_context)
        
        # 5. 통찰 생성
        insights = self._generate_insights(intent_analysis, thought_processes, judgments)
        
        # 6. 읽기 품질 평가
        reading_quality = self._evaluate_reading_quality(intent_analysis, thought_processes, judgments)
        
        return ReadingResult(
            file_path=cache_file_path,
            semantic_understanding=initial_understanding,
            thought_processes=thought_processes,
            judgments=judgments,
            insights=insights,
            reading_quality=reading_quality,
            timestamp=datetime.now()
        )
    
    def _surface_reading(self, intent_analysis: IntentAnalysis, context: ReadingContext) -> SemanticUnderstanding:
        """표면적 읽기"""
        # 기본적인 내용 파악
        content_summary = f"사용자: {intent_analysis.original_utterance[:100]}... | 하린: {intent_analysis.harin_response[:100]}..."
        
        key_concepts = []
        for intent in intent_analysis.extracted_intents[:3]:  # 상위 3개만
            key_concepts.append({
                "content": intent["content"],
                "type": intent["semantic_type"],
                "importance": intent["importance"]
            })
        
        return SemanticUnderstanding(
            content_summary=content_summary,
            key_concepts=key_concepts,
            emotional_tone="중립적",
            underlying_intents=["기본_의도_파악"],
            context_relationships={},
            confidence_level=0.6
        )
    
    def _semantic_reading(self, intent_analysis: IntentAnalysis, context: ReadingContext) -> SemanticUnderstanding:
        """의미적 읽기"""
        # 의미 단위 분석
        semantic_units = intent_analysis.semantic_analysis["semantic_units"]
        
        # 핵심 개념 추출
        key_concepts = []
        for unit in semantic_units:
            if unit["importance"] > 0.7:
                key_concepts.append({
                    "content": unit["content"],
                    "type": unit["semantic_type"],
                    "meaning": unit["meaning"],
                    "importance": unit["importance"]
                })
        
        # 감정적 톤 분석
        emotional_units = [u for u in semantic_units if u["semantic_type"] == "emotion"]
        emotional_tone = self._analyze_emotional_tone(emotional_units)
        
        # 기본 의도 추출
        underlying_intents = []
        for intent in intent_analysis.extracted_intents:
            if intent["importance"] > 0.5:
                underlying_intents.append(f"{intent['semantic_type']}: {intent['content']}")
        
        # 맥락 관계 분석
        context_relationships = self._analyze_context_relationships(intent_analysis)
        
        return SemanticUnderstanding(
            content_summary=self._generate_semantic_summary(intent_analysis),
            key_concepts=key_concepts,
            emotional_tone=emotional_tone,
            underlying_intents=underlying_intents,
            context_relationships=context_relationships,
            confidence_level=0.8
        )
    
    def _deep_reading(self, intent_analysis: IntentAnalysis, context: ReadingContext) -> SemanticUnderstanding:
        """깊은 읽기"""
        # 의미적 읽기 기반
        base_understanding = self._semantic_reading(intent_analysis, context)
        
        # 추가적인 깊은 분석
        deep_concepts = self._extract_deep_concepts(intent_analysis)
        base_understanding.key_concepts.extend(deep_concepts)
        
        # 숨겨진 의도 탐색
        hidden_intents = self._discover_hidden_intents(intent_analysis)
        base_understanding.underlying_intents.extend(hidden_intents)
        
        # 복합적 맥락 관계
        complex_relationships = self._analyze_complex_relationships(intent_analysis)
        base_understanding.context_relationships.update(complex_relationships)
        
        base_understanding.confidence_level = 0.9
        
        return base_understanding
    
    def _execute_thought_loop(self, intent_analysis: IntentAnalysis, 
                            understanding: SemanticUnderstanding,
                            context: ReadingContext) -> List[ThoughtProcess]:
        """사고 루프 실행"""
        thought_processes = []
        
        for i, stage in enumerate(self.thought_loop_stages):
            thought = self._execute_thought_stage(stage, intent_analysis, understanding, context, thought_processes)
            if thought:
                thought_processes.append(thought)
        
        return thought_processes
    
    def _execute_thought_stage(self, stage: str, intent_analysis: IntentAnalysis,
                             understanding: SemanticUnderstanding, context: ReadingContext,
                             previous_thoughts: List[ThoughtProcess]) -> Optional[ThoughtProcess]:
        """개별 사고 단계 실행"""
        
        if stage == "initial_comprehension":
            return ThoughtProcess(
                step_id=f"step_{len(previous_thoughts)}",
                thought_type="analysis",
                content="초기 이해: 사용자 발화와 하린 응답의 기본 구조 파악",
                reasoning=f"발화: {intent_analysis.original_utterance[:50]}... | 응답: {intent_analysis.harin_response[:50]}...",
                confidence=0.7,
                dependencies=[]
            )
        
        elif stage == "semantic_analysis":
            semantic_units = intent_analysis.semantic_analysis["semantic_units"]
            unit_count = len(semantic_units)
            high_importance_count = len([u for u in semantic_units if u["importance"] > 0.7])
            
            return ThoughtProcess(
                step_id=f"step_{len(previous_thoughts)}",
                thought_type="analysis",
                content=f"의미 분석: {unit_count}개 의미 단위 중 {high_importance_count}개 고중요도",
                reasoning=f"의미 단위 분포: {self._get_semantic_distribution(semantic_units)}",
                confidence=0.8,
                dependencies=[previous_thoughts[-1].step_id] if previous_thoughts else []
            )
        
        elif stage == "context_integration":
            context_info = intent_analysis.context_understanding
            temporal = context_info.get("temporal_context", "현재")
            emotional = context_info.get("emotional_context", "중립적")
            situational = context_info.get("situational_context", "일반대화")
            
            return ThoughtProcess(
                step_id=f"step_{len(previous_thoughts)}",
                thought_type="synthesis",
                content=f"맥락 통합: {temporal}적, {emotional}적, {situational} 상황",
                reasoning=f"시간: {temporal}, 감정: {emotional}, 상황: {situational}",
                confidence=0.75,
                dependencies=[previous_thoughts[-1].step_id] if previous_thoughts else []
            )
        
        elif stage == "intent_extraction":
            intents = intent_analysis.extracted_intents
            addressed_count = len([i for i in intents if i["addressed_in_response"]])
            missed_count = len(intents) - addressed_count
            
            return ThoughtProcess(
                step_id=f"step_{len(previous_thoughts)}",
                thought_type="analysis",
                content=f"의도 추출: {len(intents)}개 의도 중 {addressed_count}개 처리됨, {missed_count}개 누락",
                reasoning=f"처리율: {addressed_count/len(intents)*100:.1f}% | 누락된 의도: {self._get_missed_intents(intents)}",
                confidence=0.85,
                dependencies=[previous_thoughts[-1].step_id] if previous_thoughts else []
            )
        
        elif stage == "relationship_mapping":
            relationships = self._map_semantic_relationships(intent_analysis)
            
            return ThoughtProcess(
                step_id=f"step_{len(previous_thoughts)}",
                thought_type="synthesis",
                content=f"관계 매핑: {len(relationships)}개 의미적 관계 발견",
                reasoning=f"관계 유형: {list(relationships.keys())}",
                confidence=0.8,
                dependencies=[previous_thoughts[-1].step_id] if previous_thoughts else []
            )
        
        elif stage == "synthesis":
            synthesis = self._synthesize_understanding(intent_analysis, understanding, previous_thoughts)
            
            return ThoughtProcess(
                step_id=f"step_{len(previous_thoughts)}",
                thought_type="synthesis",
                content="통합적 이해: 모든 분석 요소를 종합한 전체적 이해",
                reasoning=synthesis,
                confidence=0.9,
                dependencies=[t.step_id for t in previous_thoughts[-3:]] if len(previous_thoughts) >= 3 else [t.step_id for t in previous_thoughts]
            )
        
        elif stage == "evaluation":
            evaluation = self._evaluate_quality(intent_analysis, understanding, previous_thoughts)
            
            return ThoughtProcess(
                step_id=f"step_{len(previous_thoughts)}",
                thought_type="evaluation",
                content="품질 평가: 분석 결과의 신뢰성과 완성도 평가",
                reasoning=evaluation,
                confidence=0.85,
                dependencies=[previous_thoughts[-1].step_id] if previous_thoughts else []
            )
        
        elif stage == "insight_generation":
            insights = self._generate_thought_insights(intent_analysis, understanding, previous_thoughts)
            
            return ThoughtProcess(
                step_id=f"step_{len(previous_thoughts)}",
                thought_type="inference",
                content=f"통찰 생성: {len(insights)}개 핵심 통찰 도출",
                reasoning=" | ".join(insights[:3]),  # 상위 3개만
                confidence=0.8,
                dependencies=[t.step_id for t in previous_thoughts[-2:]] if len(previous_thoughts) >= 2 else [t.step_id for t in previous_thoughts]
            )
        
        return None
    
    def _generate_judgments(self, intent_analysis: IntentAnalysis, 
                          thought_processes: List[ThoughtProcess],
                          context: ReadingContext) -> List[Dict[str, Any]]:
        """판단 생성"""
        judgments = []
        
        # 1. 의도 처리 품질 판단
        intents = intent_analysis.extracted_intents
        addressed_rate = len([i for i in intents if i["addressed_in_response"]]) / len(intents) if intents else 0
        
        judgments.append({
            "type": "intent_processing_quality",
            "content": f"의도 처리 품질: {addressed_rate:.1%}",
            "score": addressed_rate,
            "reasoning": f"{len(intents)}개 의도 중 {len([i for i in intents if i['addressed_in_response']])}개 처리됨"
        })
        
        # 2. 의미적 일관성 판단
        drift_score = 1.0 - intent_analysis.drift_detection["overall_drift"]
        judgments.append({
            "type": "semantic_consistency",
            "content": f"의미적 일관성: {drift_score:.1%}",
            "score": drift_score,
            "reasoning": f"드리프트 점수: {intent_analysis.drift_detection['overall_drift']:.3f}"
        })
        
        # 3. 맥락 이해도 판단
        context_understanding_score = self._evaluate_context_understanding(intent_analysis)
        judgments.append({
            "type": "context_understanding",
            "content": f"맥락 이해도: {context_understanding_score:.1%}",
            "score": context_understanding_score,
            "reasoning": "시간적, 감정적, 상황적 맥락 통합 분석"
        })
        
        # 4. 사고 과정 품질 판단
        thought_quality_score = self._evaluate_thought_quality(thought_processes)
        judgments.append({
            "type": "thought_process_quality",
            "content": f"사고 과정 품질: {thought_quality_score:.1%}",
            "score": thought_quality_score,
            "reasoning": f"{len(thought_processes)}개 사고 단계의 논리적 연결성"
        })
        
        return judgments
    
    def _generate_insights(self, intent_analysis: IntentAnalysis,
                         thought_processes: List[ThoughtProcess],
                         judgments: List[Dict[str, Any]]) -> List[str]:
        """통찰 생성"""
        insights = []
        
        # 1. 의도 처리 패턴 통찰
        intents = intent_analysis.extracted_intents
        missed_intents = [i for i in intents if not i["addressed_in_response"]]
        
        if missed_intents:
            high_importance_missed = [i for i in missed_intents if i["importance"] > 0.7]
            if high_importance_missed:
                insights.append(f"고중요도 의도 {len(high_importance_missed)}개가 누락됨 - 우선순위 처리 필요")
        
        # 2. 의미적 드리프트 통찰
        drift_indicators = intent_analysis.drift_detection
        if drift_indicators["topic_shift"] > 0.5:
            insights.append("주제 전환이 발생하여 사용자 의도와 응답 간 괴리 발생")
        
        if drift_indicators["emotion_mismatch"] > 0:
            insights.append("감정적 불일치로 인한 맥락 왜곡 가능성")
        
        # 3. 사고 과정 통찰
        thought_types = [t.thought_type for t in thought_processes]
        if "synthesis" not in thought_types:
            insights.append("통합적 사고 단계 부족 - 개별 분석은 있으나 종합적 이해 부족")
        
        # 4. 맥락 이해 통찰
        context_info = intent_analysis.context_understanding
        if context_info.get("temporal_context") == "과거" and "기억" not in intent_analysis.original_utterance:
            insights.append("과거 맥락이 언급되었으나 기억 활용도가 낮음")
        
        return insights
    
    def _evaluate_reading_quality(self, intent_analysis: IntentAnalysis,
                                thought_processes: List[ThoughtProcess],
                                judgments: List[Dict[str, Any]]) -> Dict[str, float]:
        """읽기 품질 평가"""
        quality_metrics = {}
        
        # 1. 이해 깊이
        understanding_depth = len(thought_processes) / len(self.thought_loop_stages)
        quality_metrics["understanding_depth"] = understanding_depth
        
        # 2. 분석 정확도
        analysis_accuracy = intent_analysis.confidence_score
        quality_metrics["analysis_accuracy"] = analysis_accuracy
        
        # 3. 판단 일관성
        judgment_scores = [j["score"] for j in judgments]
        judgment_consistency = sum(judgment_scores) / len(judgment_scores) if judgment_scores else 0
        quality_metrics["judgment_consistency"] = judgment_consistency
        
        # 4. 통찰 품질
        insights = self._generate_insights(intent_analysis, thought_processes, judgments)
        insight_quality = min(1.0, len(insights) / 5.0)  # 최대 5개 통찰 기준
        quality_metrics["insight_quality"] = insight_quality
        
        # 5. 전체 품질
        overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
        quality_metrics["overall_quality"] = overall_quality
        
        return quality_metrics
    
    # 헬퍼 메서드들
    def _analyze_emotional_tone(self, emotional_units: List[Dict[str, Any]]) -> str:
        """감정적 톤 분석"""
        if not emotional_units:
            return "중립적"
        
        positive_count = 0
        negative_count = 0
        
        for unit in emotional_units:
            meaning = unit.get("meaning", "")
            if "긍정" in meaning:
                positive_count += 1
            elif "부정" in meaning:
                negative_count += 1
        
        if positive_count > negative_count:
            return "긍정적"
        elif negative_count > positive_count:
            return "부정적"
        else:
            return "중립적"
    
    def _analyze_context_relationships(self, intent_analysis: IntentAnalysis) -> Dict[str, Any]:
        """맥락 관계 분석"""
        context_info = intent_analysis.context_understanding
        
        return {
            "temporal_emotional": f"{context_info.get('temporal_context', '현재')}-{context_info.get('emotional_context', '중립적')}",
            "situational_focus": context_info.get("situational_context", "일반대화"),
            "context_units": len(context_info.get("context_units", []))
        }
    
    def _generate_semantic_summary(self, intent_analysis: IntentAnalysis) -> str:
        """의미적 요약 생성"""
        intents = intent_analysis.extracted_intents
        high_importance_intents = [i for i in intents if i["importance"] > 0.7]
        
        summary_parts = []
        summary_parts.append(f"사용자 발화: {intent_analysis.original_utterance[:50]}...")
        summary_parts.append(f"하린 응답: {intent_analysis.harin_response[:50]}...")
        summary_parts.append(f"추출된 의도: {len(intents)}개 (고중요도: {len(high_importance_intents)}개)")
        
        return " | ".join(summary_parts)
    
    def _extract_deep_concepts(self, intent_analysis: IntentAnalysis) -> List[Dict[str, Any]]:
        """깊은 개념 추출"""
        deep_concepts = []
        
        # 응답에서 추가 개념 추출
        response_words = intent_analysis.harin_response.split()
        for word in response_words:
            if len(word) > 2 and word not in [i["content"] for i in intent_analysis.extracted_intents]:
                deep_concepts.append({
                    "content": word,
                    "type": "derived_concept",
                    "meaning": f"응답에서 파생된 개념: {word}",
                    "importance": 0.6
                })
        
        return deep_concepts[:5]  # 최대 5개
    
    def _discover_hidden_intents(self, intent_analysis: IntentAnalysis) -> List[str]:
        """숨겨진 의도 탐색"""
        hidden_intents = []
        
        # 드리프트에서 숨겨진 의도 추론
        drift_info = intent_analysis.drift_detection
        if drift_info["topic_shift"] > 0.3:
            hidden_intents.append("주제_전환_의도")
        
        if drift_info["emotion_mismatch"] > 0:
            hidden_intents.append("감정_조정_의도")
        
        return hidden_intents
    
    def _analyze_complex_relationships(self, intent_analysis: IntentAnalysis) -> Dict[str, Any]:
        """복합적 관계 분석"""
        return {
            "intent_response_alignment": self._calculate_alignment(intent_analysis),
            "semantic_coherence": intent_analysis.confidence_score,
            "contextual_fit": 1.0 - intent_analysis.drift_detection["overall_drift"]
        }
    
    def _calculate_alignment(self, intent_analysis: IntentAnalysis) -> float:
        """의도-응답 정렬도 계산"""
        intents = intent_analysis.extracted_intents
        if not intents:
            return 0.0
        
        addressed_intents = [i for i in intents if i["addressed_in_response"]]
        return len(addressed_intents) / len(intents)
    
    def _get_semantic_distribution(self, semantic_units: List[Dict[str, Any]]) -> str:
        """의미 분포 문자열 생성"""
        type_counts = {}
        for unit in semantic_units:
            unit_type = unit["semantic_type"]
            type_counts[unit_type] = type_counts.get(unit_type, 0) + 1
        
        return ", ".join([f"{k}:{v}" for k, v in type_counts.items()])
    
    def _get_missed_intents(self, intents: List[Dict[str, Any]]) -> str:
        """누락된 의도 문자열 생성"""
        missed = [i["content"] for i in intents if not i["addressed_in_response"]]
        return ", ".join(missed[:3]) + ("..." if len(missed) > 3 else "")
    
    def _map_semantic_relationships(self, intent_analysis: IntentAnalysis) -> Dict[str, Any]:
        """의미적 관계 매핑"""
        relationships = {}
        
        # 의도 간 관계
        intents = intent_analysis.extracted_intents
        for i, intent1 in enumerate(intents):
            for j, intent2 in enumerate(intents[i+1:], i+1):
                if intent1["semantic_type"] != intent2["semantic_type"]:
                    rel_type = f"{intent1['semantic_type']}_{intent2['semantic_type']}"
                    relationships[rel_type] = relationships.get(rel_type, 0) + 1
        
        return relationships
    
    def _synthesize_understanding(self, intent_analysis: IntentAnalysis,
                                understanding: SemanticUnderstanding,
                                thought_processes: List[ThoughtProcess]) -> str:
        """이해 종합"""
        synthesis_parts = []
        
        # 핵심 의도
        key_intents = [i for i in intent_analysis.extracted_intents if i["importance"] > 0.7]
        synthesis_parts.append(f"핵심 의도: {len(key_intents)}개")
        
        # 처리 상태
        addressed_count = len([i for i in intent_analysis.extracted_intents if i["addressed_in_response"]])
        synthesis_parts.append(f"처리 상태: {addressed_count}/{len(intent_analysis.extracted_intents)}")
        
        # 맥락 이해
        context_info = intent_analysis.context_understanding
        synthesis_parts.append(f"맥락: {context_info.get('temporal_context', '현재')}-{context_info.get('emotional_context', '중립적')}")
        
        return " | ".join(synthesis_parts)
    
    def _evaluate_quality(self, intent_analysis: IntentAnalysis,
                         understanding: SemanticUnderstanding,
                         thought_processes: List[ThoughtProcess]) -> str:
        """품질 평가"""
        quality_indicators = []
        
        # 신뢰도
        quality_indicators.append(f"신뢰도: {intent_analysis.confidence_score:.1%}")
        
        # 드리프트
        drift_score = 1.0 - intent_analysis.drift_detection["overall_drift"]
        quality_indicators.append(f"일관성: {drift_score:.1%}")
        
        # 사고 과정 완성도
        completion_rate = len(thought_processes) / len(self.thought_loop_stages)
        quality_indicators.append(f"완성도: {completion_rate:.1%}")
        
        return " | ".join(quality_indicators)
    
    def _evaluate_context_understanding(self, intent_analysis: IntentAnalysis) -> float:
        """맥락 이해도 평가"""
        context_info = intent_analysis.context_understanding
        
        # 각 맥락 요소의 명확성 평가
        temporal_clarity = 1.0 if context_info.get("temporal_context") != "불명확" else 0.5
        emotional_clarity = 1.0 if context_info.get("emotional_context") != "불명확" else 0.5
        situational_clarity = 1.0 if context_info.get("situational_context") != "일반대화" else 0.7
        
        return (temporal_clarity + emotional_clarity + situational_clarity) / 3
    
    def _evaluate_thought_quality(self, thought_processes: List[ThoughtProcess]) -> float:
        """사고 과정 품질 평가"""
        if not thought_processes:
            return 0.0
        
        # 평균 신뢰도
        avg_confidence = sum(t.confidence for t in thought_processes) / len(thought_processes)
        
        # 의존성 연결성
        dependency_score = 0.0
        for thought in thought_processes:
            if thought.dependencies:
                dependency_score += 1.0
        
        dependency_score = dependency_score / len(thought_processes)
        
        return (avg_confidence + dependency_score) / 2
    
    def _generate_thought_insights(self, intent_analysis: IntentAnalysis,
                                 understanding: SemanticUnderstanding,
                                 thought_processes: List[ThoughtProcess]) -> List[str]:
        """사고 통찰 생성"""
        insights = []
        
        # 사고 과정 패턴 분석
        thought_types = [t.thought_type for t in thought_processes]
        if "analysis" in thought_types and "synthesis" in thought_types:
            insights.append("분석-종합 패턴이 적절히 적용됨")
        
        # 신뢰도 패턴
        confidences = [t.confidence for t in thought_processes]
        if confidences and max(confidences) - min(confidences) < 0.3:
            insights.append("사고 과정의 신뢰도가 일관적임")
        
        return insights 
