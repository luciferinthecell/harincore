"""
의도 분석 캐시 시스템 (Intent Analysis Cache)
발화와 응답 간의 의미 분석을 통해 진짜 의도를 파악하고 저장
키워드 스캔 대신 의미 분석 기반으로 의도 추출
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import re


@dataclass
class IntentAnalysis:
    """의도 분석 결과"""
    id: str
    original_utterance: str
    harin_response: str
    extracted_intents: List[Dict[str, Any]]
    semantic_analysis: Dict[str, Any]
    context_understanding: Dict[str, Any]
    drift_detection: Dict[str, Any]
    confidence_score: float
    analysis_timestamp: datetime
    cache_file_path: Optional[str] = None


@dataclass
class SemanticUnit:
    """의미 단위"""
    id: str
    content: str
    semantic_type: str  # 'concept', 'emotion', 'request', 'judgment', 'context'
    meaning: str
    importance: float
    relationships: List[str]


class IntentAnalysisCache:
    """의도 분석 캐시 시스템"""
    
    def __init__(self, cache_dir: str = "intent_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 의미 분석 패턴 (키워드가 아닌 의미적 패턴)
        self.semantic_patterns = {
            "concept": {
                "patterns": [
                    r"([가-힣a-zA-Z0-9]+)에 대해",  # ~에 대해
                    r"([가-힣a-zA-Z0-9]+)의 의미",  # ~의 의미
                    r"([가-힣a-zA-Z0-9]+)가 무엇",  # ~가 무엇
                ],
                "meaning_extractor": self._extract_concept_meaning
            },
            "emotion": {
                "patterns": [
                    r"([가-힣a-zA-Z0-9]+)하다고 생각",  # ~하다고 생각
                    r"([가-힣a-zA-Z0-9]+)한 것 같",  # ~한 것 같
                    r"([가-힣a-zA-Z0-9]+)해서",  # ~해서
                ],
                "meaning_extractor": self._extract_emotion_meaning
            },
            "request": {
                "patterns": [
                    r"([가-힣a-zA-Z0-9]+)해주세요",  # ~해주세요
                    r"([가-힣a-zA-Z0-9]+)하고 싶",  # ~하고 싶
                    r"([가-힣a-zA-Z0-9]+)할 수 있",  # ~할 수 있
                ],
                "meaning_extractor": self._extract_request_meaning
            },
            "judgment": {
                "patterns": [
                    r"([가-힣a-zA-Z0-9]+)가 맞",  # ~가 맞
                    r"([가-힣a-zA-Z0-9]+)가 틀",  # ~가 틀
                    r"([가-힣a-zA-Z0-9]+)가 좋",  # ~가 좋
                ],
                "meaning_extractor": self._extract_judgment_meaning
            },
            "context": {
                "patterns": [
                    r"([가-힣a-zA-Z0-9]+) 상황에서",  # ~상황에서
                    r"([가-힣a-zA-Z0-9]+) 때",  # ~때
                    r"([가-힣a-zA-Z0-9]+) 조건",  # ~조건
                ],
                "meaning_extractor": self._extract_context_meaning
            }
        }
    
    def analyze_intent(self, user_utterance: str, harin_response: str) -> IntentAnalysis:
        """의도 분석 수행"""
        analysis_id = f"intent_{uuid.uuid4().hex[:8]}"
        
        # 1. 의미 단위 추출
        semantic_units = self._extract_semantic_units(user_utterance)
        
        # 2. 응답과의 의미적 연결 분석
        response_connections = self._analyze_response_connections(semantic_units, harin_response)
        
        # 3. 의도 추출
        extracted_intents = self._extract_intents_from_semantics(semantic_units, response_connections)
        
        # 4. 맥락 이해
        context_understanding = self._understand_context(user_utterance, harin_response, semantic_units)
        
        # 5. 드리프트 감지
        drift_detection = self._detect_semantic_drift(user_utterance, harin_response, extracted_intents)
        
        # 6. 신뢰도 계산
        confidence_score = self._calculate_confidence(semantic_units, extracted_intents, drift_detection)
        
        # 7. 분석 결과 생성
        analysis = IntentAnalysis(
            id=analysis_id,
            original_utterance=user_utterance,
            harin_response=harin_response,
            extracted_intents=extracted_intents,
            semantic_analysis={
                "semantic_units": [asdict(unit) for unit in semantic_units],
                "response_connections": response_connections
            },
            context_understanding=context_understanding,
            drift_detection=drift_detection,
            confidence_score=confidence_score,
            analysis_timestamp=datetime.now()
        )
        
        # 8. 캐시 파일 저장
        cache_file_path = self._save_to_cache(analysis)
        analysis.cache_file_path = cache_file_path
        
        return analysis
    
    def _extract_semantic_units(self, text: str) -> List[SemanticUnit]:
        """의미 단위 추출"""
        units = []
        
        for semantic_type, config in self.semantic_patterns.items():
            for pattern in config["patterns"]:
                matches = re.finditer(pattern, text)
                for match in matches:
                    content = match.group(1)
                    meaning = config["meaning_extractor"](content, match.group(0), text)
                    
                    unit = SemanticUnit(
                        id=f"unit_{uuid.uuid4().hex[:8]}",
                        content=content,
                        semantic_type=semantic_type,
                        meaning=meaning,
                        importance=self._calculate_importance(content, semantic_type, text),
                        relationships=[]
                    )
                    units.append(unit)
        
        # 관계 분석
        self._analyze_semantic_relationships(units)
        
        return units
    
    def _extract_concept_meaning(self, content: str, matched_text: str, full_text: str) -> str:
        """개념 의미 추출"""
        # 문맥에서 개념의 의미 파악
        context_before = full_text[:full_text.find(matched_text)]
        context_after = full_text[full_text.find(matched_text) + len(matched_text):]
        
        # 개념 설명 패턴 찾기
        explanation_patterns = [
            rf"{content}는\s+([가-힣a-zA-Z0-9\s]+)",
            rf"{content}이란\s+([가-힣a-zA-Z0-9\s]+)",
            rf"{content}의\s+의미는\s+([가-힣a-zA-Z0-9\s]+)"
        ]
        
        for pattern in explanation_patterns:
            match = re.search(pattern, full_text)
            if match:
                return match.group(1).strip()
        
        return f"개념: {content}"
    
    def _extract_emotion_meaning(self, content: str, matched_text: str, full_text: str) -> str:
        """감정 의미 추출"""
        # 감정의 강도와 방향 파악
        intensity_indicators = ["매우", "정말", "너무", "조금", "약간"]
        emotion_direction = "긍정" if any(word in content for word in ["좋", "기쁘", "감사"]) else "부정"
        
        intensity = "보통"
        for indicator in intensity_indicators:
            if indicator in matched_text:
                intensity = "강함" if indicator in ["매우", "정말", "너무"] else "약함"
                break
        
        return f"감정: {content} ({emotion_direction}, {intensity})"
    
    def _extract_request_meaning(self, content: str, matched_text: str, full_text: str) -> str:
        """요청 의미 추출"""
        # 요청의 성격 파악
        urgency_indicators = ["빨리", "즉시", "당장", "지금"]
        politeness_indicators = ["부탁", "요청", "바라"]
        
        urgency = "보통"
        politeness = "일반"
        
        for indicator in urgency_indicators:
            if indicator in full_text:
                urgency = "긴급"
                break
        
        for indicator in politeness_indicators:
            if indicator in full_text:
                politeness = "정중"
                break
        
        return f"요청: {content} ({urgency}, {politeness})"
    
    def _extract_judgment_meaning(self, content: str, matched_text: str, full_text: str) -> str:
        """판단 의미 추출"""
        # 판단의 근거와 확신도 파악
        confidence_indicators = ["확실히", "분명히", "아마", "어쩌면"]
        reasoning_indicators = ["왜냐하면", "이유는", "때문에"]
        
        confidence = "보통"
        for indicator in confidence_indicators:
            if indicator in full_text:
                confidence = "높음" if indicator in ["확실히", "분명히"] else "낮음"
                break
        
        return f"판단: {content} (확신도: {confidence})"
    
    def _extract_context_meaning(self, content: str, matched_text: str, full_text: str) -> str:
        """맥락 의미 추출"""
        # 맥락의 중요성과 영향도 파악
        importance_indicators = ["중요한", "핵심적인", "결정적인"]
        temporal_indicators = ["지금", "이전", "앞으로", "언젠가"]
        
        importance = "보통"
        temporal = "현재"
        
        for indicator in importance_indicators:
            if indicator in full_text:
                importance = "높음"
                break
        
        for indicator in temporal_indicators:
            if indicator in full_text:
                temporal = indicator
                break
        
        return f"맥락: {content} (중요도: {importance}, 시간: {temporal})"
    
    def _calculate_importance(self, content: str, semantic_type: str, full_text: str) -> float:
        """중요도 계산"""
        base_importance = {
            "concept": 0.8,
            "emotion": 0.7,
            "request": 0.9,
            "judgment": 0.6,
            "context": 0.5
        }
        
        importance = base_importance.get(semantic_type, 0.5)
        
        # 반복 빈도에 따른 조정
        frequency = len(re.findall(content, full_text))
        if frequency > 1:
            importance += 0.1 * (frequency - 1)
        
        # 문장 위치에 따른 조정
        if full_text.startswith(content):
            importance += 0.1
        
        return min(1.0, importance)
    
    def _analyze_semantic_relationships(self, units: List[SemanticUnit]):
        """의미적 관계 분석"""
        for i, unit1 in enumerate(units):
            for j, unit2 in enumerate(units):
                if i != j:
                    relationship = self._find_relationship(unit1, unit2)
                    if relationship:
                        unit1.relationships.append(f"{unit2.id}:{relationship}")
    
    def _find_relationship(self, unit1: SemanticUnit, unit2: SemanticUnit) -> Optional[str]:
        """두 단위 간의 관계 찾기"""
        # 의미적 연결 패턴
        if unit1.semantic_type == "concept" and unit2.semantic_type == "judgment":
            return "concept_judgment"
        elif unit1.semantic_type == "emotion" and unit2.semantic_type == "request":
            return "emotion_request"
        elif unit1.semantic_type == "context" and unit2.semantic_type in ["concept", "request"]:
            return "context_influences"
        
        return None
    
    def _analyze_response_connections(self, semantic_units: List[SemanticUnit], response: str) -> Dict[str, Any]:
        """응답과의 의미적 연결 분석"""
        connections = {
            "addressed_units": [],
            "missed_units": [],
            "response_quality": {}
        }
        
        for unit in semantic_units:
            # 응답에서 해당 의미 단위가 다뤄졌는지 확인
            if self._is_unit_addressed(unit, response):
                connections["addressed_units"].append({
                    "unit_id": unit.id,
                    "content": unit.content,
                    "semantic_type": unit.semantic_type,
                    "response_coverage": self._calculate_response_coverage(unit, response)
                })
            else:
                connections["missed_units"].append({
                    "unit_id": unit.id,
                    "content": unit.content,
                    "semantic_type": unit.semantic_type,
                    "importance": unit.importance
                })
        
        # 응답 품질 평가
        connections["response_quality"] = {
            "coverage_rate": len(connections["addressed_units"]) / len(semantic_units) if semantic_units else 0,
            "missed_important_units": [u for u in connections["missed_units"] if u["importance"] > 0.7]
        }
        
        return connections
    
    def _is_unit_addressed(self, unit: SemanticUnit, response: str) -> bool:
        """단위가 응답에서 다뤄졌는지 확인"""
        # 직접 언급
        if unit.content in response:
            return True
        
        # 의미적 유사성 (간단한 구현)
        unit_words = set(unit.content.split())
        response_words = set(response.split())
        
        if unit_words & response_words:
            return True
        
        return False
    
    def _calculate_response_coverage(self, unit: SemanticUnit, response: str) -> float:
        """응답에서의 단위 커버리지 계산"""
        # 단순한 구현: 단위 내용이 응답에서 얼마나 다뤄졌는지
        if unit.content in response:
            return 1.0
        
        # 부분적 매칭
        unit_words = set(unit.content.split())
        response_words = set(response.split())
        
        if unit_words:
            overlap = len(unit_words & response_words)
            return overlap / len(unit_words)
        
        return 0.0
    
    def _extract_intents_from_semantics(self, semantic_units: List[SemanticUnit], 
                                      response_connections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """의미 단위에서 의도 추출"""
        intents = []
        
        # 각 의미 단위를 의도로 변환
        for unit in semantic_units:
            intent = {
                "id": f"intent_{unit.id}",
                "content": unit.content,
                "semantic_type": unit.semantic_type,
                "meaning": unit.meaning,
                "importance": unit.importance,
                "relationships": unit.relationships,
                "addressed_in_response": any(
                    u["unit_id"] == unit.id for u in response_connections["addressed_units"]
                )
            }
            intents.append(intent)
        
        return intents
    
    def _understand_context(self, utterance: str, response: str, semantic_units: List[SemanticUnit]) -> Dict[str, Any]:
        """맥락 이해"""
        context_units = [u for u in semantic_units if u.semantic_type == "context"]
        
        return {
            "context_units": [asdict(u) for u in context_units],
            "temporal_context": self._extract_temporal_context(utterance),
            "emotional_context": self._extract_emotional_context(utterance),
            "situational_context": self._extract_situational_context(utterance, response)
        }
    
    def _extract_temporal_context(self, utterance: str) -> str:
        """시간적 맥락 추출"""
        temporal_indicators = {
            "과거": ["이전", "지난", "예전", "과거"],
            "현재": ["지금", "현재", "이번", "이제"],
            "미래": ["앞으로", "향후", "다음", "미래"]
        }
        
        for time_period, indicators in temporal_indicators.items():
            if any(indicator in utterance for indicator in indicators):
                return time_period
        
        return "현재"
    
    def _extract_emotional_context(self, utterance: str) -> str:
        """감정적 맥락 추출"""
        positive_emotions = ["좋", "기쁘", "감사", "만족"]
        negative_emotions = ["나쁘", "슬프", "화나", "짜증"]
        neutral_emotions = ["보통", "평범", "일반"]
        
        if any(emotion in utterance for emotion in positive_emotions):
            return "긍정적"
        elif any(emotion in utterance for emotion in negative_emotions):
            return "부정적"
        elif any(emotion in utterance for emotion in neutral_emotions):
            return "중립적"
        
        return "불명확"
    
    def _extract_situational_context(self, utterance: str, response: str) -> str:
        """상황적 맥락 추출"""
        # 간단한 상황 분류
        if "문제" in utterance or "해결" in utterance:
            return "문제해결"
        elif "설명" in utterance or "알려" in utterance:
            return "정보요청"
        elif "의견" in utterance or "생각" in utterance:
            return "의견요청"
        elif "감정" in utterance or "느낌" in utterance:
            return "감정공유"
        
        return "일반대화"
    
    def _detect_semantic_drift(self, utterance: str, response: str, intents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """의미적 드리프트 감지"""
        drift_indicators = {
            "topic_shift": 0.0,
            "emotion_mismatch": 0.0,
            "intent_ignored": 0.0,
            "overall_drift": 0.0
        }
        
        # 주제 전환 감지
        utterance_topics = self._extract_topics(utterance)
        response_topics = self._extract_topics(response)
        
        if utterance_topics and response_topics:
            topic_overlap = len(set(utterance_topics) & set(response_topics))
            drift_indicators["topic_shift"] = 1.0 - (topic_overlap / len(set(utterance_topics) | set(response_topics)))
        
        # 감정 불일치 감지
        utterance_emotion = self._extract_emotional_context(utterance)
        response_emotion = self._extract_emotional_context(response)
        
        if utterance_emotion != response_emotion:
            drift_indicators["emotion_mismatch"] = 1.0
        
        # 의도 무시 감지
        ignored_intents = [intent for intent in intents if not intent["addressed_in_response"]]
        if intents:
            drift_indicators["intent_ignored"] = len(ignored_intents) / len(intents)
        
        # 전체 드리프트 계산
        drift_indicators["overall_drift"] = sum(drift_indicators.values()) / len(drift_indicators)
        
        return drift_indicators
    
    def _extract_topics(self, text: str) -> List[str]:
        """주제 추출 (간단한 구현)"""
        # 명사구 패턴으로 주제 추출
        topic_patterns = [
            r'([가-힣]+)에 대해',
            r'([가-힣]+)의 문제',
            r'([가-힣]+)에 관한',
            r'([가-힣]+)와 관련된'
        ]
        
        topics = []
        for pattern in topic_patterns:
            matches = re.findall(pattern, text)
            topics.extend(matches)
        
        return list(set(topics))
    
    def _calculate_confidence(self, semantic_units: List[SemanticUnit], 
                            intents: List[Dict[str, Any]], 
                            drift_detection: Dict[str, Any]) -> float:
        """신뢰도 계산"""
        # 의미 단위 품질
        unit_confidence = sum(unit.importance for unit in semantic_units) / len(semantic_units) if semantic_units else 0
        
        # 의도 추출 품질
        intent_confidence = len([i for i in intents if i["addressed_in_response"]]) / len(intents) if intents else 0
        
        # 드리프트 영향
        drift_penalty = drift_detection["overall_drift"] * 0.3
        
        confidence = (unit_confidence + intent_confidence) / 2 - drift_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _save_to_cache(self, analysis: IntentAnalysis) -> str:
        """캐시 파일에 저장"""
        cache_file = self.cache_dir / f"intent_analysis_{analysis.id}.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(analysis), f, ensure_ascii=False, indent=2, default=str)
        
        return str(cache_file)
    
    def load_from_cache(self, cache_file_path: str) -> IntentAnalysis:
        """캐시 파일에서 로드"""
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # datetime 복원
        data['analysis_timestamp'] = datetime.fromisoformat(data['analysis_timestamp'])
        
        return IntentAnalysis(**data)
    
    def get_all_cache_files(self) -> List[str]:
        """모든 캐시 파일 경로 반환"""
        cache_files = list(self.cache_dir.glob("intent_analysis_*.json"))
        return [str(f) for f in cache_files]
    
    def analyze_cache_files(self) -> Dict[str, Any]:
        """캐시 파일들 분석"""
        cache_files = self.get_all_cache_files()
        
        if not cache_files:
            return {"message": "캐시 파일이 없습니다."}
        
        analyses = []
        for cache_file in cache_files:
            try:
                analysis = self.load_from_cache(cache_file)
                analyses.append(analysis)
            except Exception as e:
                print(f"캐시 파일 로드 오류: {cache_file}, {e}")
        
        # 통계 분석
        total_analyses = len(analyses)
        avg_confidence = sum(a.confidence_score for a in analyses) / total_analyses if analyses else 0
        
        # 의도 타입 분포
        intent_type_counts = {}
        for analysis in analyses:
            for intent in analysis.extracted_intents:
                intent_type = intent["semantic_type"]
                intent_type_counts[intent_type] = intent_type_counts.get(intent_type, 0) + 1
        
        # 드리프트 통계
        avg_drift = sum(a.drift_detection["overall_drift"] for a in analyses) / total_analyses if analyses else 0
        
        return {
            "total_analyses": total_analyses,
            "average_confidence": avg_confidence,
            "intent_type_distribution": intent_type_counts,
            "average_drift": avg_drift,
            "cache_files": cache_files
        } 
