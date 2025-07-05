"""
harin.memory.text_importer
~~~~~~~~~~~~~~~~~~~~~~~~~

텍스트 파일 업로드 및 기억 저장 시스템
- 대화 주제와 맥락에 따른 자동 태깅
- 관련 기억 자동 추출
- 세분화된 개념 태그
- 사고루프 연동
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from memory.palantirgraph import PalantirGraph, ThoughtNode, Relationship


@dataclass
class TextSegment:
    """텍스트 세그먼트 (대화 단위)"""
    id: str
    content: str
    speaker: str  # "user" or "assistant"
    timestamp: str
    topic: str = ""
    context_tags: List[str] = field(default_factory=list)
    concept_tags: List[str] = field(default_factory=list)
    emotion: str = "neutral"
    importance: float = 0.5


@dataclass
class ConversationMemory:
    """대화 기억 구조체"""
    session_id: str
    title: str
    segments: List[TextSegment]
    overall_topics: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    meta: Dict[str, Any] = field(default_factory=dict)


class TextImporter:
    """텍스트 파일 임포터 및 기억 저장기"""
    
    def __init__(self, memory_graph: PalantirGraph):
        self.memory = memory_graph
        self.topic_extractor = TopicExtractor()
        self.concept_extractor = ConceptExtractor()
        self.emotion_analyzer = EmotionAnalyzer()
        
    def import_text_file(self, file_path: str | Path, session_title: str = "") -> ConversationMemory:
        """텍스트 파일을 읽어서 기억으로 저장"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
        # 파일 내용 읽기
        content = file_path.read_text(encoding='utf-8')
        
        # 대화 세그먼트로 분할
        segments = self._parse_conversation(content)
        
        # 세그먼트별 분석 및 태깅
        for segment in segments:
            self._analyze_segment(segment)
            
        # 전체 주제 추출
        overall_topics = self._extract_overall_topics(segments)
        
        # 기억 저장
        conversation = ConversationMemory(
            session_id=str(uuid.uuid4()),
            title=session_title or file_path.stem,
            segments=segments,
            overall_topics=overall_topics
        )
        
        self._store_conversation(conversation)
        
        return conversation
    
    def import_api_data(self, conversation_data: Dict[str, Any]) -> ConversationMemory:
        """API로 전송된 대화 데이터를 기억으로 저장"""
        segments = []
        
        for msg in conversation_data.get('messages', []):
            segment = TextSegment(
                id=str(uuid.uuid4()),
                content=msg.get('content', ''),
                speaker=msg.get('role', 'user'),
                timestamp=msg.get('timestamp', datetime.now().isoformat())
            )
            self._analyze_segment(segment)
            segments.append(segment)
            
        overall_topics = self._extract_overall_topics(segments)
        
        conversation = ConversationMemory(
            session_id=conversation_data.get('session_id', str(uuid.uuid4())),
            title=conversation_data.get('title', 'API 대화'),
            segments=segments,
            overall_topics=overall_topics,
            meta=conversation_data.get('meta', {})
        )
        
        self._store_conversation(conversation)
        
        return conversation
    
    def _parse_conversation(self, content: str) -> List[TextSegment]:
        """텍스트를 대화 세그먼트로 분할"""
        segments = []
        lines = content.split('\n')
        current_segment = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 사용자/어시스턴트 구분 패턴
            user_match = re.match(r'^(사용자|User|U):\s*(.+)', line)
            assistant_match = re.match(r'^(어시스턴트|Assistant|A|GPT|ChatGPT):\s*(.+)', line)
            
            if user_match:
                if current_segment:
                    segments.append(current_segment)
                current_segment = TextSegment(
                    id=str(uuid.uuid4()),
                    content=user_match.group(2),
                    speaker="user",
                    timestamp=datetime.now().isoformat()
                )
            elif assistant_match:
                if current_segment:
                    segments.append(current_segment)
                current_segment = TextSegment(
                    id=str(uuid.uuid4()),
                    content=assistant_match.group(2),
                    speaker="assistant",
                    timestamp=datetime.now().isoformat()
                )
            elif current_segment:
                # 연속된 텍스트
                current_segment.content += " " + line
                
        if current_segment:
            segments.append(current_segment)
            
        return segments
    
    def _analyze_segment(self, segment: TextSegment):
        """세그먼트 분석 및 태깅"""
        # 주제 추출
        segment.topic = self.topic_extractor.extract(segment.content)
        
        # 맥락 태그 추출
        segment.context_tags = self.topic_extractor.extract_context_tags(segment.content)
        
        # 개념 태그 추출
        segment.concept_tags = self.concept_extractor.extract_concepts(segment.content)
        
        # 감정 분석
        segment.emotion = self.emotion_analyzer.analyze(segment.content)
        
        # 중요도 계산
        segment.importance = self._calculate_importance(segment)
    
    def _extract_overall_topics(self, segments: List[TextSegment]) -> List[str]:
        """전체 대화의 주제 추출"""
        all_content = " ".join([s.content for s in segments])
        return self.topic_extractor.extract_multiple_topics(all_content)
    
    def _calculate_importance(self, segment: TextSegment) -> float:
        """세그먼트 중요도 계산"""
        importance = 0.5  # 기본값
        
        # 길이 기반
        if len(segment.content) > 100:
            importance += 0.1
            
        # 질문인 경우
        if '?' in segment.content:
            importance += 0.1
            
        # 개념 태그 수
        importance += len(segment.concept_tags) * 0.05
        
        # 감정 강도
        if segment.emotion in ['excited', 'curious', 'concerned']:
            importance += 0.1
            
        return min(1.0, importance)
    
    def _store_conversation(self, conversation: ConversationMemory):
        """대화를 메모리 그래프에 저장"""
        # 대화 세션 노드 생성
        session_node = ThoughtNode.create(
            content=f"대화 세션: {conversation.title}",
            node_type="conversation_session",
            meta={
                "session_id": conversation.session_id,
                "title": conversation.title,
                "topics": conversation.overall_topics,
                "segment_count": len(conversation.segments),
                "created_at": conversation.created_at
            }
        )
        self.memory.add_node(session_node)
        
        # 각 세그먼트를 노드로 저장
        for segment in conversation.segments:
            segment_node = ThoughtNode.create(
                content=segment.content,
                node_type="conversation_segment",
                vectors={
                    "T": segment.importance,  # Topic importance
                    "C": 0.8,  # Context relevance
                    "I": 0.7,  # Information density
                    "E": self._emotion_to_vector(segment.emotion),  # Emotion
                    "M": segment.importance  # Memory importance
                },
                meta={
                    "segment_id": segment.id,
                    "speaker": segment.speaker,
                    "topic": segment.topic,
                    "context_tags": segment.context_tags,
                    "concept_tags": segment.concept_tags,
                    "emotion": segment.emotion,
                    "importance": segment.importance,
                    "timestamp": segment.timestamp,
                    "session_id": conversation.session_id
                }
            )
            self.memory.add_node(segment_node)
            
            # 세션과 세그먼트 연결
            session_edge = Relationship.create(
                source=session_node.id,
                target=segment_node.id,
                predicate="contains",
                weight=segment.importance
            )
            self.memory.add_edge(session_edge)
            
            # 관련 개념들 연결
            for concept in segment.concept_tags:
                concept_node = self._get_or_create_concept_node(concept)
                concept_edge = Relationship.create(
                    source=segment_node.id,
                    target=concept_node.id,
                    predicate="mentions",
                    weight=0.8
                )
                self.memory.add_edge(concept_edge)
        
        # 메모리 저장
        self.memory.save()
    
    def _get_or_create_concept_node(self, concept: str) -> ThoughtNode:
        """개념 노드 생성 또는 기존 노드 반환"""
        # 기존 개념 노드 찾기
        for node in self.memory.nodes.values():
            if (node.node_type == "concept" and 
                node.content.lower() == concept.lower()):
                return node
        
        # 새 개념 노드 생성
        concept_node = ThoughtNode.create(
            content=concept,
            node_type="concept",
            vectors={"M": 0.6},  # Memory importance
            meta={"concept_type": "extracted", "usage_count": 1}
        )
        self.memory.add_node(concept_node)
        return concept_node
    
    def _emotion_to_vector(self, emotion: str) -> float:
        """감정을 벡터 값으로 변환"""
        emotion_map = {
            "neutral": 0.5,
            "happy": 0.8,
            "excited": 0.9,
            "curious": 0.7,
            "concerned": 0.3,
            "sad": 0.2,
            "angry": 0.1
        }
        return emotion_map.get(emotion, 0.5)


class TopicExtractor:
    """주제 추출기"""
    
    def __init__(self):
        self.topic_keywords = {
            "기술": ["프로그래밍", "코딩", "개발", "소프트웨어", "알고리즘", "데이터베이스"],
            "과학": ["실험", "연구", "이론", "가설", "분석", "측정"],
            "예술": ["창작", "디자인", "미술", "음악", "문학", "영화"],
            "경제": ["투자", "경영", "마케팅", "재무", "거래", "시장"],
            "교육": ["학습", "교육", "강의", "훈련", "지식", "스킬"],
            "건강": ["운동", "영양", "의학", "치료", "예방", "웰빙"],
            "정치": ["정책", "법률", "정부", "선거", "사회", "국가"],
            "환경": ["기후", "생태", "자연", "보존", "지속가능", "친환경"]
        }
    
    def extract(self, text: str) -> str:
        """주요 주제 추출"""
        text_lower = text.lower()
        scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[topic] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "일반"
    
    def extract_multiple_topics(self, text: str) -> List[str]:
        """여러 주제 추출"""
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topics.append(topic)
        
        return topics[:3]  # 상위 3개 주제
    
    def extract_context_tags(self, text: str) -> List[str]:
        """맥락 태그 추출"""
        tags = []
        text_lower = text.lower()
        
        # 질문 패턴
        if '?' in text:
            tags.append("질문")
        
        # 설명 패턴
        if any(word in text_lower for word in ["설명", "이해", "알려줘", "어떻게"]):
            tags.append("설명요청")
        
        # 예시 패턴
        if any(word in text_lower for word in ["예시", "예제", "사례", "예를"]):
            tags.append("예시요청")
        
        # 비교 패턴
        if any(word in text_lower for word in ["비교", "차이", "다른", "vs"]):
            tags.append("비교")
        
        # 문제해결 패턴
        if any(word in text_lower for word in ["문제", "해결", "방법", "해결책"]):
            tags.append("문제해결")
        
        return tags


class ConceptExtractor:
    """개념 추출기"""
    
    def __init__(self):
        self.concept_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 대문자로 시작하는 명사구
            r'\b\d+[A-Za-z]+\b',  # 숫자+문자 조합
            r'\b[A-Z]{2,}\b',  # 대문자 약어
        ]
    
    def extract_concepts(self, text: str) -> List[str]:
        """개념 추출"""
        concepts = set()
        
        # 패턴 매칭
        for pattern in self.concept_patterns:
            matches = re.findall(pattern, text)
            concepts.update(matches)
        
        # 키워드 기반 추출
        keywords = self._extract_keywords(text)
        concepts.update(keywords)
        
        return list(concepts)[:10]  # 상위 10개
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 사용 가능)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 불용어 제거
        stopwords = {'이', '그', '저', '것', '수', '등', '및', '또는', '그리고', '하지만', '그런데'}
        keywords = [word for word in words if word not in stopwords and len(word) > 1]
        
        # 빈도 기반 상위 키워드
        from collections import Counter
        counter = Counter(keywords)
        return [word for word, count in counter.most_common(5)]


class EmotionAnalyzer:
    """감정 분석기"""
    
    def __init__(self):
        self.emotion_keywords = {
            "happy": ["좋아", "행복", "즐거워", "재미있어", "감사해"],
            "excited": ["흥미", "신기", "놀라워", "대단해", "멋져"],
            "curious": ["궁금", "알고싶어", "어떻게", "왜", "무엇"],
            "concerned": ["걱정", "염려", "불안", "어려워", "힘들어"],
            "sad": ["슬퍼", "우울", "실망", "아쉬워", "후회"],
            "angry": ["화나", "짜증", "분노", "열받", "불만"]
        }
    
    def analyze(self, text: str) -> str:
        """감정 분석"""
        text_lower = text.lower()
        scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[emotion] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "neutral"


# 사용 예시
def create_text_importer(memory_path: str = "memory/data/palantir_graph.json") -> TextImporter:
    """TextImporter 인스턴스 생성"""
    memory = PalantirGraph(persist_path=memory_path)
    return TextImporter(memory) 
