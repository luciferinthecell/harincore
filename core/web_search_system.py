"""
하린코어 웹 검색 및 API 통합 시스템 - Web Search & API Integration System
실시간 웹 검색, API 연동, 정보 수집 및 처리 시스템
"""

import asyncio
import aiohttp
import requests
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
import urllib.parse
from bs4 import BeautifulSoup
import logging

# 검색 엔진 API 키 (실제 사용 시 환경변수로 관리)
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
except ImportError:
    GOOGLE_API_KEY = None
    GOOGLE_CSE_ID = None
    OPENAI_API_KEY = None


class SearchEngine(Enum):
    """검색 엔진 유형"""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    WIKIPEDIA = "wikipedia"
    NEWS = "news"
    ACADEMIC = "academic"


class APIType(Enum):
    """API 유형"""
    WEATHER = "weather"
    NEWS = "news"
    TRANSLATION = "translation"
    CALCULATION = "calculation"
    KNOWLEDGE = "knowledge"
    SOCIAL = "social"
    FINANCE = "finance"
    HEALTH = "health"


class ContentType(Enum):
    """콘텐츠 유형"""
    TEXT = "text"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class SearchResult:
    """검색 결과"""
    title: str
    url: str
    snippet: str
    content_type: ContentType
    source: SearchEngine
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    """API 응답"""
    success: bool
    data: Any
    api_type: APIType
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SearchQuery(BaseModel):
    """검색 쿼리"""
    query: str
    search_engines: List[SearchEngine] = Field(default_factory=lambda: [SearchEngine.GOOGLE])
    max_results: int = 10
    language: str = "ko"
    time_range: Optional[str] = None  # "1d", "1w", "1m", "1y"
    content_type: Optional[ContentType] = None
    filters: Dict[str, Any] = Field(default_factory=dict)


class APIConfig(BaseModel):
    """API 설정"""
    api_type: APIType
    base_url: str
    api_key: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    rate_limit: int = 100  # 분당 요청 수
    timeout: int = 30
    retry_count: int = 3


class WebSearchSystem:
    """웹 검색 시스템"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 검색 엔진 설정
        self.search_engines = {
            SearchEngine.GOOGLE: {
                "enabled": bool(GOOGLE_API_KEY and GOOGLE_CSE_ID),
                "api_key": GOOGLE_API_KEY,
                "cse_id": GOOGLE_CSE_ID,
                "base_url": "https://www.googleapis.com/customsearch/v1"
            },
            SearchEngine.BING: {
                "enabled": True,
                "base_url": "https://api.bing.microsoft.com/v7.0/search"
            },
            SearchEngine.DUCKDUCKGO: {
                "enabled": True,
                "base_url": "https://api.duckduckgo.com/"
            },
            SearchEngine.WIKIPEDIA: {
                "enabled": True,
                "base_url": "https://ko.wikipedia.org/api/rest_v1/"
            }
        }
        
        # API 설정
        self.api_configs = self._initialize_api_configs()
        
        # 캐시 시스템
        self.search_cache = {}
        self.api_cache = {}
        self.cache_ttl = 3600  # 1시간
        
        # 요청 제한 관리
        self.rate_limits = {}
        self.last_requests = {}
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        print("웹 검색 시스템 초기화 완료")
    
    def _initialize_api_configs(self) -> Dict[APIType, APIConfig]:
        """API 설정 초기화"""
        configs = {
            APIType.WEATHER: APIConfig(
                api_type=APIType.WEATHER,
                base_url="https://api.openweathermap.org/data/2.5/",
                api_key=os.getenv('OPENWEATHER_API_KEY'),
                rate_limit=60
            ),
            APIType.NEWS: APIConfig(
                api_type=APIType.NEWS,
                base_url="https://newsapi.org/v2/",
                api_key=os.getenv('NEWS_API_KEY'),
                rate_limit=100
            ),
            APIType.TRANSLATION: APIConfig(
                api_type=APIType.TRANSLATION,
                base_url="https://translation.googleapis.com/language/translate/v2",
                api_key=GOOGLE_API_KEY,
                rate_limit=100
            ),
            APIType.CALCULATION: APIConfig(
                api_type=APIType.CALCULATION,
                base_url="https://api.mathjs.org/v4/",
                rate_limit=1000
            ),
            APIType.KNOWLEDGE: APIConfig(
                api_type=APIType.KNOWLEDGE,
                base_url="https://api.wolframalpha.com/v1/",
                api_key=os.getenv('WOLFRAM_API_KEY'),
                rate_limit=50
            )
        }
        return configs
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """통합 검색 실행"""
        results = []
        
        # 캐시 확인
        cache_key = self._generate_cache_key(query)
        if cache_key in self.search_cache:
            cached_results, timestamp = self.search_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                self.logger.info(f"캐시된 검색 결과 사용: {query.query}")
                return cached_results
        
        # 각 검색 엔진에서 검색
        for engine in query.search_engines:
            if not self.search_engines[engine]["enabled"]:
                continue
            
            try:
                engine_results = await self._search_with_engine(query, engine)
                results.extend(engine_results)
                
                # 요청 제한 준수
                await self._respect_rate_limit(engine)
                
            except Exception as e:
                self.logger.error(f"{engine.value} 검색 실패: {e}")
        
        # 결과 정렬 및 필터링
        results = self._process_search_results(results, query)
        
        # 캐시 저장
        self.search_cache[cache_key] = (results, datetime.now())
        
        return results
    
    async def _search_with_engine(self, query: SearchQuery, engine: SearchEngine) -> List[SearchResult]:
        """특정 검색 엔진으로 검색"""
        if engine == SearchEngine.GOOGLE:
            return await self._google_search(query)
        elif engine == SearchEngine.BING:
            return await self._bing_search(query)
        elif engine == SearchEngine.DUCKDUCKGO:
            return await self._duckduckgo_search(query)
        elif engine == SearchEngine.WIKIPEDIA:
            return await self._wikipedia_search(query)
        else:
            return []
    
    async def _google_search(self, query: SearchQuery) -> List[SearchResult]:
        """Google Custom Search API 사용"""
        if not self.search_engines[SearchEngine.GOOGLE]["enabled"]:
            return []
        
        params = {
            'key': self.search_engines[SearchEngine.GOOGLE]["api_key"],
            'cx': self.search_engines[SearchEngine.GOOGLE]["cse_id"],
            'q': query.query,
            'num': min(query.max_results, 10),
            'lr': f"lang_{query.language}"
        }
        
        if query.time_range:
            params['dateRestrict'] = query.time_range
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.search_engines[SearchEngine.GOOGLE]["base_url"],
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_google_results(data)
                else:
                    self.logger.error(f"Google 검색 API 오류: {response.status}")
                    return []
    
    async def _bing_search(self, query: SearchQuery) -> List[SearchResult]:
        """Bing Search API 사용"""
        headers = {
            'Ocp-Apim-Subscription-Key': os.getenv('BING_API_KEY', ''),
            'Accept': 'application/json'
        }
        
        params = {
            'q': query.query,
            'count': query.max_results,
            'mkt': 'ko-KR'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.search_engines[SearchEngine.BING]["base_url"],
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_bing_results(data)
                else:
                    self.logger.error(f"Bing 검색 API 오류: {response.status}")
                    return []
    
    async def _duckduckgo_search(self, query: SearchQuery) -> List[SearchResult]:
        """DuckDuckGo Instant Answer API 사용"""
        params = {
            'q': query.query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.search_engines[SearchEngine.DUCKDUCKGO]["base_url"],
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_duckduckgo_results(data)
                else:
                    self.logger.error(f"DuckDuckGo 검색 API 오류: {response.status}")
                    return []
    
    async def _wikipedia_search(self, query: SearchQuery) -> List[SearchResult]:
        """Wikipedia API 사용"""
        # 검색 API
        search_url = f"{self.search_engines[SearchEngine.WIKIPEDIA]['base_url']}page/search/{urllib.parse.quote(query.query)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_wikipedia_results(data)
                else:
                    self.logger.error(f"Wikipedia 검색 API 오류: {response.status}")
                    return []
    
    def _parse_google_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Google 검색 결과 파싱"""
        results = []
        
        if 'items' in data:
            for item in data['items']:
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    content_type=ContentType.TEXT,
                    source=SearchEngine.GOOGLE,
                    relevance_score=0.8,
                    metadata={
                        'displayLink': item.get('displayLink', ''),
                        'pagemap': item.get('pagemap', {})
                    }
                )
                results.append(result)
        
        return results
    
    def _parse_bing_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Bing 검색 결과 파싱"""
        results = []
        
        if 'webPages' in data and 'value' in data['webPages']:
            for item in data['webPages']['value']:
                result = SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    content_type=ContentType.TEXT,
                    source=SearchEngine.BING,
                    relevance_score=0.8,
                    metadata={
                        'displayUrl': item.get('displayUrl', ''),
                        'dateLastCrawled': item.get('dateLastCrawled', '')
                    }
                )
                results.append(result)
        
        return results
    
    def _parse_duckduckgo_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """DuckDuckGo 검색 결과 파싱"""
        results = []
        
        # Instant Answer
        if data.get('Abstract'):
            result = SearchResult(
                title=data.get('Heading', ''),
                url=data.get('AbstractURL', ''),
                snippet=data.get('Abstract', ''),
                content_type=ContentType.TEXT,
                source=SearchEngine.DUCKDUCKGO,
                relevance_score=0.9,
                metadata={
                    'abstractSource': data.get('AbstractSource', ''),
                    'image': data.get('Image', '')
                }
            )
            results.append(result)
        
        # Related Topics
        if 'RelatedTopics' in data:
            for topic in data['RelatedTopics'][:5]:
                if isinstance(topic, dict) and 'Text' in topic:
                    result = SearchResult(
                        title=topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                        url=topic.get('FirstURL', ''),
                        snippet=topic.get('Text', ''),
                        content_type=ContentType.TEXT,
                        source=SearchEngine.DUCKDUCKGO,
                        relevance_score=0.7
                    )
                    results.append(result)
        
        return results
    
    def _parse_wikipedia_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Wikipedia 검색 결과 파싱"""
        results = []
        
        if 'pages' in data:
            for page in data['pages']:
                result = SearchResult(
                    title=page.get('title', ''),
                    url=page.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    snippet=page.get('extract', '')[:200] + '...' if page.get('extract') else '',
                    content_type=ContentType.TEXT,
                    source=SearchEngine.WIKIPEDIA,
                    relevance_score=0.9,
                    metadata={
                        'pageid': page.get('pageid', ''),
                        'thumbnail': page.get('thumbnail', {}).get('source', '') if page.get('thumbnail') else ''
                    }
                )
                results.append(result)
        
        return results
    
    def _process_search_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """검색 결과 처리 및 정렬"""
        # 중복 제거
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # 관련성 점수 계산
        for result in unique_results:
            result.relevance_score = self._calculate_relevance_score(result, query)
        
        # 관련성 점수로 정렬
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 최대 결과 수 제한
        return unique_results[:query.max_results]
    
    def _calculate_relevance_score(self, result: SearchResult, query: SearchQuery) -> float:
        """관련성 점수 계산"""
        score = result.relevance_score
        
        # 쿼리 키워드 매칭
        query_words = query.query.lower().split()
        title_words = result.title.lower().split()
        snippet_words = result.snippet.lower().split()
        
        # 제목에서 키워드 매칭
        title_matches = sum(1 for word in query_words if word in title_words)
        score += title_matches * 0.2
        
        # 스니펫에서 키워드 매칭
        snippet_matches = sum(1 for word in query_words if word in snippet_words)
        score += snippet_matches * 0.1
        
        # 도메인 신뢰도
        if 'wikipedia.org' in result.url:
            score += 0.1
        elif 'edu' in result.url:
            score += 0.05
        
        return min(score, 1.0)
    
    async def call_api(self, api_type: APIType, endpoint: str, params: Dict[str, Any] = None) -> APIResponse:
        """API 호출"""
        start_time = time.time()
        
        if api_type not in self.api_configs:
            return APIResponse(
                success=False,
                data=None,
                api_type=api_type,
                error_message=f"지원하지 않는 API 유형: {api_type.value}"
            )
        
        config = self.api_configs[api_type]
        
        # 캐시 확인
        cache_key = f"{api_type.value}_{endpoint}_{hash(str(params))}"
        if cache_key in self.api_cache:
            cached_response, timestamp = self.api_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return cached_response
        
        # 요청 제한 확인
        if not await self._check_rate_limit(api_type):
            return APIResponse(
                success=False,
                data=None,
                api_type=api_type,
                error_message="요청 제한 초과"
            )
        
        try:
            # API 호출
            url = f"{config.base_url}{endpoint}"
            headers = config.headers.copy()
            
            if config.api_key:
                if api_type == APIType.WEATHER:
                    params = params or {}
                    params['appid'] = config.api_key
                else:
                    headers['Authorization'] = f"Bearer {config.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        api_response = APIResponse(
                            success=True,
                            data=data,
                            api_type=api_type,
                            response_time=response_time
                        )
                        
                        # 캐시 저장
                        self.api_cache[cache_key] = (api_response, datetime.now())
                        
                        return api_response
                    else:
                        return APIResponse(
                            success=False,
                            data=None,
                            api_type=api_type,
                            response_time=response_time,
                            error_message=f"API 오류: {response.status}"
                        )
        
        except Exception as e:
            response_time = time.time() - start_time
            return APIResponse(
                success=False,
                data=None,
                api_type=api_type,
                response_time=response_time,
                error_message=str(e)
            )
    
    async def get_weather(self, city: str, country_code: str = "KR") -> APIResponse:
        """날씨 정보 조회"""
        return await self.call_api(
            APIType.WEATHER,
            "weather",
            {
                'q': f"{city},{country_code}",
                'units': 'metric',
                'lang': 'kr'
            }
        )
    
    async def get_news(self, query: str = None, category: str = None) -> APIResponse:
        """뉴스 정보 조회"""
        params = {'language': 'ko', 'sortBy': 'publishedAt'}
        
        if query:
            params['q'] = query
        if category:
            params['category'] = category
        
        return await self.call_api(APIType.NEWS, "everything", params)
    
    async def translate_text(self, text: str, target_lang: str = "en", source_lang: str = "ko") -> APIResponse:
        """텍스트 번역"""
        return await self.call_api(
            APIType.TRANSLATION,
            "",
            {
                'q': text,
                'target': target_lang,
                'source': source_lang
            }
        )
    
    async def calculate_expression(self, expression: str) -> APIResponse:
        """수학 표현식 계산"""
        return await self.call_api(
            APIType.CALCULATION,
            "",
            {'expr': expression}
        )
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """캐시 키 생성"""
        return f"search_{hash(query.query)}_{hash(str(query.search_engines))}_{query.max_results}"
    
    async def _respect_rate_limit(self, engine: SearchEngine):
        """요청 제한 준수"""
        if engine.value in self.rate_limits:
            limit = self.rate_limits[engine.value]
            if limit > 0:
                await asyncio.sleep(60 / limit)  # 분당 요청 수에 따른 대기
    
    async def _check_rate_limit(self, api_type: APIType) -> bool:
        """요청 제한 확인"""
        config = self.api_configs[api_type]
        current_time = time.time()
        
        if api_type.value not in self.last_requests:
            self.last_requests[api_type.value] = []
        
        # 1분 이내 요청들 필터링
        recent_requests = [
            req_time for req_time in self.last_requests[api_type.value]
            if current_time - req_time < 60
        ]
        
        if len(recent_requests) >= config.rate_limit:
            return False
        
        self.last_requests[api_type.value] = recent_requests + [current_time]
        return True
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 조회"""
        return {
            "cache_size": {
                "search_cache": len(self.search_cache),
                "api_cache": len(self.api_cache)
            },
            "enabled_engines": [
                engine.value for engine, config in self.search_engines.items()
                if config["enabled"]
            ],
            "enabled_apis": [
                api_type.value for api_type in self.api_configs.keys()
            ],
            "rate_limits": self.rate_limits
        }
    
    def clear_cache(self, cache_type: str = "all"):
        """캐시 정리"""
        if cache_type == "all" or cache_type == "search":
            self.search_cache.clear()
        if cache_type == "all" or cache_type == "api":
            self.api_cache.clear()
        
        self.logger.info(f"캐시 정리 완료: {cache_type}")


# 전역 웹 검색 시스템 인스턴스
web_search_system = WebSearchSystem() 