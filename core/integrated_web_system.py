"""
하린코어 통합 웹 시스템 - Integrated Web System
웹 검색, API 통합, 실시간 정보 수집을 통합하는 시스템
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# 하린코어 시스템 import
from core.web_search_system import WebSearchSystem, SearchQuery, SearchEngine, APIType
from core.cognitive_cycle import CognitiveCycle
from core.emotion_system import EmotionSystem
from core.action_system import ActionSystem
from core.integrated_monitoring import IntegratedMonitoringSystem


class IntegratedWebSystem:
    """통합 웹 시스템"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 웹 검색 시스템 초기화
        self.web_search_system = WebSearchSystem()
        
        # 하린코어 시스템들
        self.cognitive_cycle = None
        self.emotion_system = None
        self.action_system = None
        self.monitoring_system = None
        
        # 웹 시스템 설정
        self.web_active = False
        self.search_interval = self.config.get('search_interval', 10.0)  # 10초
        self.api_interval = self.config.get('api_interval', 30.0)  # 30초
        
        # 데이터 저장 경로
        self.data_dir = Path("data/web_integration")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 웹 시스템 스레드
        self.web_thread = None
        self.last_search_update = 0
        self.last_api_update = 0
        
        # 검색 히스토리
        self.search_history = []
        self.api_call_history = []
        
        print("통합 웹 시스템 초기화 완료")
    
    def connect_systems(self, cognitive_cycle: CognitiveCycle = None, 
                       emotion_system: EmotionSystem = None, 
                       action_system: ActionSystem = None,
                       monitoring_system: IntegratedMonitoringSystem = None):
        """하린코어 시스템들 연결"""
        try:
            # 시스템 연결
            if cognitive_cycle:
                self.cognitive_cycle = cognitive_cycle
                cognitive_cycle.set_web_search_system(self.web_search_system)
                print("인지 사이클이 웹 검색 시스템에 연결되었습니다.")
            
            if emotion_system:
                self.emotion_system = emotion_system
                print("감정 시스템이 웹 시스템에 연결되었습니다.")
            
            if action_system:
                self.action_system = action_system
                print("행동 시스템이 웹 시스템에 연결되었습니다.")
            
            if monitoring_system:
                self.monitoring_system = monitoring_system
                print("모니터링 시스템이 웹 시스템에 연결되었습니다.")
            
            # 웹 시스템 시작
            self.start_web_system()
            
        except Exception as e:
            print(f"시스템 연결 오류: {e}")
    
    def start_web_system(self):
        """웹 시스템 시작"""
        if not self.web_active:
            self.web_active = True
            self.web_thread = asyncio.create_task(self._web_system_loop())
            print("통합 웹 시스템 시작")
    
    def stop_web_system(self):
        """웹 시스템 중지"""
        self.web_active = False
        if self.web_thread:
            self.web_thread.cancel()
        print("통합 웹 시스템 중지")
    
    async def _web_system_loop(self):
        """웹 시스템 루프"""
        while self.web_active:
            try:
                # 주기적 검색 업데이트
                current_time = time.time()
                
                if current_time - self.last_search_update >= self.search_interval:
                    await self._update_search_data()
                    self.last_search_update = current_time
                
                if current_time - self.last_api_update >= self.api_interval:
                    await self._update_api_data()
                    self.last_api_update = current_time
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"웹 시스템 루프 오류: {e}")
                await asyncio.sleep(5.0)
    
    async def _update_search_data(self):
        """검색 데이터 업데이트"""
        try:
            # 인지 사이클에서 검색 요약 조회
            if self.cognitive_cycle:
                search_summary = self.cognitive_cycle.get_web_search_summary()
                if search_summary.get('status') == 'success':
                    # 검색 히스토리에 추가
                    self.search_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'summary': search_summary
                    })
                    
                    # 히스토리 제한 (최근 100개)
                    if len(self.search_history) > 100:
                        self.search_history = self.search_history[-100:]
                    
                    # 모니터링 시스템에 검색 메트릭 전송
                    if self.monitoring_system:
                        self._send_search_metrics_to_monitoring(search_summary)
                
        except Exception as e:
            print(f"검색 데이터 업데이트 오류: {e}")
    
    async def _update_api_data(self):
        """API 데이터 업데이트"""
        try:
            # API 호출 통계 수집
            api_statistics = self.web_search_system.get_search_statistics()
            
            # API 호출 히스토리에 추가
            self.api_call_history.append({
                'timestamp': datetime.now().isoformat(),
                'statistics': api_statistics
            })
            
            # 히스토리 제한 (최근 50개)
            if len(self.api_call_history) > 50:
                self.api_call_history = self.api_call_history[-50:]
            
            # 모니터링 시스템에 API 메트릭 전송
            if self.monitoring_system:
                self._send_api_metrics_to_monitoring(api_statistics)
                
        except Exception as e:
            print(f"API 데이터 업데이트 오류: {e}")
    
    def _send_search_metrics_to_monitoring(self, search_summary: Dict[str, Any]):
        """검색 메트릭을 모니터링 시스템에 전송"""
        try:
            if hasattr(self.monitoring_system, 'monitoring_system'):
                # 검색 성공률 메트릭
                total_searches = search_summary.get('web_search_results', 0)
                if total_searches > 0:
                    success_rate = min(1.0, total_searches / 100.0)  # 정규화
                    self.monitoring_system.monitoring_system._add_metric_point(
                        self.monitoring_system.monitoring_system.MetricType.PERFORMANCE,
                        'search_success_rate',
                        success_rate,
                        time.time()
                    )
                
                # 검색 효율성 메트릭
                recent_searches = search_summary.get('recent_searches', [])
                if recent_searches:
                    avg_relevance = sum(s.get('relevance_score', 0.0) for s in recent_searches) / len(recent_searches)
                    self.monitoring_system.monitoring_system._add_metric_point(
                        self.monitoring_system.monitoring_system.MetricType.COGNITIVE,
                        'search_relevance',
                        avg_relevance,
                        time.time()
                    )
                
        except Exception as e:
            print(f"검색 메트릭 전송 오류: {e}")
    
    def _send_api_metrics_to_monitoring(self, api_statistics: Dict[str, Any]):
        """API 메트릭을 모니터링 시스템에 전송"""
        try:
            if hasattr(self.monitoring_system, 'monitoring_system'):
                # API 응답 시간 메트릭
                cache_size = api_statistics.get('cache_size', {})
                api_cache_size = cache_size.get('api_cache', 0)
                normalized_cache_size = min(1.0, api_cache_size / 100.0)
                
                self.monitoring_system.monitoring_system._add_metric_point(
                    self.monitoring_system.monitoring_system.MetricType.PERFORMANCE,
                    'api_cache_efficiency',
                    normalized_cache_size,
                    time.time()
                )
                
                # 활성 API 수 메트릭
                enabled_apis = api_statistics.get('enabled_apis', [])
                api_count = len(enabled_apis)
                normalized_api_count = min(1.0, api_count / 10.0)
                
                self.monitoring_system.monitoring_system._add_metric_point(
                    self.monitoring_system.monitoring_system.MetricType.SYSTEM,
                    'active_api_count',
                    normalized_api_count,
                    time.time()
                )
                
        except Exception as e:
            print(f"API 메트릭 전송 오류: {e}")
    
    async def perform_intelligent_search(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """지능형 검색 수행"""
        try:
            # 컨텍스트 기반 검색 엔진 선택
            search_engines = self._select_search_engines_by_context(context)
            
            # 검색 쿼리 최적화
            optimized_query = self._optimize_search_query(query, context)
            
            # 검색 실행
            search_query = SearchQuery(
                query=optimized_query,
                search_engines=search_engines,
                max_results=10,
                language="ko"
            )
            
            results = await self.web_search_system.search(search_query)
            
            # 결과 후처리
            processed_results = self._process_search_results_for_context(results, context)
            
            # 검색 히스토리에 추가
            self.search_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'optimized_query': optimized_query,
                'results_count': len(results),
                'context': context
            })
            
            return {
                'success': True,
                'query': query,
                'optimized_query': optimized_query,
                'results': processed_results,
                'total_results': len(results),
                'search_engines_used': [engine.value for engine in search_engines]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
    
    def _select_search_engines_by_context(self, context: Dict[str, Any] = None) -> List[SearchEngine]:
        """컨텍스트 기반 검색 엔진 선택"""
        if not context:
            return [SearchEngine.GOOGLE, SearchEngine.WIKIPEDIA]
        
        search_engines = []
        
        # 학술적 정보인 경우
        if context.get('academic', False):
            search_engines.extend([SearchEngine.GOOGLE, SearchEngine.WIKIPEDIA])
        
        # 뉴스 정보인 경우
        elif context.get('news', False):
            search_engines.extend([SearchEngine.GOOGLE, SearchEngine.BING])
        
        # 기술적 정보인 경우
        elif context.get('technical', False):
            search_engines.extend([SearchEngine.GOOGLE, SearchEngine.WIKIPEDIA])
        
        # 일반 정보인 경우
        else:
            search_engines.extend([SearchEngine.GOOGLE, SearchEngine.DUCKDUCKGO])
        
        return search_engines
    
    def _optimize_search_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """검색 쿼리 최적화"""
        optimized_query = query
        
        # 컨텍스트 기반 키워드 추가
        if context:
            if context.get('academic', False):
                optimized_query += " 학술 논문 연구"
            elif context.get('news', False):
                optimized_query += " 최신 뉴스"
            elif context.get('technical', False):
                optimized_query += " 기술 문서"
        
        # 언어 최적화
        if not any(ord(char) > 127 for char in query):  # 한글만 있는 경우
            optimized_query += " 한국어"
        
        return optimized_query
    
    def _process_search_results_for_context(self, results: List, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """컨텍스트 기반 검색 결과 후처리"""
        processed_results = []
        
        for result in results:
            processed_result = {
                'title': result.title,
                'url': result.url,
                'snippet': result.snippet,
                'source': result.source.value,
                'relevance_score': result.relevance_score,
                'context_relevance': self._calculate_context_relevance(result, context)
            }
            
            # 컨텍스트 관련성 점수로 필터링
            if processed_result['context_relevance'] > 0.3:
                processed_results.append(processed_result)
        
        # 컨텍스트 관련성으로 정렬
        processed_results.sort(key=lambda x: x['context_relevance'], reverse=True)
        
        return processed_results
    
    def _calculate_context_relevance(self, result, context: Dict[str, Any] = None) -> float:
        """컨텍스트 관련성 계산"""
        if not context:
            return result.relevance_score
        
        relevance = result.relevance_score
        
        # 학술적 컨텍스트
        if context.get('academic', False):
            if 'wikipedia.org' in result.url or 'edu' in result.url:
                relevance += 0.2
            if any(word in result.title.lower() for word in ['연구', '논문', '학술', '분석']):
                relevance += 0.1
        
        # 뉴스 컨텍스트
        elif context.get('news', False):
            if any(word in result.title.lower() for word in ['뉴스', '소식', '발표', '발표']):
                relevance += 0.2
            if any(word in result.snippet.lower() for word in ['최신', '오늘', '어제', '발표']):
                relevance += 0.1
        
        # 기술적 컨텍스트
        elif context.get('technical', False):
            if any(word in result.title.lower() for word in ['기술', '개발', '프로그래밍', '코드']):
                relevance += 0.2
            if 'github.com' in result.url or 'stackoverflow.com' in result.url:
                relevance += 0.1
        
        return min(relevance, 1.0)
    
    async def get_comprehensive_information(self, query: str, info_types: List[str] = None) -> Dict[str, Any]:
        """종합 정보 조회"""
        try:
            info_types = info_types or ['search', 'weather', 'news']
            comprehensive_info = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'sources': {}
            }
            
            # 웹 검색
            if 'search' in info_types:
                search_results = await self.perform_intelligent_search(query)
                comprehensive_info['sources']['web_search'] = search_results
            
            # 날씨 정보
            if 'weather' in info_types:
                weather_response = await self.web_search_system.get_weather('Seoul')
                comprehensive_info['sources']['weather'] = {
                    'success': weather_response.success,
                    'data': weather_response.data if weather_response.success else None
                }
            
            # 뉴스 정보
            if 'news' in info_types:
                news_response = await self.web_search_system.get_news(query=query)
                comprehensive_info['sources']['news'] = {
                    'success': news_response.success,
                    'data': news_response.data if news_response.success else None
                }
            
            # 번역 정보
            if 'translation' in info_types:
                translation_response = await self.web_search_system.translate_text(query)
                comprehensive_info['sources']['translation'] = {
                    'success': translation_response.success,
                    'data': translation_response.data if translation_response.success else None
                }
            
            return comprehensive_info
            
        except Exception as e:
            return {
                'error': str(e),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_web_system_summary(self) -> Dict[str, Any]:
        """웹 시스템 요약 조회"""
        try:
            # 검색 통계
            search_statistics = self.web_search_system.get_search_statistics()
            
            # 최근 검색 히스토리
            recent_searches = self.search_history[-10:] if self.search_history else []
            
            # 최근 API 호출 히스토리
            recent_api_calls = self.api_call_history[-10:] if self.api_call_history else []
            
            # 시스템별 웹 검색 요약
            cognitive_web_summary = None
            if self.cognitive_cycle:
                cognitive_web_summary = self.cognitive_cycle.get_web_search_summary()
            
            return {
                'status': 'success',
                'web_system_active': self.web_active,
                'search_statistics': search_statistics,
                'recent_searches': recent_searches,
                'recent_api_calls': recent_api_calls,
                'cognitive_web_summary': cognitive_web_summary,
                'connected_systems': {
                    'cognitive_cycle': self.cognitive_cycle is not None,
                    'emotion_system': self.emotion_system is not None,
                    'action_system': self.action_system is not None,
                    'monitoring_system': self.monitoring_system is not None
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def generate_web_dashboard(self, filepath: str):
        """웹 시스템 대시보드 생성"""
        try:
            # 웹 시스템 요약 조회
            summary = self.get_web_system_summary()
            
            if summary.get('status') != 'success':
                print("웹 시스템 요약 조회 실패")
                return
            
            # 대시보드 HTML 생성
            dashboard_html = self._create_web_dashboard_html(summary)
            
            # 파일 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            
            print(f"웹 시스템 대시보드 생성 완료: {filepath}")
            
        except Exception as e:
            print(f"웹 시스템 대시보드 생성 실패: {e}")
    
    def _create_web_dashboard_html(self, summary: Dict[str, Any]) -> str:
        """웹 시스템 대시보드 HTML 생성"""
        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>하린코어 웹 시스템 대시보드</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .content {{
            padding: 20px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #fafafa;
        }}
        .section h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            min-width: 150px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .search-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .search-query {{
            font-weight: bold;
            color: #333;
        }}
        .search-meta {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .status-active {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-inactive {{
            color: #dc3545;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 하린코어 웹 시스템 대시보드</h1>
            <p>실시간 웹 검색 및 API 통합 모니터링</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h3>📊 시스템 상태</h3>
                <div class="metric">
                    <div class="metric-value">{'활성' if summary.get('web_system_active') else '비활성'}</div>
                    <div class="metric-label">웹 시스템 상태</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(summary.get('recent_searches', []))}</div>
                    <div class="metric-label">최근 검색 수</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(summary.get('recent_api_calls', []))}</div>
                    <div class="metric-label">최근 API 호출 수</div>
                </div>
            </div>
            
            <div class="section">
                <h3>🔍 최근 검색 히스토리</h3>
                {self._generate_search_history_html(summary.get('recent_searches', []))}
            </div>
            
            <div class="section">
                <h3>🔗 연결된 시스템</h3>
                {self._generate_connected_systems_html(summary.get('connected_systems', {}))}
            </div>
            
            <div class="section">
                <h3>📈 검색 통계</h3>
                {self._generate_search_statistics_html(summary.get('search_statistics', {}))}
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_search_history_html(self, searches: List[Dict[str, Any]]) -> str:
        """검색 히스토리 HTML 생성"""
        if not searches:
            return "<p>검색 히스토리가 없습니다.</p>"
        
        html = ""
        for search in reversed(searches[-5:]):  # 최근 5개
            query = search.get('query', 'N/A')
            timestamp = search.get('timestamp', 'N/A')
            results_count = search.get('results_count', 0)
            
            html += f"""
            <div class="search-item">
                <div class="search-query">{query}</div>
                <div class="search-meta">
                    시간: {timestamp} | 결과: {results_count}개
                </div>
            </div>
            """
        
        return html
    
    def _generate_connected_systems_html(self, systems: Dict[str, bool]) -> str:
        """연결된 시스템 HTML 생성"""
        html = ""
        for system_name, connected in systems.items():
            status_class = "status-active" if connected else "status-inactive"
            status_text = "연결됨" if connected else "연결되지 않음"
            
            html += f"""
            <div class="metric">
                <div class="metric-value {status_class}">{status_text}</div>
                <div class="metric-label">{system_name}</div>
            </div>
            """
        
        return html
    
    def _generate_search_statistics_html(self, statistics: Dict[str, Any]) -> str:
        """검색 통계 HTML 생성"""
        html = ""
        
        # 캐시 크기
        cache_size = statistics.get('cache_size', {})
        search_cache = cache_size.get('search_cache', 0)
        api_cache = cache_size.get('api_cache', 0)
        
        html += f"""
        <div class="metric">
            <div class="metric-value">{search_cache}</div>
            <div class="metric-label">검색 캐시</div>
        </div>
        <div class="metric">
            <div class="metric-value">{api_cache}</div>
            <div class="metric-label">API 캐시</div>
        </div>
        """
        
        # 활성 엔진 수
        enabled_engines = statistics.get('enabled_engines', [])
        html += f"""
        <div class="metric">
            <div class="metric-value">{len(enabled_engines)}</div>
            <div class="metric-label">활성 검색 엔진</div>
        </div>
        """
        
        return html
    
    def export_web_data(self, filepath: str):
        """웹 시스템 데이터 내보내기"""
        try:
            summary = self.get_web_system_summary()
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'web_system_summary': summary,
                'search_history': self.search_history,
                'api_call_history': self.api_call_history
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"웹 시스템 데이터 내보내기 완료: {filepath}")
            
        except Exception as e:
            print(f"웹 시스템 데이터 내보내기 실패: {e}")


# 전역 통합 웹 시스템 인스턴스
integrated_web_system = IntegratedWebSystem() 