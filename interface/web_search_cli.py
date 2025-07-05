"""
하린코어 웹 검색 CLI - Web Search Command Line Interface
웹 검색 및 API 통합 시스템 제어
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# 하린코어 시스템 import
import sys
sys.path.append('..')

from core.integrated_web_system import IntegratedWebSystem
from core.web_search_system import SearchQuery, SearchEngine, APIType
from core.cognitive_cycle import CognitiveCycle
from core.emotion_system import EmotionSystem
from core.action_system import ActionSystem


class WebSearchCLI:
    """웹 검색 CLI 클래스"""
    
    def __init__(self):
        self.integrated_web_system = IntegratedWebSystem()
        self.running = False
    
    def start_web_system(self, config: Dict[str, Any] = None):
        """웹 시스템 시작"""
        try:
            # 하린코어 시스템들 초기화 (실제 환경에서는 기존 인스턴스 사용)
            cognitive_cycle = CognitiveCycle()
            emotion_system = EmotionSystem()
            action_system = ActionSystem()
            
            # 시스템들 연결
            self.integrated_web_system.connect_systems(
                cognitive_cycle=cognitive_cycle,
                emotion_system=emotion_system,
                action_system=action_system
            )
            
            # 웹 시스템 시작
            self.integrated_web_system.start_web_system()
            self.running = True
            
            print("✅ 웹 검색 시스템이 시작되었습니다.")
            print("🌐 실시간 검색 및 API 통합이 활성화되었습니다.")
            
        except Exception as e:
            print(f"❌ 웹 시스템 시작 실패: {e}")
    
    def stop_web_system(self):
        """웹 시스템 중지"""
        try:
            self.integrated_web_system.stop_web_system()
            self.running = False
            print("✅ 웹 검색 시스템이 중지되었습니다.")
            
        except Exception as e:
            print(f"❌ 웹 시스템 중지 실패: {e}")
    
    def get_status(self):
        """현재 상태 조회"""
        try:
            summary = self.integrated_web_system.get_web_system_summary()
            
            print("\n" + "="*60)
            print("🌐 하린코어 웹 검색 시스템 상태")
            print("="*60)
            
            # 기본 상태
            print(f"📡 웹 시스템 활성화: {'✅ 활성' if summary.get('web_system_active') else '❌ 비활성'}")
            
            # 연결된 시스템
            print("\n🔗 연결된 시스템:")
            connected_systems = summary.get('connected_systems', {})
            for system_name, connected in connected_systems.items():
                status = "✅ 연결됨" if connected else "❌ 연결되지 않음"
                print(f"  - {system_name}: {status}")
            
            # 검색 통계
            print("\n📊 검색 통계:")
            search_stats = summary.get('search_statistics', {})
            cache_size = search_stats.get('cache_size', {})
            print(f"  - 검색 캐시: {cache_size.get('search_cache', 0)}개")
            print(f"  - API 캐시: {cache_size.get('api_cache', 0)}개")
            
            enabled_engines = search_stats.get('enabled_engines', [])
            print(f"  - 활성 검색 엔진: {len(enabled_engines)}개")
            for engine in enabled_engines:
                print(f"    * {engine}")
            
            # 최근 검색
            print("\n🔍 최근 검색:")
            recent_searches = summary.get('recent_searches', [])
            if recent_searches:
                for search in recent_searches[-3:]:  # 최근 3개
                    query = search.get('query', 'N/A')
                    timestamp = search.get('timestamp', 'N/A')
                    results_count = search.get('results_count', 0)
                    print(f"  - '{query}' ({results_count}개 결과) - {timestamp}")
            else:
                print("  - 검색 기록이 없습니다.")
            
            # 인지 사이클 웹 검색 요약
            cognitive_summary = summary.get('cognitive_web_summary', {})
            if cognitive_summary.get('status') == 'success':
                print("\n🧠 인지 사이클 웹 검색:")
                print(f"  - 웹 검색 결과: {cognitive_summary.get('web_search_results', 0)}개")
                print(f"  - API 응답: {cognitive_summary.get('api_responses', 0)}개")
            
            print("="*60)
            
        except Exception as e:
            print(f"❌ 상태 조회 실패: {e}")
    
    async def search(self, query: str, engines: List[str] = None, max_results: int = 10):
        """웹 검색 수행"""
        try:
            print(f"🔍 검색 중: '{query}'")
            
            # 검색 엔진 선택
            if not engines:
                engines = ['google', 'wikipedia']
            
            search_engines = []
            for engine in engines:
                if engine.lower() == 'google':
                    search_engines.append(SearchEngine.GOOGLE)
                elif engine.lower() == 'bing':
                    search_engines.append(SearchEngine.BING)
                elif engine.lower() == 'duckduckgo':
                    search_engines.append(SearchEngine.DUCKDUCKGO)
                elif engine.lower() == 'wikipedia':
                    search_engines.append(SearchEngine.WIKIPEDIA)
            
            # 지능형 검색 수행
            result = await self.integrated_web_system.perform_intelligent_search(
                query, 
                context={'general': True}
            )
            
            if result.get('success'):
                print(f"✅ 검색 완료: {result.get('total_results', 0)}개 결과")
                print(f"🔧 최적화된 쿼리: '{result.get('optimized_query', query)}'")
                print(f"🌐 사용된 검색 엔진: {', '.join(result.get('search_engines_used', []))}")
                
                # 결과 표시
                results = result.get('results', [])
                for i, res in enumerate(results[:5], 1):  # 상위 5개만 표시
                    print(f"\n{i}. {res.get('title', 'N/A')}")
                    print(f"   URL: {res.get('url', 'N/A')}")
                    print(f"   스니펫: {res.get('snippet', 'N/A')[:100]}...")
                    print(f"   관련성: {res.get('relevance_score', 0.0):.3f}")
                    print(f"   컨텍스트 관련성: {res.get('context_relevance', 0.0):.3f}")
            else:
                print(f"❌ 검색 실패: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
    
    async def get_weather(self, city: str = "Seoul"):
        """날씨 정보 조회"""
        try:
            print(f"🌤️ {city} 날씨 정보 조회 중...")
            
            response = await self.integrated_web_system.web_search_system.get_weather(city)
            
            if response.success:
                weather_data = response.data
                print(f"✅ {city} 날씨 정보:")
                print(f"   온도: {weather_data.get('main', {}).get('temp', 'N/A')}°C")
                print(f"   습도: {weather_data.get('main', {}).get('humidity', 'N/A')}%")
                print(f"   날씨: {weather_data.get('weather', [{}])[0].get('description', 'N/A')}")
                print(f"   응답 시간: {response.response_time:.3f}초")
            else:
                print(f"❌ 날씨 정보 조회 실패: {response.error_message}")
                
        except Exception as e:
            print(f"❌ 날씨 조회 오류: {e}")
    
    async def get_news(self, query: str = None):
        """뉴스 정보 조회"""
        try:
            if query:
                print(f"📰 '{query}' 관련 뉴스 조회 중...")
            else:
                print("📰 최신 뉴스 조회 중...")
            
            response = await self.integrated_web_system.web_search_system.get_news(query=query)
            
            if response.success:
                news_data = response.data
                articles = news_data.get('articles', [])
                print(f"✅ 뉴스 조회 완료: {len(articles)}개 기사")
                
                for i, article in enumerate(articles[:3], 1):  # 상위 3개만 표시
                    print(f"\n{i}. {article.get('title', 'N/A')}")
                    print(f"   출처: {article.get('source', {}).get('name', 'N/A')}")
                    print(f"   설명: {article.get('description', 'N/A')[:100]}...")
                    print(f"   URL: {article.get('url', 'N/A')}")
            else:
                print(f"❌ 뉴스 조회 실패: {response.error_message}")
                
        except Exception as e:
            print(f"❌ 뉴스 조회 오류: {e}")
    
    async def translate_text(self, text: str, target_lang: str = "en"):
        """텍스트 번역"""
        try:
            print(f"🌍 텍스트 번역 중: '{text}' → {target_lang}")
            
            response = await self.integrated_web_system.web_search_system.translate_text(
                text, target_lang, "ko"
            )
            
            if response.success:
                translation_data = response.data
                translated_text = translation_data.get('data', {}).get('translations', [{}])[0].get('translatedText', 'N/A')
                print(f"✅ 번역 완료: '{translated_text}'")
                print(f"   응답 시간: {response.response_time:.3f}초")
            else:
                print(f"❌ 번역 실패: {response.error_message}")
                
        except Exception as e:
            print(f"❌ 번역 오류: {e}")
    
    async def calculate_expression(self, expression: str):
        """수학 표현식 계산"""
        try:
            print(f"🧮 계산 중: {expression}")
            
            response = await self.integrated_web_system.web_search_system.calculate_expression(expression)
            
            if response.success:
                result = response.data.get('result', 'N/A')
                print(f"✅ 계산 완료: {result}")
                print(f"   응답 시간: {response.response_time:.3f}초")
            else:
                print(f"❌ 계산 실패: {response.error_message}")
                
        except Exception as e:
            print(f"❌ 계산 오류: {e}")
    
    async def get_comprehensive_info(self, query: str):
        """종합 정보 조회"""
        try:
            print(f"🔍 종합 정보 조회 중: '{query}'")
            
            result = await self.integrated_web_system.get_comprehensive_information(
                query, 
                ['search', 'news', 'weather']
            )
            
            if 'error' not in result:
                print("✅ 종합 정보 조회 완료:")
                
                # 웹 검색 결과
                web_search = result.get('sources', {}).get('web_search', {})
                if web_search.get('success'):
                    print(f"   🔍 웹 검색: {web_search.get('total_results', 0)}개 결과")
                
                # 뉴스 결과
                news = result.get('sources', {}).get('news', {})
                if news.get('success'):
                    articles = news.get('data', {}).get('articles', [])
                    print(f"   📰 뉴스: {len(articles)}개 기사")
                
                # 날씨 결과
                weather = result.get('sources', {}).get('weather', {})
                if weather.get('success'):
                    print(f"   🌤️ 날씨: 정보 획득")
                
            else:
                print(f"❌ 종합 정보 조회 실패: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ 종합 정보 조회 오류: {e}")
    
    def generate_dashboard(self, filepath: str = None):
        """웹 대시보드 생성"""
        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"data/web_integration/web_dashboard_{timestamp}.html"
            
            self.integrated_web_system.generate_web_dashboard(filepath)
            print(f"✅ 웹 대시보드 생성 완료: {filepath}")
            
        except Exception as e:
            print(f"❌ 웹 대시보드 생성 실패: {e}")
    
    def export_data(self, filepath: str = None):
        """웹 데이터 내보내기"""
        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"data/web_integration/web_data_{timestamp}.json"
            
            self.integrated_web_system.export_web_data(filepath)
            print(f"✅ 웹 데이터 내보내기 완료: {filepath}")
            
        except Exception as e:
            print(f"❌ 웹 데이터 내보내기 실패: {e}")
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            self.integrated_web_system.web_search_system.clear_cache()
            print("✅ 캐시 정리 완료")
            
        except Exception as e:
            print(f"❌ 캐시 정리 실패: {e}")
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n🌐 하린코어 웹 검색 대화형 모드")
        print("명령어를 입력하세요 (help로 도움말 확인)")
        
        while True:
            try:
                command = input("\n🌐 명령어 > ").strip().lower()
                
                if command in ["quit", "exit", "q"]:
                    if self.running:
                        self.stop_web_system()
                    print("👋 웹 검색 CLI를 종료합니다.")
                    break
                
                elif command in ["help", "h"]:
                    self.show_help()
                
                elif command in ["start", "s"]:
                    self.start_web_system()
                
                elif command in ["stop", "st"]:
                    self.stop_web_system()
                
                elif command in ["status", "st"]:
                    self.get_status()
                
                elif command.startswith("search "):
                    query = command[7:]  # "search " 제거
                    asyncio.run(self.search(query))
                
                elif command.startswith("weather "):
                    city = command[8:]  # "weather " 제거
                    asyncio.run(self.get_weather(city))
                
                elif command.startswith("news "):
                    query = command[5:]  # "news " 제거
                    asyncio.run(self.get_news(query))
                
                elif command.startswith("translate "):
                    text = command[10:]  # "translate " 제거
                    asyncio.run(self.translate_text(text))
                
                elif command.startswith("calculate "):
                    expression = command[10:]  # "calculate " 제거
                    asyncio.run(self.calculate_expression(expression))
                
                elif command.startswith("comprehensive "):
                    query = command[14:]  # "comprehensive " 제거
                    asyncio.run(self.get_comprehensive_info(query))
                
                elif command in ["dashboard", "d"]:
                    self.generate_dashboard()
                
                elif command in ["export", "e"]:
                    self.export_data()
                
                elif command in ["clear", "c"]:
                    self.clear_cache()
                
                else:
                    print("❓ 알 수 없는 명령어입니다. 'help'를 입력하여 도움말을 확인하세요.")
                
            except KeyboardInterrupt:
                print("\n👋 웹 검색 CLI를 종료합니다.")
                if self.running:
                    self.stop_web_system()
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
    
    def show_help(self):
        """도움말 표시"""
        help_text = """
🌐 하린코어 웹 검색 CLI 도움말

기본 명령어:
  start, s                    - 웹 검색 시스템 시작
  stop, st                    - 웹 검색 시스템 중지
  status, st                  - 현재 상태 조회
  dashboard, d                - 웹 대시보드 생성
  export, e                   - 웹 데이터 내보내기
  clear, c                    - 캐시 정리
  help, h                     - 이 도움말 표시
  quit, exit, q               - 종료

검색 및 API 명령어:
  search <쿼리>               - 웹 검색 수행
  weather <도시>              - 날씨 정보 조회
  news <키워드>               - 뉴스 정보 조회
  translate <텍스트>          - 텍스트 번역
  calculate <표현식>          - 수학 계산
  comprehensive <쿼리>        - 종합 정보 조회

예시:
  > start                     # 웹 시스템 시작
  > search 인공지능           # 인공지능 검색
  > weather Seoul             # 서울 날씨 조회
  > news AI                   # AI 관련 뉴스
  > translate 안녕하세요      # 텍스트 번역
  > calculate 2+3*4           # 수학 계산
  > comprehensive 코로나      # 종합 정보 조회
  > dashboard                 # 대시보드 생성
  > quit                      # 종료
"""
        print(help_text)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="하린코어 웹 검색 CLI")
    parser.add_argument("command", nargs="?", choices=[
        "start", "stop", "status", "search", "weather", "news", "translate", 
        "calculate", "comprehensive", "dashboard", "export", "clear", "interactive"
    ], help="실행할 명령어")
    parser.add_argument("--query", help="검색 쿼리 또는 텍스트")
    parser.add_argument("--file", help="출력 파일 경로")
    
    args = parser.parse_args()
    
    cli = WebSearchCLI()
    
    if args.command == "start":
        cli.start_web_system()
    elif args.command == "stop":
        cli.stop_web_system()
    elif args.command == "status":
        cli.get_status()
    elif args.command == "search":
        if args.query:
            asyncio.run(cli.search(args.query))
        else:
            print("❌ 검색 쿼리를 입력해주세요.")
    elif args.command == "weather":
        city = args.query or "Seoul"
        asyncio.run(cli.get_weather(city))
    elif args.command == "news":
        asyncio.run(cli.get_news(args.query))
    elif args.command == "translate":
        if args.query:
            asyncio.run(cli.translate_text(args.query))
        else:
            print("❌ 번역할 텍스트를 입력해주세요.")
    elif args.command == "calculate":
        if args.query:
            asyncio.run(cli.calculate_expression(args.query))
        else:
            print("❌ 계산할 표현식을 입력해주세요.")
    elif args.command == "comprehensive":
        if args.query:
            asyncio.run(cli.get_comprehensive_info(args.query))
        else:
            print("❌ 조회할 쿼리를 입력해주세요.")
    elif args.command == "dashboard":
        cli.generate_dashboard(args.file)
    elif args.command == "export":
        cli.export_data(args.file)
    elif args.command == "clear":
        cli.clear_cache()
    elif args.command == "interactive":
        cli.interactive_mode()
    else:
        # 대화형 모드로 시작
        cli.interactive_mode()


if __name__ == "__main__":
    main() 