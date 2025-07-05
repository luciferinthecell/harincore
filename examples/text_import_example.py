"""
harin.examples.text_import_example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

텍스트 파일 임포트 및 기억 활용 예시
"""

from pathlib import Path
import json

from memory.text_importer import create_text_importer
from core.enhanced_main_loop import create_enhanced_harin


def create_sample_conversation_file():
    """샘플 대화 파일 생성"""
    sample_content = """사용자: 파이썬으로 웹 스크래핑을 어떻게 하는지 알려줘세요.

어시스턴트: 파이썬에서 웹 스크래핑을 하려면 주로 requests와 BeautifulSoup 라이브러리를 사용합니다. 
먼저 pip install requests beautifulsoup4로 설치하고, requests.get()으로 웹페이지를 가져온 후 
BeautifulSoup으로 HTML을 파싱하면 됩니다.

사용자: 예시 코드를 보여줄 수 있나요?

어시스턴트: 물론입니다! 간단한 예시 코드를 보여드리겠습니다:

import requests
from bs4 import BeautifulSoup

url = "https://example.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 제목 추출
title = soup.find('h1').text
print(title)

사용자: 에러 처리는 어떻게 하나요?

어시스턴트: 에러 처리는 try-except 구문을 사용합니다. 
requests.get()에서 ConnectionError나 Timeout이 발생할 수 있고, 
BeautifulSoup에서 None을 반환할 수도 있으므로 적절한 예외 처리가 필요합니다.

사용자: 감사합니다! 정말 도움이 되었어요.

어시스턴트: 천만에요! 웹 스크래핑에 대해 더 궁금한 점이 있으시면 언제든 물어보세요.
"""
    
    file_path = Path("sample_conversation.txt")
    file_path.write_text(sample_content, encoding='utf-8')
    return file_path


def example_text_import():
    """텍스트 파일 임포트 예시"""
    print("=== 텍스트 파일 임포트 예시 ===")
    
    # 샘플 파일 생성
    sample_file = create_sample_conversation_file()
    print(f"📄 샘플 파일 생성: {sample_file}")
    
    # Harin 시스템 초기화
    harin = create_enhanced_harin("example_memory.json")
    
    # 텍스트 파일 임포트
    print("\n📥 텍스트 파일 임포트 중...")
    result = harin.import_text_file(sample_file, "파이썬 웹 스크래핑 대화")
    
    if result["success"]:
        print(f"✅ 임포트 성공!")
        print(f"   세션 ID: {result['session_id']}")
        print(f"   세그먼트 수: {result['segments_imported']}")
        print(f"   주제: {', '.join(result['topics'])}")
    else:
        print(f"❌ 임포트 실패: {result['error']}")
        return
    
    # 메모리 통계 확인
    print("\n📊 메모리 통계:")
    stats = harin.get_memory_stats()
    print(f"   총 노드 수: {stats['total_nodes']}")
    print(f"   총 엣지 수: {stats['total_edges']}")
    print(f"   노드 타입: {stats['node_types']}")
    
    # 기억 검색 테스트
    print("\n🔍 기억 검색 테스트:")
    queries = [
        "파이썬 웹 스크래핑",
        "BeautifulSoup 사용법",
        "에러 처리 방법",
        "requests 라이브러리"
    ]
    
    for query in queries:
        print(f"\n검색어: '{query}'")
        memories = harin.search_memories(query, max_results=3)
        
        if memories:
            for i, memory in enumerate(memories, 1):
                print(f"  {i}. {memory['content'][:80]}... (관련도: {memory['relevance_score']:.2f})")
        else:
            print("  관련 기억 없음")
    
    # 사고루프 테스트
    print("\n🧠 사고루프 테스트:")
    test_queries = [
        "파이썬으로 웹 스크래핑할 때 주의사항은?",
        "BeautifulSoup 대신 다른 라이브러리는?",
        "웹 스크래핑의 법적 문제는?"
    ]
    
    for query in test_queries:
        print(f"\n질문: {query}")
        result = harin.run_session(query)
        print(f"답변: {result['output'][:200]}...")
        print(f"사용된 루프: {result['best_loop']}")
        print(f"사용된 기억 수: {result['used_memories']}")
    
    # 정리
    sample_file.unlink(missing_ok=True)
    print(f"\n🧹 샘플 파일 정리 완료")


def example_api_import():
    """API 데이터 임포트 예시"""
    print("\n=== API 데이터 임포트 예시 ===")
    
    # 샘플 API 데이터
    api_data = {
        "session_id": "api_session_123",
        "title": "API 대화 세션",
        "messages": [
            {
                "role": "user",
                "content": "머신러닝과 딥러닝의 차이점이 궁금해요.",
                "timestamp": "2024-01-15T10:30:00"
            },
            {
                "role": "assistant", 
                "content": "머신러닝은 데이터로부터 패턴을 학습하는 알고리즘의 총칭이고, 딥러닝은 신경망을 사용하는 머신러닝의 한 분야입니다. 딥러닝은 더 복잡한 패턴을 학습할 수 있지만 더 많은 데이터와 계산 자원이 필요합니다.",
                "timestamp": "2024-01-15T10:30:30"
            },
            {
                "role": "user",
                "content": "텐서플로우와 파이토치 중 어떤 것을 추천하나요?",
                "timestamp": "2024-01-15T10:31:00"
            },
            {
                "role": "assistant",
                "content": "텐서플로우는 프로덕션 환경에서 안정적이고, 파이토치는 연구와 실험에 유연합니다. 초보자라면 파이토치가 더 직관적일 수 있고, 대규모 서비스라면 텐서플로우가 적합할 수 있습니다.",
                "timestamp": "2024-01-15T10:31:30"
            }
        ],
        "meta": {
            "source": "chat_api",
            "user_id": "user123",
            "session_duration": 90
        }
    }
    
    # Harin 시스템 초기화
    harin = create_enhanced_harin("example_api_memory.json")
    
    # API 데이터 임포트
    print("📥 API 데이터 임포트 중...")
    result = harin.import_api_data(api_data)
    
    if result["success"]:
        print(f"✅ API 임포트 성공!")
        print(f"   세션 ID: {result['session_id']}")
        print(f"   세그먼트 수: {result['segments_imported']}")
        print(f"   주제: {', '.join(result['topics'])}")
    else:
        print(f"❌ API 임포트 실패: {result['error']}")
        return
    
    # 기억 검색 테스트
    print("\n🔍 API 기억 검색 테스트:")
    api_queries = [
        "머신러닝 딥러닝 차이",
        "텐서플로우 파이토치",
        "신경망 알고리즘"
    ]
    
    for query in api_queries:
        print(f"\n검색어: '{query}'")
        memories = harin.search_memories(query, max_results=2)
        
        if memories:
            for i, memory in enumerate(memories, 1):
                print(f"  {i}. {memory['content'][:80]}... (관련도: {memory['relevance_score']:.2f})")
        else:
            print("  관련 기억 없음")


def example_memory_analysis():
    """기억 분석 예시"""
    print("\n=== 기억 분석 예시 ===")
    
    harin = create_enhanced_harin("example_memory.json")
    
    # 메모리 통계
    stats = harin.get_memory_stats()
    print("📊 전체 메모리 통계:")
    print(f"   총 노드 수: {stats['total_nodes']}")
    print(f"   총 엣지 수: {stats['total_edges']}")
    print(f"   메모리 크기: {stats['memory_size_mb']:.2f} MB")
    
    # 주제별 분석
    print("\n🏷️  주제별 분석:")
    for topic, count in stats['topics'].items():
        percentage = (count / stats['total_nodes']) * 100
        print(f"   {topic}: {count}개 ({percentage:.1f}%)")
    
    # 노드 타입별 분석
    print("\n📁 노드 타입별 분석:")
    for node_type, count in stats['node_types'].items():
        percentage = (count / stats['total_nodes']) * 100
        print(f"   {node_type}: {count}개 ({percentage:.1f}%)")
    
    # 개념 연결 분석
    print("\n🔗 개념 연결 분석:")
    memory = harin.memory
    
    # 가장 많이 언급된 개념 찾기
    concept_counts = {}
    for node in memory.nodes.values():
        concepts = node.meta.get("concept_tags", [])
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
    
    top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("   가장 많이 언급된 개념:")
    for concept, count in top_concepts:
        print(f"     {concept}: {count}회")


if __name__ == "__main__":
    # 예시 실행
    example_text_import()
    example_api_import()
    example_memory_analysis()
    
    print("\n✅ 모든 예시 실행 완료!")
    print("📝 생성된 파일들:")
    print("   - example_memory.json (텍스트 임포트 결과)")
    print("   - example_api_memory.json (API 임포트 결과)") 
