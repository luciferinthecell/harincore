"""
harin.cli.memory_cli
~~~~~~~~~~~~~~~~~~~

메모리 관리 CLI 도구
- 텍스트 파일 임포트
- 기억 검색
- 메모리 상태 확인
- 기억 파일 관리
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

from memory.palantirgraph import PalantirGraph, initialize_memory
from memory.conscious_reader import ConsciousReader
from memory.memory_retriever import MemoryRetriever
from memory.text_importer import TextImporter


def cmd_import(args: argparse.Namespace):
    """텍스트 파일을 기억으로 임포트"""
    print(f"📖 텍스트 파일 임포트: {args.file}")
    
    # 메모리 그래프 초기화
    graph = PalantirGraph("memory/data/palantir_graph.json")
    
    # 텍스트 임포터 생성
    importer = TextImporter(graph)
    
    # 파일 임포트
    result = importer.import_text_file(args.file, args.topic)
    
    print(f"✅ 임포트 완료:")
    print(f"   - 세션 ID: {result.session_id}")
    print(f"   - 제목: {result.title}")
    print(f"   - 세그먼트 수: {len(result.segments)}개")
    print(f"   - 전체 주제: {', '.join(result.overall_topics)}")
    
    # 메모리 저장
    graph.save()
    print(f"💾 메모리 저장됨: {graph.persist_path}")


def cmd_search(args: argparse.Namespace):
    """기억 검색"""
    print(f"🔍 기억 검색: {args.query}")
    
    # 메모리 그래프 로드
    graph = PalantirGraph("memory/data/palantir_graph.json")
    
    # 메모리 검색기 생성
    retriever = MemoryRetriever(graph)
    
    # 검색 실행
    memories = retriever.retrieve_for_thinking(args.query)
    
    print(f"📋 검색 결과 ({len(memories)}개):")
    for i, memory in enumerate(memories, 1):
        print(f"\n{i}. {memory.retrieval_reason} (관련도: {memory.relevance_score:.2f})")
        print(f"   내용: {memory.node.content[:100]}...")
        if memory.node.meta.get("topic"):
            print(f"   주제: {memory.node.meta['topic']}")


def cmd_status(args: argparse.Namespace):
    """메모리 상태 확인"""
    print("📊 메모리 상태")
    
    # 메모리 그래프 로드
    graph = PalantirGraph("memory/data/palantir_graph.json")
    
    print(f"📈 노드 수: {len(graph.nodes)}개")
    print(f"🔗 관계 수: {len(graph.edges)}개")
    
    # 노드 타입별 통계
    node_types = {}
    for node in graph.nodes.values():
        node_type = node.node_type
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\n📋 노드 타입별 통계:")
    for node_type, count in node_types.items():
        print(f"   {node_type}: {count}개")
    
    # 주제별 통계
    topics = {}
    for node in graph.nodes.values():
        topic = node.meta.get("topic", "기타")
        topics[topic] = topics.get(topic, 0) + 1
    
    print("\n🏷️ 주제별 통계:")
    for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {topic}: {count}개")


def cmd_read(args: argparse.Namespace):
    """의식적 정독으로 텍스트 처리"""
    print(f"🧠 의식적 정독: {args.file}")
    
    # 메모리 그래프 로드
    graph = PalantirGraph("memory/data/palantir_graph.json")
    
    # 의식적 읽기 시스템 생성
    reader = ConsciousReader(graph)
    
    # 파일 읽기
    with open(args.file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 의식적 정독 실행
    result = reader.read_consciously(text)
    
    print(f"✅ 정독 완료:")
    print(f"   - 상황 기억: {len(result.situational_memories)}개")
    print(f"   - 감정 변화: {len(result.emotional_journey)}개")
    print(f"   - 인지 통찰: {len(result.cognitive_insights)}개")
    print(f"   - 개인 진화: {len(result.personal_evolution)}개")
    print(f"   - 읽기 품질: {result.reading_quality:.2f}")
    
    # 메모리 저장
    graph.save()


def cmd_load(args: argparse.Namespace):
    """기존 메모리 파일들을 로드"""
    print("🔄 기존 메모리 파일 로드 중...")
    
    # 메모리 그래프 초기화
    graph = PalantirGraph("memory/data/palantir_graph.json")
    
    # 기존 메모리 데이터 로드
    memory_files = [
        "memory/data/harin_v6_formatted_memory.jsonl",
        "memory/data/harin_v6_summary_nodes.jsonl"
    ]
    
    edge_files = [
        "memory/data/harin_v6_memory_edges.jsonl",
        "memory/data/harin_v6_summary_edges.jsonl"
    ]
    
    loaded_nodes = 0
    loaded_edges = 0
    
    for file_path in memory_files:
        if Path(file_path).exists():
            try:
                print(f"   📄 로드 중: {file_path}")
                graph.load_jsonl(file_path)
                loaded_nodes += 1
            except Exception as e:
                print(f"   ⚠️ 경고: {file_path} 로드 실패 - {e}")
    
    for file_path in edge_files:
        if Path(file_path).exists():
            try:
                print(f"   🔗 로드 중: {file_path}")
                graph.load_edges(file_path)
                loaded_edges += 1
            except Exception as e:
                print(f"   ⚠️ 경고: {file_path} 로드 실패 - {e}")
    
    # 메모리 저장
    graph.save()
    
    print(f"✅ 로드 완료:")
    print(f"   - 노드 파일: {loaded_nodes}개")
    print(f"   - 엣지 파일: {loaded_edges}개")
    print(f"   - 총 노드: {len(graph.nodes)}개")
    print(f"   - 총 엣지: {len(graph.edges)}개")


def cmd_list_files(args: argparse.Namespace):
    """사용 가능한 메모리 파일 목록"""
    print("📁 사용 가능한 메모리 파일:")
    
    data_dir = Path("memory/data")
    if not data_dir.exists():
        print("   메모리 데이터 디렉토리가 없습니다.")
        return
    
    # JSONL 파일들
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if jsonl_files:
        print("\n📄 JSONL 파일:")
        for file in jsonl_files:
            size = file.stat().st_size
            print(f"   {file.name} ({size:,} bytes)")
    
    # JSON 파일들
    json_files = list(data_dir.glob("*.json"))
    if json_files:
        print("\n📋 JSON 파일:")
        for file in json_files:
            size = file.stat().st_size
            print(f"   {file.name} ({size:,} bytes)")
    
    # TXT 파일들
    txt_files = list(data_dir.glob("*.txt"))
    if txt_files:
        print("\n📝 TXT 파일:")
        for file in txt_files:
            size = file.stat().st_size
            print(f"   {file.name} ({size:,} bytes)")


def build_parser() -> argparse.ArgumentParser:
    """CLI 파서 구성"""
    parser = argparse.ArgumentParser(prog="memory_cli", description="Harin 메모리 관리 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # import 명령
    import_parser = subparsers.add_parser("import", help="텍스트 파일을 기억으로 임포트")
    import_parser.add_argument("file", help="임포트할 텍스트 파일 경로")
    import_parser.add_argument("--topic", help="주제 태그")
    import_parser.set_defaults(func=cmd_import)
    
    # search 명령
    search_parser = subparsers.add_parser("search", help="기억 검색")
    search_parser.add_argument("query", help="검색 쿼리")
    search_parser.set_defaults(func=cmd_search)
    
    # status 명령
    status_parser = subparsers.add_parser("status", help="메모리 상태 확인")
    status_parser.set_defaults(func=cmd_status)
    
    # read 명령
    read_parser = subparsers.add_parser("read", help="의식적 정독으로 텍스트 처리")
    read_parser.add_argument("file", help="정독할 텍스트 파일 경로")
    read_parser.set_defaults(func=cmd_read)
    
    # load 명령
    load_parser = subparsers.add_parser("load", help="기존 메모리 파일들을 로드")
    load_parser.set_defaults(func=cmd_load)
    
    # list 명령
    list_parser = subparsers.add_parser("list", help="사용 가능한 메모리 파일 목록")
    list_parser.set_defaults(func=cmd_list_files)
    
    return parser


def main(argv: List[str] | None = None):
    """메인 함수"""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help(sys.stderr)


if __name__ == "__main__":
    main() 
