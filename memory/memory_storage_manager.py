"""
하린코어 메모리 저장 관리자 (Memory Storage Manager)
====================================================

모든 메모리 파일들을 체계적으로 관리하고 저장하는 시스템

저장 구조:
├── memory/
│   ├── data/
│   │   ├── integrated/           # 통합 메모리 시스템
│   │   │   ├── hot_memory.jsonl
│   │   │   ├── warm_memory.jsonl
│   │   │   ├── cold_memory.jsonl
│   │   │   ├── contextual_memory.jsonl
│   │   │   └── vector_memory.jsonl
│   │   ├── palantir/            # Palantir 시스템
│   │   │   ├── main_graph.json
│   │   │   ├── multiverse_graph.json
│   │   │   └── relationships.jsonl
│   │   ├── conductor/           # MemoryConductor
│   │   │   ├── entities.jsonl
│   │   │   ├── relationships.jsonl
│   │   │   ├── contradictions.jsonl
│   │   │   └── narratives.jsonl
│   │   ├── data_memory/         # DataMemoryManager
│   │   │   ├── h1.jsonl
│   │   │   ├── h2.jsonl
│   │   │   ├── h3.jsonl
│   │   │   └── ha1.jsonl
│   │   ├── cache/               # 캐시 시스템
│   │   │   ├── thought_cache.jsonl
│   │   │   ├── memory_cache.jsonl
│   │   │   ├── loop_cache.jsonl
│   │   │   ├── emotional_cache.jsonl
│   │   │   └── cache_manager.json
│   │   ├── intent_cache/        # 의도 분석 캐시
│   │   │   ├── intent_analysis_*.json
│   │   │   └── intent_cache_manager.json
│   │   └── legacy/              # 기존 파일들 (마이그레이션용)
│   │       ├── old_palantir_graph.json
│   │       └── old_memory_data/
│   └── backups/                 # 백업 파일들
│       ├── daily/
│       ├── weekly/
│       └── monthly/
"""

import json
import shutil
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import uuid
import os

# ----------------------------------------------
# Optional LangChain / LangGraph integration
# ----------------------------------------------
try:
    from langchain.schema import Document
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END  # type: ignore
    from langgraph.prebuilt import ToolNode  # type: ignore
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


@dataclass
class StorageConfig:
    """저장 설정"""
    base_path: str = "memory/data"
    backup_enabled: bool = True
    compression_enabled: bool = True
    max_file_size_mb: int = 100
    auto_cleanup_days: int = 30
    backup_retention_days: int = 90


@dataclass
class FileInfo:
    """파일 정보"""
    path: Path
    size_bytes: int
    last_modified: datetime
    file_type: str
    memory_count: int = 0
    compression_ratio: float = 1.0


class MemoryStorageManager:
    """메모리 저장 관리자"""
    
    def __init__(self, config: Optional[StorageConfig] = None, v8_mode: bool = False):
        """메모리 저장 관리자 초기화"""
        self.config = config or StorageConfig()
        self.base_path = Path(self.config.base_path) if isinstance(self.config.base_path, str) else self.config.base_path
        self.file_registry: Dict[str, FileInfo] = {}
        self._load_file_registry()
        self.v8_mode = v8_mode
        if v8_mode:
            try:
                self.v8_router = V8MemoryStoreRouter()
            except Exception as e:
                print(f"V8MemoryStoreRouter 초기화 실패: {e}")
                self.v8_router = None
        else:
            self.v8_router = None
        
        # LangChain VectorStore 캐시
        self._vectorstore = None  # 생성 시 캐시해두고 재사용
    
    def _get_category_path(self, category: Optional[str]) -> Path:
        if not category:
            category = "unknown"
        return self.base_path / category
    
    def _get_subcategory_path(self, category: Optional[str], subcategory: Optional[str]) -> Path:
        if not category:
            category = "unknown"
        if not subcategory:
            subcategory = "general"
        return self.base_path / category / subcategory
    
    def _get_timestamped_filename(self, prefix: Optional[str], category: Optional[str] = "", subcategory: Optional[str] = "") -> str:
        if not prefix:
            prefix = "memory"
        if not category:
            category = "general"
        if not subcategory:
            subcategory = "data"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
    
    def _get_compression_path(self, original_path: Optional[Path]) -> Path:
        if not original_path:
            return self.base_path / "compressed.jsonl.gz"
        return original_path.with_suffix('.jsonl.gz')
    
    def _get_decompression_path(self, compressed_path: Optional[Path]) -> Path:
        if not compressed_path:
            return self.base_path / "decompressed.jsonl"
        return compressed_path.with_suffix('.jsonl')
    
    def _load_jsonl_file(self, file_path: Optional[Path]) -> List[Dict[str, Any]]:
        if not file_path or not hasattr(file_path, 'exists') or not file_path.exists():
            return []
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return [json.loads(line.strip()) for line in f if line.strip()]
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [json.loads(line.strip()) for line in f if line.strip()]
        except Exception as e:
            print(f"JSONL 로드 실패: {e}")
            return []
    
    def _load_json_file(self, file_path: Optional[Path]) -> List[Dict[str, Any]]:
        if not file_path or not hasattr(file_path, 'exists') or not file_path.exists():
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [data] if isinstance(data, dict) else data
        except Exception as e:
            print(f"JSON 로드 실패: {e}")
            return []
    
    def _load_file_registry(self):
        """기존 파일 스캔"""
        print("🔍 기존 메모리 파일 스캔 중...")
        
        # 기존 파일들을 적절한 위치로 이동
        self._migrate_existing_files()
        
        # 모든 파일 정보 수집
        for file_path in self._get_category_path("").rglob("*"):
            if file_path.is_file():
                self._register_file(file_path)
        
        print(f"✅ {len(self.file_registry)}개 파일 등록 완료")
    
    def _migrate_existing_files(self):
        """기존 파일들을 새로운 구조로 마이그레이션"""
        migrations = [
            # Palantir 파일들
            ("memory/data/palantir_graph.json", "palantir/main_graph.json"),
            
            # MemoryConductor 파일들
            ("data/memory_data/entities.jsonl", "conductor/entities.jsonl"),
            ("data/memory_data/relationships.jsonl", "conductor/relationships.jsonl"),
            ("data/memory_data/contradictions.jsonl", "conductor/contradictions.jsonl"),
            ("data/memory_data/narrative_memories.jsonl", "conductor/narratives.jsonl"),
            
            # DataMemory 파일들
            ("data/memory_data/h1.jsonl", "data_memory/h1.jsonl"),
            ("data/memory_data/h2.jsonl", "data_memory/h2.jsonl"),
            ("data/memory_data/h3.jsonl", "data_memory/h3.jsonl"),
            ("data/memory_data/ha1.jsonl", "data_memory/ha1.jsonl"),
            
            # 캐시 파일들
            ("data/memory_data/thought_cache.jsonl", "cache/thought_cache.jsonl"),
            ("data/memory_data/memory_cache.jsonl", "cache/memory_cache.jsonl"),
            ("data/memory_data/loop_cache.jsonl", "cache/loop_cache.jsonl"),
            ("data/memory_data/emotional_cache.jsonl", "cache/emotional_cache.jsonl"),
            ("data/memory_data/cache_manager.json", "cache/cache_manager.json"),
            
            # 의도 캐시 파일들
            ("data/test_intent_cache/", "intent_cache/"),
        ]
        
        for source, destination in migrations:
            source_path = Path(source)
            dest_path = self._get_category_path(destination.split('/')[0]) / '/'.join(destination.split('/')[1:])
            
            if source_path.exists():
                try:
                    if source_path.is_dir():
                        # 디렉토리 복사
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.copytree(source_path, dest_path)
                    else:
                        # 파일 복사
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, dest_path)
                    
                    print(f"📁 {source} → {destination}")
                except Exception as e:
                    print(f"⚠️ 마이그레이션 실패 {source}: {e}")
    
    def _register_file(self, file_path: Path):
        """파일 등록"""
        try:
            stat = file_path.stat()
            file_info = FileInfo(
                path=file_path,
                size_bytes=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                file_type=self._determine_file_type(file_path),
                memory_count=self._count_memories_in_file(file_path)
            )
            
            relative_path = str(file_path.relative_to(self._get_category_path("")))
            self.file_registry[relative_path] = file_info
            
        except Exception as e:
            print(f"⚠️ 파일 등록 실패 {file_path}: {e}")
    
    def _determine_file_type(self, file_path: Path) -> str:
        """파일 타입 결정"""
        if file_path.suffix == '.json':
            return 'json'
        elif file_path.suffix == '.jsonl':
            return 'jsonl'
        elif file_path.suffix == '.gz':
            return 'compressed'
        else:
            return 'unknown'
    
    def _count_memories_in_file(self, file_path: Path) -> int:
        """파일 내 메모리 개수 계산"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # JSON 객체의 경우 키 개수나 배열 길이로 추정
                        if 'nodes' in data:
                            return len(data.get('nodes', []))
                        elif 'memories' in data:
                            return len(data.get('memories', []))
                        else:
                            return 1
                    elif isinstance(data, list):
                        return len(data)
                    else:
                        return 1
            elif file_path.suffix == '.jsonl':
                count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            count += 1
                return count
            else:
                return 0
        except Exception:
            return 0
    
    def save_memory(self, memory_data: Dict[str, Any], category: str, 
                   subcategory: str = None, filename: str = None, meta: Dict = None) -> str:
        """메모리 저장 (V8 옵션 지원)"""
        if self.v8_mode and self.v8_router is not None:
            # V8 구조의 노드/딕셔너리 모두 지원
            return self.v8_router.store_with_condition(memory_data, meta or {})
        # ... 기존 저장 로직 ...
        save_path = self._determine_save_path(category, subcategory, filename)
        return self._save_as_jsonl(memory_data, save_path)
    
    def _determine_save_path(self, category: str, subcategory: str = None, 
                           filename: str = None) -> Path:
        """저장 경로 결정"""
        
        # 카테고리별 기본 경로 매핑 (v7 기본 + v8 확장)
        category_paths = {
            # v7 기존 경로
            'integrated': 'integrated',
            'palantir': 'palantir',
            'conductor': 'conductor',
            'data_memory': 'data_memory',
            'cache': 'cache',
            'intent_cache': 'intent_cache',

            # v8 신규 레이어 경로
            'hot': 'hot',      # 실시간 세션 버퍼
            'warm': 'warm',    # 세션 단위 문맥 풀
            'cold': 'cold',    # 장기 보존 아카이브
            'graphs': 'graphs' # PalantirGraph 및 Universe 브랜치
        }
        
        base_category = category_paths.get(category, 'legacy')
        path_parts = [base_category]
        
        if subcategory:
            path_parts.append(subcategory)
        
        if filename:
            path_parts.append(filename)
        else:
            # 기본 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_parts.append(f"{category}_{timestamp}")
        
        return self._get_category_path('/'.join(path_parts))
    
    def _determine_file_format(self, category: str) -> str:
        """파일 형식 결정"""
        # 카테고리별 기본 저장 형식 정의
        format_map = {
            'integrated': 'jsonl',
            'palantir': 'json',
            'conductor': 'jsonl',
            'data_memory': 'jsonl',
            'cache': 'jsonl',
            'intent_cache': 'json',

            # v8 레이어별 형식
            'hot': 'jsonl',   # 다량 append → line 단위
            'warm': 'jsonl',  # 토픽 세션 그대로 append
            'cold': 'jsonl',  # 요약·Narrative 구조
            'graphs': 'json'  # PalantirGraph 구조
        }
        
        return format_map.get(category, 'jsonl')
    
    def _save_as_jsonl(self, memory_data: Dict[str, Any], save_path: Path) -> str:
        if not save_path:
            return ""
        save_path = save_path.with_suffix('.jsonl')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        should_compress = False
        if hasattr(self, 'config') and getattr(self.config, 'compression_enabled', False) and save_path.exists():
            if save_path.stat().st_size > 1024 * 1024:
                should_compress = True
                save_path = save_path.with_suffix('.jsonl.gz')
        try:
            if should_compress:
                with gzip.open(save_path, 'wt', encoding='utf-8') as f:
                    f.write(json.dumps(memory_data, ensure_ascii=False) + '\n')
            else:
                with open(save_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(memory_data, ensure_ascii=False) + '\n')
            self._register_file(save_path)
            return str(save_path)
        except Exception as e:
            print(f"JSONL 저장 실패: {e}")
            return ""
    
    def _save_as_json(self, memory_data: Dict[str, Any], save_path: Path) -> str:
        if not save_path:
            return ""
        save_path = save_path.with_suffix('.json')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            self._register_file(save_path)
            return str(save_path)
        except Exception as e:
            print(f"JSON 저장 실패: {e}")
            return ""
    
    def load_memory(self, file_path: str, memory_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """메모리 로드"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        try:
            if path.suffix == '.json':
                return self._load_json_file(path)
            elif path.suffix == '.jsonl':
                return self._load_jsonl_file(path)
            elif path.suffix == '.gz':
                return self._load_compressed(path, memory_id)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {path.suffix}")
                
        except Exception as e:
            raise Exception(f"메모리 로드 실패: {e}")
    
    def _load_compressed(self, path: Path, memory_id: str = None) -> List[Dict[str, Any]]:
        """압축 파일 로드"""
        memories = []
        
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    memory = json.loads(line)
                    if memory_id is None or memory.get('id') == memory_id:
                        memories.append(memory)
                        if memory_id:  # 특정 ID 찾았으면 중단
                            break
        
        return memories if memory_id is None else (memories[0] if memories else None)
    
    def search_memories(self, query: str, categories: List[str] = None, 
                       limit: int = 10) -> List[Dict[str, Any]]:
        """메모리 검색"""
        results = []
        
        # 검색할 카테고리 결정
        if categories is None:
            categories = ['integrated', 'palantir', 'conductor', 'data_memory']
        
        for category in categories:
            category_results = self._search_in_category(query, category, limit // len(categories))
            results.extend(category_results)
        
        # 관련성 순으로 정렬
        results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        return results[:limit]
    
    def _search_in_category(self, query: str, category: str, limit: int) -> List[Dict[str, Any]]:
        """카테고리 내 검색"""
        results = []
        
        # 카테고리 디렉토리 찾기
        category_path = self._get_category_path(category)
        if not category_path.exists():
            return results
        
        # 카테고리 내 모든 파일 검색
        for file_path in category_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.json', '.jsonl', '.gz']:
                try:
                    file_results = self._search_in_file(query, file_path, limit // 2)
                    results.extend(file_results)
                except Exception as e:
                    print(f"⚠️ 파일 검색 실패 {file_path}: {e}")
        
        return results[:limit]
    
    def _search_in_file(self, query: str, file_path: Path, limit: int) -> List[Dict[str, Any]]:
        """파일 내 검색"""
        results = []
        
        try:
            memories = self.load_memory(str(file_path))
            
            if isinstance(memories, list):
                for memory in memories:
                    relevance = self._calculate_relevance(query, memory)
                    if relevance > 0.3:  # 임계값
                        memory['relevance'] = relevance
                        memory['source_file'] = str(file_path)
                        results.append(memory)
                        
                        if len(results) >= limit:
                            break
            elif isinstance(memories, dict):
                relevance = self._calculate_relevance(query, memories)
                if relevance > 0.3:
                    memories['relevance'] = relevance
                    memories['source_file'] = str(file_path)
                    results.append(memories)
                    
        except Exception as e:
            print(f"⚠️ 파일 검색 오류 {file_path}: {e}")
        
        return results
    
    def _calculate_relevance(self, query: str, memory: Dict[str, Any]) -> float:
        """관련성 계산"""
        relevance = 0.0
        query_lower = query.lower()
        
        # 내용 매칭
        content = memory.get('content', '').lower()
        if query_lower in content:
            relevance += 0.5
        
        # 태그 매칭
        tags = memory.get('tags', [])
        for tag in tags:
            if query_lower in tag.lower():
                relevance += 0.3
        
        # 제목 매칭
        title = memory.get('title', '').lower()
        if query_lower in title:
            relevance += 0.4
        
        return min(1.0, relevance)
    
    def create_backup(self, backup_type: str = 'daily') -> str:
        """백업 생성"""
        if not self.config.backup_enabled:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{backup_type}_{timestamp}"
        backup_path = self._get_category_path("backups") / backup_type / backup_name
        
        try:
            # 전체 메모리 디렉토리 백업
            shutil.copytree(self._get_category_path("integrated"), backup_path / "integrated")
            shutil.copytree(self._get_category_path("palantir"), backup_path / "palantir")
            shutil.copytree(self._get_category_path("conductor"), backup_path / "conductor")
            shutil.copytree(self._get_category_path("data_memory"), backup_path / "data_memory")
            shutil.copytree(self._get_category_path("cache"), backup_path / "cache")
            
            # 백업 메타데이터 생성
            backup_meta = {
                "backup_type": backup_type,
                "created_at": datetime.now().isoformat(),
                "file_count": len(self.file_registry),
                "total_size": sum(fi.size_bytes for fi in self.file_registry.values())
            }
            
            with open(backup_path / "backup_meta.json", 'w', encoding='utf-8') as f:
                json.dump(backup_meta, f, ensure_ascii=False, indent=2)
            
            print(f"💾 백업 생성 완료: {backup_name}")
            return str(backup_path)
            
        except Exception as e:
            print(f"⚠️ 백업 생성 실패: {e}")
            return None
    
    def cleanup_old_files(self):
        """오래된 파일 정리"""
        cutoff_date = datetime.now() - timedelta(days=self.config.auto_cleanup_days)
        
        files_to_remove = []
        for relative_path, file_info in self.file_registry.items():
            if file_info.last_modified < cutoff_date:
                files_to_remove.append(relative_path)
        
        for relative_path in files_to_remove:
            file_path = self._get_category_path(relative_path)
            try:
                file_path.unlink()
                del self.file_registry[relative_path]
                print(f"🗑️ 오래된 파일 삭제: {relative_path}")
            except Exception as e:
                print(f"⚠️ 파일 삭제 실패 {relative_path}: {e}")
    
    def cleanup_old_backups(self):
        """오래된 백업 정리"""
        if not self.config.backup_enabled:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
        backup_base = self._get_category_path("backups")
        
        for backup_type in ['daily', 'weekly', 'monthly']:
            backup_dir = backup_base / backup_type
            if not backup_dir.exists():
                continue
            
            for backup_path in backup_dir.iterdir():
                if backup_path.is_dir():
                    # 백업 메타데이터 확인
                    meta_file = backup_path / "backup_meta.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file, 'r', encoding='utf-8') as f:
                                meta = json.load(f)
                                created_at = datetime.fromisoformat(meta['created_at'])
                                
                                if created_at < cutoff_date:
                                    shutil.rmtree(backup_path)
                                    print(f"🗑️ 오래된 백업 삭제: {backup_path.name}")
                        except Exception as e:
                            print(f"⚠️ 백업 메타데이터 읽기 실패 {meta_file}: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """저장소 통계"""
        total_size = sum(fi.size_bytes for fi in self.file_registry.values())
        total_memories = sum(fi.memory_count for fi in self.file_registry.values())
        
        stats = {
            "total_files": len(self.file_registry),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_memories": total_memories,
            "categories": {},
            "file_types": {},
            "largest_files": [],
            "recent_files": []
        }
        
        # 카테고리별 통계
        for relative_path, file_info in self.file_registry.items():
            category = relative_path.split('/')[0]
            if category not in stats["categories"]:
                stats["categories"][category] = {
                    "file_count": 0,
                    "total_size": 0,
                    "memory_count": 0
                }
            
            stats["categories"][category]["file_count"] += 1
            stats["categories"][category]["total_size"] += file_info.size_bytes
            stats["categories"][category]["memory_count"] += file_info.memory_count
        
        # 파일 타입별 통계
        for file_info in self.file_registry.values():
            file_type = file_info.file_type
            if file_type not in stats["file_types"]:
                stats["file_types"][file_type] = 0
            stats["file_types"][file_type] += 1
        
        # 가장 큰 파일들
        largest_files = sorted(self.file_registry.items(), 
                             key=lambda x: x[1].size_bytes, reverse=True)[:10]
        stats["largest_files"] = [
            {"path": path, "size_mb": fi.size_bytes / (1024 * 1024)}
            for path, fi in largest_files
        ]
        
        # 최근 파일들
        recent_files = sorted(self.file_registry.items(), 
                            key=lambda x: x[1].last_modified, reverse=True)[:10]
        stats["recent_files"] = [
            {"path": path, "modified": fi.last_modified.isoformat()}
            for path, fi in recent_files
        ]
        
        return stats
    
    def optimize_storage(self):
        """저장소 최적화"""
        print("🔧 저장소 최적화 시작...")
        
        # 1. 오래된 파일 정리
        self.cleanup_old_files()
        
        # 2. 오래된 백업 정리
        self.cleanup_old_backups()
        
        # 3. 큰 파일 압축
        self._compress_large_files()
        
        # 4. 중복 파일 검사
        self._deduplicate_files()
        
        # 5. 파일 정보 재스캔
        self._rescan_files()
        
        print("✅ 저장소 최적화 완료")
    
    def _compress_large_files(self):
        """큰 파일 압축"""
        for relative_path, file_info in self.file_registry.items():
            if (file_info.size_bytes > self.config.max_file_size_mb * 1024 * 1024 and 
                file_info.file_type in ['json', 'jsonl']):
                
                file_path = self._get_category_path(relative_path)
                compressed_path = self._get_compression_path(file_path)
                
                try:
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # 원본 파일 삭제
                    file_path.unlink()
                    
                    # 파일 정보 업데이트
                    self._register_file(compressed_path)
                    del self.file_registry[relative_path]
                    
                    print(f"🗜️ 파일 압축: {relative_path}")
                    
                except Exception as e:
                    print(f"⚠️ 파일 압축 실패 {relative_path}: {e}")
    
    def _deduplicate_files(self):
        """중복 파일 검사 및 정리"""
        # 간단한 중복 검사 (파일 크기와 내용 기반)
        file_hashes = {}
        
        for relative_path, file_info in self.file_registry.items():
            file_path = self._get_category_path(relative_path)
            
            try:
                # 간단한 해시 생성 (실제로는 더 정교한 해시 사용)
                with open(file_path, 'rb') as f:
                    content = f.read()
                    file_hash = hash(content)
                
                if file_hash in file_hashes:
                    # 중복 파일 발견
                    original_path = file_hashes[file_hash]
                    print(f"🔄 중복 파일 발견: {relative_path} → {original_path}")
                    
                    # 중복 파일 삭제
                    file_path.unlink()
                    del self.file_registry[relative_path]
                else:
                    file_hashes[file_hash] = relative_path
                    
            except Exception as e:
                print(f"⚠️ 중복 검사 실패 {relative_path}: {e}")
    
    def _rescan_files(self):
        """파일 정보 재스캔"""
        self.file_registry.clear()
        for file_path in self._get_category_path("").rglob("*"):
            if file_path.is_file():
                self._register_file(file_path)

    def some_function_returning_str(self, value: Optional[str]) -> str:
        if value is None:
            return ""
        return value

    def some_function_returning_list(self, value: Optional[list]) -> list:
        if value is None:
            return []
        return value

    # 아래와 같이 str 파라미터에 None이 들어올 수 있는 함수들 모두 Optional[str]로 변경
    def example_func(self, arg1: Optional[str], arg2: Optional[str] = None) -> str:
        if not arg1:
            return ""
        if not arg2:
            arg2 = "default"
        return f"{arg1}-{arg2}"

    # 반환 타입이 str인데 None이 반환될 수 있는 함수는 모두 빈 문자열 반환
    def another_func(self, arg: Optional[str]) -> str:
        if not arg:
            return ""
        return arg

    # 반환 타입이 List[str]인데 None이 반환될 수 있는 함수는 모두 빈 리스트 반환
    def another_list_func(self, arg: Optional[list]) -> list:
        if not arg:
            return []
        return arg

    def register_existing_files(self) -> int:
        """기존 파일들을 스캔하여 file_registry에 등록"""
        registered_count = 0
        
        try:
            # base_path가 존재하는지 확인
            if not self.base_path.exists():
                print(f"⚠️ 기본 경로가 존재하지 않음: {self.base_path}")
                return 0
            
            # 모든 하위 디렉토리와 파일을 재귀적으로 스캔
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.json', '.jsonl', '.jsonl.gz']:
                    try:
                        # 파일 정보 생성
                        file_info = FileInfo(
                            path=str(file_path),
                            size_bytes=file_path.stat().st_size,
                            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                            file_type=self._determine_file_type(file_path),
                            memory_count=self._count_memories_in_file(file_path)
                        )
                        
                        # file_registry에 등록
                        self.file_registry[str(file_path)] = file_info
                        registered_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ 파일 등록 실패 {file_path}: {e}")
            
            # file_registry 저장
            self._save_file_registry()
            
            print(f"✅ {registered_count}개 파일 등록 완료")
            return registered_count
            
        except Exception as e:
            print(f"❌ 파일 등록 중 오류: {e}")
            return registered_count
    
    def _extract_category_from_path(self, file_path: Path) -> str:
        """파일 경로에서 카테고리 추출"""
        try:
            # base_path를 제외한 상대 경로에서 첫 번째 디렉토리를 카테고리로 사용
            relative_path = file_path.relative_to(self.base_path)
            parts = relative_path.parts
            
            if len(parts) > 0:
                return parts[0]
            return "unknown"
        except:
            return "unknown"
    
    def _extract_subcategory_from_path(self, file_path: Path) -> str:
        """파일 경로에서 서브카테고리 추출"""
        try:
            # base_path를 제외한 상대 경로에서 두 번째 디렉토리를 서브카테고리로 사용
            relative_path = file_path.relative_to(self.base_path)
            parts = relative_path.parts
            
            if len(parts) > 1:
                return parts[1]
            return "general"
        except:
            return "general"

    # --------------------
    # V8 DATA-MEMORY HELPER
    # --------------------
    def _data_memory_paths(self) -> List[Path]:
        """h, h1, h2, h3, ha1 등 데이터 메모리 파일 경로 반환"""
        candidates = [
            self.base_path / "data_memory" / fname for fname in (
                "h.jsonl", "h1.jsonl", "h2.jsonl", "h3.jsonl", "ha1.jsonl"
            )
        ]
        return [p for p in candidates if p.exists()]

    def _load_data_memory_entries(self) -> List[Dict[str, Any]]:
        """데이터 메모리(h, h1~3, ha1) 모든 노드 로드"""
        entries: List[Dict[str, Any]] = []
        for path in self._data_memory_paths():
            try:
                entries.extend(self._load_jsonl_file(path))
            except Exception as e:
                print(f"⚠️ 데이터 메모리 로드 실패 {path}: {e}")
        return entries

    # --- 유사도 계산 유틸 ---
    def _tokenize(self, text: str) -> set:
        import re
        return set(re.findall(r"[\w가-힣]+", text.lower()))

    def _jaccard_similarity(self, a: str, b: str) -> float:
        """간단한 Jaccard 유사도 (토큰 기반)"""
        set_a, set_b = self._tokenize(a), self._tokenize(b)
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    # --- 공개 API ---
    def find_similar_memories(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """query 텍스트와 유사한 h* 메모리 노드를 top_k 개 반환"""
        entries = self._load_data_memory_entries()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for entry in entries:
            content = entry.get("content") or ""
            score = self._jaccard_similarity(query, content)
            if score >= threshold:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def build_similarity_map(self, threshold: float = 0.4) -> Dict[str, List[str]]:
        """h* 메모리들 간 유사도 ≥ threshold 인 노드 매핑(id -> [similar_ids])"""
        entries = self._load_data_memory_entries()
        id_to_content = {e.get("id"): e.get("content", "") for e in entries if e.get("id")}
        ids = list(id_to_content.keys())
        mapping: Dict[str, List[str]] = {}
        for i, id_a in enumerate(ids):
            content_a = id_to_content[id_a]
            sims: List[str] = []
            for id_b in ids[i+1:]:
                content_b = id_to_content[id_b]
                if self._jaccard_similarity(content_a, content_b) >= threshold:
                    sims.append(id_b)
            if sims:
                mapping[id_a] = sims
        return mapping

    # ==================================================
    # LangChain VectorStore & LangGraph Tool integration
    # ==================================================

    # --- LangChain Helper ---
    def _to_document(self, entry: Dict[str, Any]):
        """h* entry → LangChain Document 변환 (LangChain 설치 시)"""
        if not LANGCHAIN_AVAILABLE:
            return None
        metadata = {k: entry.get(k) for k in ("id", "type", "tags", "context")}
        return Document(page_content=entry.get("content", ""), metadata=metadata)

    def build_vectorstore(self, force_rebuild: bool = False):
        """FAISS VectorStore를 생성/반환 (LangChain 필요)"""
        if not LANGCHAIN_AVAILABLE:
            print("⚠️ LangChain 라이브러리가 설치되지 않아 VectorStore를 생성할 수 없습니다.")
            return None
        if self._vectorstore is not None and not force_rebuild:
            return self._vectorstore

        docs = [self._to_document(e) for e in self._load_data_memory_entries()]
        docs = [d for d in docs if d is not None]
        if not docs:
            print("⚠️ 벡터스토어로 변환할 문서가 없습니다.")
            return None

        embeddings = OpenAIEmbeddings()
        self._vectorstore = FAISS.from_documents(docs, embeddings)
        return self._vectorstore

    def query_vectorstore(self, query: str, k: int = 5):
        """VectorStore 유사 문서 검색"""
        vs = self.build_vectorstore()
        if vs is None:
            return []
        try:
            return vs.similarity_search(query, k=k)
        except Exception as e:
            print(f"⚠️ VectorStore 검색 실패: {e}")
            return []

    # --- LangGraph Helper ---
    def _find_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        for entry in self._load_data_memory_entries():
            if entry.get("id") == entry_id:
                return entry
        return None

    def _entry_to_toolnode(self, entry: Dict[str, Any]):
        """h* entry를 LangGraph ToolNode 로 변환"""
        if not LANGGRAPH_AVAILABLE or entry is None:
            return None

        def _tool_fn(state):  # LangGraph state → dict 반환
            return {
                "memory": entry.get("content", ""),
                "tags": entry.get("tags", [])
            }

        return ToolNode(_tool_fn, name=f"memory_tool_{entry.get('id', '')[:6]}")

    def export_memory_to_langgraph(self, graph: "StateGraph", entry_ids: List[str]):  # type: ignore
        """지정 id 노드들을 LangGraph 그래프에 ToolNode 로 삽입"""
        if not LANGGRAPH_AVAILABLE:
            print("⚠️ LangGraph 라이브러리가 설치되지 않아 그래프에 메모리를 삽입할 수 없습니다.")
            return graph

        for eid in entry_ids:
            entry = self._find_entry_by_id(eid)
            tool_node = self._entry_to_toolnode(entry)
            if tool_node:
                try:
                    graph.add_node(tool_node)
                    graph.add_edge(tool_node, END)  # type: ignore
                except Exception as e:
                    print(f"⚠️ ToolNode 삽입 실패({eid}): {e}")

        # 메모리 세이버로 체크포인트 활성화
        try:
            MemorySaver(graph)
        except Exception as e:
            print(f"⚠️ MemorySaver 초기화 실패: {e}")
        return graph


# 사용 예시
def create_memory_storage_manager(config: StorageConfig = None) -> MemoryStorageManager:
    """메모리 저장 관리자 생성"""
    return MemoryStorageManager(config)


# 기본 설정
DEFAULT_STORAGE_CONFIG = StorageConfig(
    base_path="memory/data",
    backup_enabled=True,
    compression_enabled=True,
    max_file_size_mb=50,
    auto_cleanup_days=30,
    backup_retention_days=90
)

# === V8 저장/라우팅/통계 구조 통합 ===
from pathlib import Path as SysPath
from typing import Optional as TypingOptional, Dict as TypingDict
from datetime import datetime as dt_datetime
import json as sys_json
import os as sys_os

class V8MemoryStoreRouter:
    def __init__(self):
        self.paths = {
            "cold": {
                "h1": SysPath("data/memory_data/h1.jsonl"),
                "h2": SysPath("data/memory_data/h2.jsonl"),
                "h3": SysPath("data/memory_data/h3.jsonl"),
            },
            "hot": SysPath("data/memory_data/ha1.jsonl"),
            "warm": {
                "emotional": SysPath("data/memory_data/emotional_cache.jsonl"),
                "loop": SysPath("data/memory_data/loop_cache.jsonl"),
                "general": SysPath("data/memory_data/memory_cache.jsonl")
            }
        }
        self.stats = {
            "cold": 0,
            "hot": 0,
            "warm": 0,
            "errors": 0
        }
        sys_os.makedirs("data/memory_data", exist_ok=True)

    def store_with_condition(self, node, meta: TypingDict) -> TypingOptional[str]:
        try:
            if meta.get("user_acknowledged") is True:
                h_level = meta.get("target_h_level", "h1")
                path = self.paths["cold"].get(h_level, self.paths["cold"]["h1"])
                self.stats["cold"] += 1
                return self._append_line(path, node)

            elif not getattr(node, 'scar_flagged', False) and getattr(node, 'confidence', 0.0) >= 0.7:
                self.stats["hot"] += 1
                return self._append_line(self.paths["hot"], node)

            else:
                if getattr(node, 'emotion_vector', None) and sum(getattr(node, 'emotion_vector', [])) > 0.5:
                    self.stats["warm"] += 1
                    return self._append_line(self.paths["warm"]["emotional"], node)
                elif hasattr(node, 'tags') and "loop" in getattr(node, 'tags', []):
                    self.stats["warm"] += 1
                    return self._append_line(self.paths["warm"]["loop"], node)
                else:
                    self.stats["warm"] += 1
                    return self._append_line(self.paths["warm"]["general"], node)

        except Exception as e:
            self.stats["errors"] += 1
            self._log_error(node, e)
            return None

    def _append_line(self, path: SysPath, node) -> str:
        # node는 dict 또는 dataclass 모두 지원
        if hasattr(node, '__dict__'):
            line = sys_json.dumps(node.__dict__, ensure_ascii=False, default=str)
        else:
            line = sys_json.dumps(node, ensure_ascii=False, default=str)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
        return str(path)

    def _log_error(self, node, error: Exception):
        log_path = SysPath("data/memory_data/store_errors.log")
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{dt_datetime.utcnow()}] Error storing {getattr(node, 'node_id', 'unknown')}: {str(error)}\n")

    def report_stats(self) -> TypingDict:
        return self.stats 