"""
하린코어 통합 메모리 시스템 (Integrated Memory System)
=======================================================

모든 메모리 시스템을 통합하여 관리하는 중앙 시스템
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import uuid

# 기존 메모리 시스템들 import
try:
    from .palantir import PalantirGraph
    from .conductor import MemoryConductor
    from .data_memory import DataMemoryManager
    from .engine import MemoryEngine
    from .adapter import MemoryAdapter
    from .memory_storage_manager import MemoryStorageManager, StorageConfig
except ImportError:
    # 폴백 처리
    print("⚠️ 일부 메모리 모듈을 찾을 수 없습니다. 기본 기능으로 실행합니다.")
    PalantirGraph = None
    MemoryConductor = None
    DataMemoryManager = None
    MemoryEngine = None
    MemoryAdapter = None
    MemoryStorageManager = None
    StorageConfig = None


class IntegratedMemorySystem:
    """통합 메모리 시스템"""
    
    def __init__(self, storage_manager: Optional[MemoryStorageManager] = None):
        """통합 메모리 시스템 초기화"""
        self.storage_manager = storage_manager or MemoryStorageManager()
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.memory_systems: Dict[str, Any] = {}
        self.memory_state = {
            "total_memories": 0,
            "active_systems": 0,
            "last_sync": datetime.now().isoformat(),
            "health_status": "healthy"
        }
        self.routing_rules = {
            "hot": {"systems": ["cache", "engine"], "priority": "high"},
            "warm": {"systems": ["conductor", "data_memory"], "priority": "medium"},
            "cold": {"systems": ["palantir", "integrated"], "priority": "low"}
        }
        
        # 메모리 시스템 초기화
        self._initialize_memory_systems()
    
    def _initialize_memory_systems(self):
        """메모리 시스템들 초기화"""
        print("🔧 메모리 시스템 초기화 중...")
        
        # Palantir 시스템
        if PalantirGraph:
            try:
                self.memory_systems['palantir'] = PalantirGraph()
                print("✅ Palantir 시스템 초기화 완료")
            except Exception as e:
                print(f"⚠️ Palantir 시스템 초기화 실패: {e}")
        
        # MemoryConductor 시스템
        if MemoryConductor:
            try:
                self.memory_systems['conductor'] = MemoryConductor()
                print("✅ MemoryConductor 시스템 초기화 완료")
            except Exception as e:
                print(f"⚠️ MemoryConductor 시스템 초기화 실패: {e}")
        
        # DataMemory 시스템
        if DataMemoryManager:
            try:
                self.memory_systems['data_memory'] = DataMemoryManager()
                print("✅ DataMemory 시스템 초기화 완료")
            except Exception as e:
                print(f"⚠️ DataMemory 시스템 초기화 실패: {e}")
        
        # MemoryEngine 시스템
        if MemoryEngine:
            try:
                self.memory_systems['engine'] = MemoryEngine()
                print("✅ MemoryEngine 시스템 초기화 완료")
            except Exception as e:
                print(f"⚠️ MemoryEngine 시스템 초기화 실패: {e}")
        
        # MemoryAdapter 시스템
        if MemoryAdapter:
            try:
                self.memory_systems['adapter'] = MemoryAdapter()
                print("✅ MemoryAdapter 시스템 초기화 완료")
            except Exception as e:
                print(f"⚠️ MemoryAdapter 시스템 초기화 실패: {e}")
        
        print(f"✅ {len(self.memory_systems)}개 메모리 시스템 초기화 완료")
    
    def _create_routing_rules(self) -> Dict[str, Any]:
        """메모리 라우팅 규칙 생성"""
        return {
            "hot_memory": {
                "systems": ["palantir", "conductor"],
                "priority": "high",
                "max_age_hours": 1,
                "access_pattern": "frequent"
            },
            "warm_memory": {
                "systems": ["data_memory", "engine"],
                "priority": "medium",
                "max_age_hours": 24,
                "access_pattern": "moderate"
            },
            "cold_memory": {
                "systems": ["adapter"],
                "priority": "low",
                "max_age_hours": 168,  # 1주일
                "access_pattern": "rare"
            },
            "contextual_memory": {
                "systems": ["conductor", "palantir"],
                "priority": "high",
                "max_age_hours": 12,
                "access_pattern": "context_dependent"
            },
            "vector_memory": {
                "systems": ["engine", "adapter"],
                "priority": "medium",
                "max_age_hours": 72,
                "access_pattern": "semantic_search"
            }
        }
    
    def save_memory(self, memory_data: Dict[str, Any], category: str = "general", 
                   subcategory: str = "", memory_type: str = "hot") -> str:
        """메모리 저장"""
        try:
            # 메모리 ID 생성
            memory_id = str(uuid.uuid4())
            memory_data['id'] = memory_id
            memory_data['timestamp'] = datetime.now().isoformat()
            memory_data['category'] = category
            memory_data['subcategory'] = subcategory
            memory_data['type'] = memory_type
            
            # 저장 경로 결정
            if memory_type == "hot":
                save_path = self.storage_manager.config.base_path / "integrated" / "hot_memory"
            elif memory_type == "warm":
                save_path = self.storage_manager.config.base_path / "integrated" / "warm_memory"
            else:
                save_path = self.storage_manager.config.base_path / "integrated" / "cold_memory"
            
            # 서브카테고리가 있으면 하위 디렉토리 생성
            if subcategory:
                save_path = save_path / subcategory
            
            # 파일명 생성
            filename = f"integrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 저장 실행
            saved_path = self.storage_manager.save_memory(
                memory_data, 
                save_path, 
                filename,
                format_type="jsonl"
            )
            
            # 메모리 시스템에 등록
            self.memories[memory_id] = {
                'data': memory_data,
                'path': saved_path,
                'category': category,
                'subcategory': subcategory,
                'type': memory_type,
                'created_at': datetime.now().isoformat()
            }
            
            print(f"✅ 메모리 저장 완료: {memory_id}")
            return memory_id
            
        except Exception as e:
            print(f"❌ 메모리 저장 실패: {e}")
            return ""
    
    def load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """메모리 로드"""
        try:
            # 메모리 시스템에서 직접 로드
            if memory_id in self.memories:
                return self.memories[memory_id]['data']
            
            # 저장소에서 검색
            search_results = self.storage_manager.search_memories(
                query=memory_id,
                category="",
                limit=1
            )
            
            if search_results:
                memory_data = search_results[0]
                # 메모리 시스템에 등록
                self.memories[memory_id] = {
                    'data': memory_data,
                    'path': f"searched_{memory_id}",
                    'category': memory_data.get('category', 'unknown'),
                    'subcategory': memory_data.get('subcategory', ''),
                    'type': memory_data.get('type', 'unknown'),
                    'created_at': memory_data.get('timestamp', datetime.now().isoformat())
                }
                return memory_data
            
            print(f"⚠️ 메모리를 찾을 수 없음: {memory_id}")
            return None
            
        except Exception as e:
            print(f"❌ 메모리 로드 실패: {e}")
            return None
    
    def search_memories(self, query: str, memory_types: List[str] = None, 
                       limit: int = 10) -> List[Dict[str, Any]]:
        """메모리 검색"""
        try:
            all_results = []
            
            # 검색할 시스템들 결정
            if memory_types:
                target_systems = []
                for memory_type in memory_types:
                    if memory_type in self.routing_rules:
                        target_systems.extend(self.routing_rules[memory_type]["systems"])
                target_systems = list(set(target_systems))  # 중복 제거
            else:
                target_systems = list(self.memory_systems.keys())
            
            # 각 시스템에서 검색
            for system_name in target_systems:
                if system_name in self.memory_systems:
                    try:
                        if hasattr(self.memory_systems[system_name], 'search_memories'):
                            results = self.memory_systems[system_name].search_memories(query, limit // len(target_systems))
                            for result in results:
                                result['source_system'] = system_name
                            all_results.extend(results)
                        else:
                            # 기본 검색
                            if self.storage_manager:
                                results = self.storage_manager.search_memories(
                                    query, [system_name], limit // len(target_systems)
                                )
                                for result in results:
                                    result['source_system'] = system_name
                                all_results.extend(results)
                    except Exception as e:
                        print(f"⚠️ {system_name} 시스템 검색 실패: {e}")
            
            # 결과 정렬 및 중복 제거
            unique_results = self._deduplicate_results(all_results)
            sorted_results = sorted(unique_results, key=lambda x: x.get('relevance', 0), reverse=True)
            
            return sorted_results[:limit]
            
        except Exception as e:
            print(f"❌ 메모리 검색 실패: {e}")
            return []
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검색 결과 중복 제거"""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            memory_id = result.get('id')
            if memory_id and memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_results.append(result)
        
        return unique_results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계"""
        stats = {
            "total_memories": self.memory_state["total_memories"],
            "active_systems": list(self.memory_systems.keys()),
            "last_sync": self.memory_state["last_sync"],
            "system_stats": {}
        }
        
        # 저장소 통계 추가
        if self.storage_manager:
            stats["storage_stats"] = self.storage_manager.get_storage_stats()
        
        # 각 시스템별 통계
        for system_name, system in self.memory_systems.items():
            try:
                if hasattr(system, 'get_stats'):
                    stats["system_stats"][system_name] = system.get_stats()
                else:
                    stats["system_stats"][system_name] = {
                        "status": "active",
                        "memory_count": "unknown"
                    }
            except Exception as e:
                stats["system_stats"][system_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return stats
    
    def optimize_memory(self):
        """메모리 최적화"""
        print("🔧 메모리 최적화 시작...")
        
        # 저장소 최적화
        if self.storage_manager:
            self.storage_manager.optimize_storage()
        
        # 각 시스템별 최적화
        for system_name, system in self.memory_systems.items():
            try:
                if hasattr(system, 'optimize'):
                    system.optimize()
                    print(f"✅ {system_name} 시스템 최적화 완료")
            except Exception as e:
                print(f"⚠️ {system_name} 시스템 최적화 실패: {e}")
        
        # 메모리 상태 업데이트
        self.memory_state["last_sync"] = datetime.now().isoformat()
        
        print("✅ 메모리 최적화 완료")
    
    def create_backup(self, backup_type: str = 'full') -> str:
        """전체 메모리 백업"""
        print(f"💾 {backup_type} 백업 생성 중...")
        
        # 저장소 백업
        storage_backup = None
        if self.storage_manager:
            storage_backup = self.storage_manager.create_backup(backup_type)
        
        # 각 시스템별 백업
        system_backups = {}
        for system_name, system in self.memory_systems.items():
            try:
                if hasattr(system, 'create_backup'):
                    backup_path = system.create_backup(backup_type)
                    system_backups[system_name] = backup_path
            except Exception as e:
                print(f"⚠️ {system_name} 시스템 백업 실패: {e}")
        
        # 통합 백업 메타데이터
        backup_meta = {
            "backup_type": backup_type,
            "created_at": datetime.now().isoformat(),
            "storage_backup": storage_backup,
            "system_backups": system_backups,
            "memory_state": self.memory_state
        }
        
        # 백업 메타데이터 저장
        backup_meta_path = f"memory/data/backups/{backup_type}/backup_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(backup_meta_path).parent.mkdir(parents=True, exist_ok=True)
        with open(backup_meta_path, 'w', encoding='utf-8') as f:
            json.dump(backup_meta, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {backup_type} 백업 생성 완료")
        return backup_meta_path
    
    def sync_memory_systems(self):
        """메모리 시스템 동기화"""
        print("🔄 메모리 시스템 동기화 중...")
        
        # 각 시스템 간 데이터 동기화
        for system_name, system in self.memory_systems.items():
            try:
                if hasattr(system, 'sync'):
                    system.sync()
                    print(f"✅ {system_name} 시스템 동기화 완료")
            except Exception as e:
                print(f"⚠️ {system_name} 시스템 동기화 실패: {e}")
        
        # 메모리 상태 업데이트
        self.memory_state["last_sync"] = datetime.now().isoformat()
        
        print("✅ 메모리 시스템 동기화 완료")
    
    def get_memory_health(self) -> Dict[str, Any]:
        """메모리 시스템 건강 상태"""
        health = {
            "overall_status": "healthy",
            "systems": {},
            "issues": [],
            "recommendations": []
        }
        
        # 각 시스템 건강 상태 확인
        for system_name, system in self.memory_systems.items():
            try:
                if hasattr(system, 'get_health'):
                    system_health = system.get_health()
                    health["systems"][system_name] = system_health
                    
                    if system_health.get("status") != "healthy":
                        health["issues"].append(f"{system_name}: {system_health.get('message', 'Unknown issue')}")
                else:
                    health["systems"][system_name] = {
                        "status": "unknown",
                        "message": "Health check not implemented"
                    }
            except Exception as e:
                health["systems"][system_name] = {
                    "status": "error",
                    "message": str(e)
                }
                health["issues"].append(f"{system_name}: {str(e)}")
        
        # 전체 상태 결정
        if health["issues"]:
            health["overall_status"] = "warning" if len(health["issues"]) < 3 else "critical"
        
        # 권장사항 생성
        if health["overall_status"] != "healthy":
            health["recommendations"].append("메모리 시스템 최적화를 실행하세요")
            health["recommendations"].append("백업을 생성하세요")
        
        return health


# 사용 예시
def create_integrated_memory_system(config: Dict[str, Any] = None) -> IntegratedMemorySystem:
    """통합 메모리 시스템 생성"""
    return IntegratedMemorySystem(config)


# 기본 설정
DEFAULT_INTEGRATED_CONFIG = {
    "auto_sync": True,
    "sync_interval_hours": 1,
    "backup_enabled": True,
    "optimization_enabled": True,
    "health_check_enabled": True
} 