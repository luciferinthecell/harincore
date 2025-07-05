"""
하린코어 메모리 파일 마이그레이션 스크립트
===========================================

기존 파일들을 새로운 저장 구조로 체계적으로 정리하고 매핑
"""

import json
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import hashlib


class MemoryFileMigration:
    """메모리 파일 마이그레이션 관리자"""
    
    def __init__(self):
        self.base_path = Path("memory/data")
        self.old_data_path = Path("data/memory_data")
        self.old_intent_cache_path = Path("data/intent_cache")
        self.old_test_intent_cache_path = Path("data/test_intent_cache")
        
        # 새로운 디렉토리 구조
        self.new_structure = {
            "integrated": {
                "hot_memory.jsonl": [],
                "warm_memory.jsonl": [],
                "cold_memory.jsonl": [],
                "contextual_memory.jsonl": [],
                "vector_memory.jsonl": []
            },
            "palantir": {
                "main_graph.json": None,
                "multiverse_graph.json": None,
                "relationships.jsonl": []
            },
            "conductor": {
                "entities.jsonl": [],
                "relationships.jsonl": [],
                "contradictions.jsonl": [],
                "narratives.jsonl": []
            },
            "data_memory": {
                "h1.jsonl": [],
                "h2.jsonl": [],
                "h3.jsonl": [],
                "ha1.jsonl": [],
                "h.json": None
            },
            "cache": {
                "thought_cache.jsonl": [],
                "memory_cache.jsonl": [],
                "loop_cache.jsonl": [],
                "emotional_cache.jsonl": [],
                "cache_manager.json": None
            },
            "intent_cache": {
                "intent_analysis_*.json": [],
                "intent_cache_manager.json": None
            },
            "legacy": {
                "old_memory_data/": [],
                "old_intent_cache/": [],
                "migration_log.json": None
            }
        }
        
        # 마이그레이션 로그
        self.migration_log = {
            "migration_date": datetime.now().isoformat(),
            "files_migrated": [],
            "files_skipped": [],
            "errors": [],
            "statistics": {}
        }
    
    def analyze_existing_files(self) -> Dict[str, Any]:
        """기존 파일 분석"""
        print("🔍 기존 파일 분석 중...")
        
        analysis = {
            "memory_data": {},
            "intent_cache": {},
            "palantir": {},
            "total_files": 0,
            "total_size": 0,
            "file_types": {},
            "duplicates": []
        }
        
        # memory_data 분석
        if self.old_data_path.exists():
            for file_path in self.old_data_path.iterdir():
                if file_path.is_file():
                    file_info = self._analyze_file(file_path)
                    analysis["memory_data"][file_path.name] = file_info
                    analysis["total_files"] += 1
                    analysis["total_size"] += file_info["size_bytes"]
                    
                    # 파일 타입 통계
                    file_type = file_info["file_type"]
                    if file_type not in analysis["file_types"]:
                        analysis["file_types"][file_type] = 0
                    analysis["file_types"][file_type] += 1
        
        # intent_cache 분석
        if self.old_intent_cache_path.exists():
            for file_path in self.old_intent_cache_path.iterdir():
                if file_path.is_file():
                    file_info = self._analyze_file(file_path)
                    analysis["intent_cache"][file_path.name] = file_info
                    analysis["total_files"] += 1
                    analysis["total_size"] += file_info["size_bytes"]
        
        # test_intent_cache 분석
        if self.old_test_intent_cache_path.exists():
            for file_path in self.old_test_intent_cache_path.iterdir():
                if file_path.is_file():
                    file_info = self._analyze_file(file_path)
                    analysis["intent_cache"][file_path.name] = file_info
                    analysis["total_files"] += 1
                    analysis["total_size"] += file_info["size_bytes"]
        
        # palantir 분석
        palantir_file = self.base_path / "palantir_graph.json"
        if palantir_file.exists():
            file_info = self._analyze_file(palantir_file)
            analysis["palantir"]["palantir_graph.json"] = file_info
            analysis["total_files"] += 1
            analysis["total_size"] += file_info["size_bytes"]
        
        # 중복 파일 검사
        analysis["duplicates"] = self._find_duplicates(analysis)
        
        print(f"✅ 분석 완료: {analysis['total_files']}개 파일, {analysis['total_size'] / 1024:.1f}KB")
        return analysis
    
    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """개별 파일 분석"""
        stat = file_path.stat()
        
        file_info = {
            "path": str(file_path),
            "size_bytes": stat.st_size,
            "size_kb": stat.st_size / 1024,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_type": file_path.suffix,
            "content_type": self._determine_content_type(file_path),
            "memory_count": self._count_memories_in_file(file_path),
            "file_hash": self._calculate_file_hash(file_path)
        }
        
        return file_info
    
    def _determine_content_type(self, file_path: Path) -> str:
        """파일 내용 타입 결정"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if 'nodes' in data and 'edges' in data:
                            return 'graph'
                        elif 'memories' in data:
                            return 'memory_collection'
                        elif 'cache' in data:
                            return 'cache'
                        else:
                            return 'general_json'
                    elif isinstance(data, list):
                        return 'memory_list'
                    else:
                        return 'unknown'
            elif file_path.suffix == '.jsonl':
                return 'memory_stream'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'
    
    def _count_memories_in_file(self, file_path: Path) -> int:
        """파일 내 메모리 개수 계산"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
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
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def _find_duplicates(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """중복 파일 찾기"""
        duplicates = []
        file_hashes = {}
        
        for category in ['memory_data', 'intent_cache', 'palantir']:
            for filename, file_info in analysis.get(category, {}).items():
                file_hash = file_info.get('file_hash', '')
                if file_hash:
                    if file_hash in file_hashes:
                        duplicates.append({
                            'original': file_hashes[file_hash],
                            'duplicate': {'category': category, 'filename': filename, 'info': file_info}
                        })
                    else:
                        file_hashes[file_hash] = {'category': category, 'filename': filename, 'info': file_info}
        
        return duplicates
    
    def create_migration_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """마이그레이션 계획 생성"""
        print("📋 마이그레이션 계획 생성 중...")
        
        plan = {
            "migration_steps": [],
            "file_mappings": {},
            "backup_plan": {},
            "cleanup_plan": {},
            "estimated_time": "5-10분"
        }
        
        # 1단계: 백업 생성
        plan["migration_steps"].append({
            "step": 1,
            "action": "backup_existing_files",
            "description": "기존 파일들 백업",
            "files": list(analysis.get("memory_data", {}).keys()) + 
                    list(analysis.get("intent_cache", {}).keys()) +
                    list(analysis.get("palantir", {}).keys())
        })
        
        # 2단계: 새 디렉토리 구조 생성
        plan["migration_steps"].append({
            "step": 2,
            "action": "create_new_structure",
            "description": "새로운 디렉토리 구조 생성",
            "directories": list(self.new_structure.keys())
        })
        
        # 3단계: 파일 매핑 및 이동
        file_mappings = self._create_file_mappings(analysis)
        plan["file_mappings"] = file_mappings
        
        for mapping in file_mappings:
            plan["migration_steps"].append({
                "step": 3,
                "action": "migrate_file",
                "description": f"파일 이동: {mapping['source']} → {mapping['destination']}",
                "source": mapping['source'],
                "destination": mapping['destination'],
                "category": mapping['category']
            })
        
        # 4단계: 중복 파일 정리
        if analysis.get("duplicates"):
            plan["migration_steps"].append({
                "step": 4,
                "action": "cleanup_duplicates",
                "description": "중복 파일 정리",
                "duplicates": analysis["duplicates"]
            })
        
        # 5단계: 검증 및 로그 생성
        plan["migration_steps"].append({
            "step": 5,
            "action": "verify_migration",
            "description": "마이그레이션 검증 및 로그 생성"
        })
        
        print(f"✅ 마이그레이션 계획 생성 완료: {len(plan['migration_steps'])}단계")
        return plan
    
    def _create_file_mappings(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """파일 매핑 생성"""
        mappings = []
        
        # memory_data 파일 매핑
        memory_data_files = analysis.get("memory_data", {})
        
        # Palantir 관련
        if "palantir_graph.json" in memory_data_files:
            mappings.append({
                "source": "memory/data/palantir_graph.json",
                "destination": "memory/data/palantir/main_graph.json",
                "category": "palantir",
                "action": "move"
            })
        
        # MemoryConductor 관련
        conductor_mappings = {
            "entities.jsonl": "conductor/entities.jsonl",
            "relationships.jsonl": "conductor/relationships.jsonl", 
            "contradictions.jsonl": "conductor/contradictions.jsonl",
            "narrative_memories.jsonl": "conductor/narratives.jsonl"
        }
        
        for old_name, new_path in conductor_mappings.items():
            if old_name in memory_data_files:
                mappings.append({
                    "source": f"data/memory_data/{old_name}",
                    "destination": f"memory/data/{new_path}",
                    "category": "conductor",
                    "action": "move"
                })
        
        # DataMemory 관련
        data_memory_mappings = {
            "h1.jsonl": "data_memory/h1.jsonl",
            "h2.jsonl": "data_memory/h2.jsonl", 
            "h3.jsonl": "data_memory/h3.jsonl",
            "ha1.jsonl": "data_memory/ha1.jsonl",
            "h.json": "data_memory/h.json"
        }
        
        for old_name, new_path in data_memory_mappings.items():
            if old_name in memory_data_files:
                mappings.append({
                    "source": f"data/memory_data/{old_name}",
                    "destination": f"memory/data/{new_path}",
                    "category": "data_memory",
                    "action": "move"
                })
        
        # Cache 관련
        cache_mappings = {
            "thought_cache.jsonl": "cache/thought_cache.jsonl",
            "memory_cache.jsonl": "cache/memory_cache.jsonl",
            "loop_cache.jsonl": "cache/loop_cache.jsonl", 
            "emotional_cache.jsonl": "cache/emotional_cache.jsonl",
            "cache_manager.json": "cache/cache_manager.json"
        }
        
        for old_name, new_path in cache_mappings.items():
            if old_name in memory_data_files:
                mappings.append({
                    "source": f"data/memory_data/{old_name}",
                    "destination": f"memory/data/{new_path}",
                    "category": "cache",
                    "action": "move"
                })
        
        # Intent Cache 관련
        intent_cache_files = analysis.get("intent_cache", {})
        for filename in intent_cache_files:
            if filename.startswith("intent_analysis_"):
                mappings.append({
                    "source": f"data/test_intent_cache/{filename}",
                    "destination": f"memory/data/intent_cache/{filename}",
                    "category": "intent_cache",
                    "action": "move"
                })
        
        return mappings
    
    def execute_migration(self, plan: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        """마이그레이션 실행"""
        print(f"🚀 마이그레이션 {'시뮬레이션' if dry_run else '실행'} 시작...")
        
        results = {
            "success": True,
            "migrated_files": [],
            "skipped_files": [],
            "errors": [],
            "backup_created": None,
            "execution_time": None
        }
        
        start_time = datetime.now()
        
        try:
            # 1단계: 백업 생성
            if not dry_run:
                backup_path = self._create_backup()
                results["backup_created"] = backup_path
                print(f"💾 백업 생성 완료: {backup_path}")
            
            # 2단계: 새 디렉토리 구조 생성
            if not dry_run:
                self._create_new_directories()
                print("📁 새 디렉토리 구조 생성 완료")
            
            # 3단계: 파일 이동
            for mapping in plan["file_mappings"]:
                try:
                    if not dry_run:
                        success = self._migrate_file(mapping)
                        if success:
                            results["migrated_files"].append(mapping)
                            print(f"✅ {mapping['source']} → {mapping['destination']}")
                        else:
                            results["skipped_files"].append(mapping)
                            print(f"⚠️ 건너뜀: {mapping['source']}")
                    else:
                        # 시뮬레이션 모드
                        results["migrated_files"].append(mapping)
                        print(f"🔍 시뮬레이션: {mapping['source']} → {mapping['destination']}")
                        
                except Exception as e:
                    error_msg = f"파일 이동 실패 {mapping['source']}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
            
            # 4단계: 중복 파일 정리
            if plan.get("migration_steps"):
                duplicate_step = next((step for step in plan["migration_steps"] 
                                     if step["action"] == "cleanup_duplicates"), None)
                if duplicate_step and not dry_run:
                    self._cleanup_duplicates(duplicate_step["duplicates"])
                    print("🧹 중복 파일 정리 완료")
            
            # 5단계: 마이그레이션 로그 생성
            if not dry_run:
                self._create_migration_log(results)
                print("📝 마이그레이션 로그 생성 완료")
            
            results["execution_time"] = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ 마이그레이션 {'시뮬레이션' if dry_run else '실행'} 완료")
            print(f"📊 결과: {len(results['migrated_files'])}개 이동, {len(results['skipped_files'])}개 건너뜀, {len(results['errors'])}개 오류")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"마이그레이션 실패: {str(e)}")
            print(f"❌ 마이그레이션 실패: {e}")
        
        return results
    
    def _create_backup(self) -> str:
        """백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.base_path / "legacy" / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 기존 파일들 백업
        if self.old_data_path.exists():
            shutil.copytree(self.old_data_path, backup_path / "memory_data")
        
        if self.old_intent_cache_path.exists():
            shutil.copytree(self.old_intent_cache_path, backup_path / "intent_cache")
        
        if self.old_test_intent_cache_path.exists():
            shutil.copytree(self.old_test_intent_cache_path, backup_path / "test_intent_cache")
        
        # palantir 파일 백업
        palantir_file = self.base_path / "palantir_graph.json"
        if palantir_file.exists():
            shutil.copy2(palantir_file, backup_path / "palantir_graph.json")
        
        return str(backup_path)
    
    def _create_new_directories(self):
        """새 디렉토리 구조 생성"""
        for category in self.new_structure.keys():
            category_path = self.base_path / category
            category_path.mkdir(parents=True, exist_ok=True)
    
    def _migrate_file(self, mapping: Dict[str, Any]) -> bool:
        """개별 파일 마이그레이션"""
        source_path = Path(mapping["source"])
        dest_path = self.base_path / mapping["destination"]
        
        if not source_path.exists():
            return False
        
        # 목적 디렉토리 생성
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 이동
        shutil.move(str(source_path), str(dest_path))
        
        # 마이그레이션 로그 업데이트
        self.migration_log["files_migrated"].append({
            "source": mapping["source"],
            "destination": mapping["destination"],
            "category": mapping["category"],
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def _cleanup_duplicates(self, duplicates: List[Dict[str, Any]]):
        """중복 파일 정리"""
        for duplicate_info in duplicates:
            duplicate = duplicate_info["duplicate"]
            category = duplicate["category"]
            filename = duplicate["filename"]
            
            if category == "memory_data":
                file_path = self.old_data_path / filename
            elif category == "intent_cache":
                file_path = self.old_intent_cache_path / filename
            else:
                continue
            
            if file_path.exists():
                # 중복 파일을 legacy로 이동
                legacy_path = self.base_path / "legacy" / "duplicates" / filename
                legacy_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(legacy_path))
                
                self.migration_log["files_skipped"].append({
                    "file": str(file_path),
                    "reason": "duplicate",
                    "timestamp": datetime.now().isoformat()
                })
    
    def _create_migration_log(self, results: Dict[str, Any]):
        """마이그레이션 로그 생성"""
        self.migration_log.update({
            "migration_results": results,
            "statistics": {
                "total_migrated": len(results["migrated_files"]),
                "total_skipped": len(results["skipped_files"]),
                "total_errors": len(results["errors"]),
                "execution_time_seconds": results["execution_time"]
            }
        })
        
        log_path = self.base_path / "legacy" / "migration_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.migration_log, f, ensure_ascii=False, indent=2)
    
    def verify_migration(self) -> Dict[str, Any]:
        """마이그레이션 검증"""
        print("🔍 마이그레이션 검증 중...")
        
        verification = {
            "success": True,
            "missing_files": [],
            "unexpected_files": [],
            "file_count_matches": True,
            "structure_correct": True
        }
        
        # 새 구조 검증
        for category, files in self.new_structure.items():
            category_path = self.base_path / category
            if not category_path.exists():
                verification["structure_correct"] = False
                verification["missing_files"].append(f"디렉토리 없음: {category}")
                continue
            
            for filename in files.keys():
                if filename.endswith('*'):
                    # 와일드카드 패턴
                    pattern = filename.replace('*', '')
                    matching_files = list(category_path.glob(f"*{pattern}*"))
                    if not matching_files:
                        verification["missing_files"].append(f"{category}/{filename}")
                else:
                    file_path = category_path / filename
                    if not file_path.exists():
                        verification["missing_files"].append(f"{category}/{filename}")
        
        # 기존 파일들이 모두 이동되었는지 확인
        if self.old_data_path.exists():
            remaining_files = list(self.old_data_path.iterdir())
            if remaining_files:
                verification["unexpected_files"].extend([str(f) for f in remaining_files])
        
        verification["success"] = (
            verification["structure_correct"] and
            len(verification["missing_files"]) == 0 and
            len(verification["unexpected_files"]) == 0
        )
        
        if verification["success"]:
            print("✅ 마이그레이션 검증 성공")
        else:
            print("⚠️ 마이그레이션 검증 실패")
            if verification["missing_files"]:
                print(f"   누락된 파일: {verification['missing_files']}")
            if verification["unexpected_files"]:
                print(f"   예상치 못한 파일: {verification['unexpected_files']}")
        
        return verification


def main():
    """메인 실행 함수"""
    print("🔄 하린코어 메모리 파일 마이그레이션 시작")
    print("=" * 50)
    
    migrator = MemoryFileMigration()
    
    # 1. 기존 파일 분석
    analysis = migrator.analyze_existing_files()
    
    # 2. 마이그레이션 계획 생성
    plan = migrator.create_migration_plan(analysis)
    
    # 3. 시뮬레이션 실행
    print("\n🔍 마이그레이션 시뮬레이션 실행...")
    simulation_results = migrator.execute_migration(plan, dry_run=True)
    
    # 4. 사용자 확인
    print(f"\n📊 시뮬레이션 결과:")
    print(f"   이동할 파일: {len(simulation_results['migrated_files'])}개")
    print(f"   건너뛸 파일: {len(simulation_results['skipped_files'])}개")
    print(f"   오류: {len(simulation_results['errors'])}개")
    
    if simulation_results['errors']:
        print(f"\n❌ 오류 발생:")
        for error in simulation_results['errors']:
            print(f"   - {error}")
        return
    
    # 5. 실제 마이그레이션 실행
    print(f"\n🚀 실제 마이그레이션을 실행하시겠습니까? (y/N): ", end="")
    user_input = input().strip().lower()
    
    if user_input == 'y':
        print("\n🔄 실제 마이그레이션 실행...")
        results = migrator.execute_migration(plan, dry_run=False)
        
        # 6. 검증
        verification = migrator.verify_migration()
        
        if verification["success"]:
            print("\n🎉 마이그레이션 완료!")
            print("새로운 메모리 저장 구조가 준비되었습니다.")
        else:
            print("\n⚠️ 마이그레이션에 문제가 있습니다.")
            print("legacy 폴더의 migration_log.json을 확인해주세요.")
    else:
        print("\n❌ 마이그레이션이 취소되었습니다.")


if __name__ == "__main__":
    main() 