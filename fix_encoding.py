#!/usr/bin/env python3
"""
파일 인코딩 변환 스크립트 (UTF-16 -> UTF-8)
"""

import os

def fix_encoding(file_path):
    """파일 인코딩을 UTF-16에서 UTF-8로 변환"""
    try:
        # UTF-16으로 읽기
        with open(file_path, 'r', encoding='utf-16') as f:
            content = f.read()
        
        # 백업 파일 생성
        backup_path = file_path + '.utf16_backup'
        with open(backup_path, 'w', encoding='utf-16') as f:
            f.write(content)
        
        # UTF-8로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {file_path}: UTF-16 -> UTF-8 변환 완료")
        return True
        
    except Exception as e:
        print(f"❌ {file_path}: 오류 - {str(e)}")
        return False

def main():
    """메인 실행"""
    print("🔄 인코딩 변환 시작...")
    
    # 변환할 파일들
    files_to_fix = [
        'core/multi_intent_parser.py',
        'core/parallel_reasoning_unit.py'
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_encoding(file_path):
                fixed_count += 1
    
    print(f"\n📊 변환 완료: {fixed_count}개 파일")

if __name__ == "__main__":
    main() 
