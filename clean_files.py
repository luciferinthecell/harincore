#!/usr/bin/env python3
"""
파일에서 null bytes 제거 스크립트
"""

import os

def clean_file(file_path):
    """파일에서 null bytes 제거"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        null_count = content.count(b'\x00')
        if null_count > 0:
            clean_content = content.replace(b'\x00', b'')
            
            # 백업 파일 생성
            backup_path = file_path + '.backup'
            with open(backup_path, 'wb') as f:
                f.write(content)
            
            # 정리된 파일 저장
            with open(file_path, 'wb') as f:
                f.write(clean_content)
            
            print(f"✅ {file_path}: {null_count}개 null bytes 제거됨")
            return True
        else:
            print(f"✅ {file_path}: null bytes 없음")
            return False
            
    except Exception as e:
        print(f"❌ {file_path}: 오류 - {str(e)}")
        return False

def main():
    """메인 실행"""
    print("🧹 파일 정리 시작...")
    
    # 정리할 파일들
    files_to_clean = [
        'core/multi_intent_parser.py',
        'core/parallel_reasoning_unit.py',
        'core/enhanced_main_loop.py'
    ]
    
    cleaned_count = 0
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            if clean_file(file_path):
                cleaned_count += 1
    
    print(f"\n📊 정리 완료: {cleaned_count}개 파일")

if __name__ == "__main__":
    main() 
