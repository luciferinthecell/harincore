"""
하린코어 모니터링 CLI - Monitoring Command Line Interface
모니터링 시스템 제어 및 대시보드 관리
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

from core.integrated_monitoring import IntegratedMonitoringSystem
from core.cognitive_cycle import CognitiveCycle
from core.emotion_system import EmotionSystem
from core.action_system import ActionSystem


class MonitoringCLI:
    """모니터링 CLI 클래스"""
    
    def __init__(self):
        self.integrated_monitoring = IntegratedMonitoringSystem()
        self.running = False
    
    def start_monitoring(self, config: Dict[str, Any] = None):
        """모니터링 시작"""
        try:
            # 하린코어 시스템들 초기화 (실제 환경에서는 기존 인스턴스 사용)
            cognitive_cycle = CognitiveCycle()
            emotion_system = EmotionSystem()
            action_system = ActionSystem()
            
            # 시스템들 연결
            self.integrated_monitoring.connect_systems(
                cognitive_cycle=cognitive_cycle,
                emotion_system=emotion_system,
                action_system=action_system
            )
            
            # 모니터링 시작
            self.integrated_monitoring.start_monitoring()
            self.running = True
            
            print("✅ 모니터링이 시작되었습니다.")
            print("📊 대시보드가 data/monitoring/ 폴더에 생성됩니다.")
            
        except Exception as e:
            print(f"❌ 모니터링 시작 실패: {e}")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        try:
            self.integrated_monitoring.stop_monitoring()
            self.running = False
            print("✅ 모니터링이 중지되었습니다.")
            
        except Exception as e:
            print(f"❌ 모니터링 중지 실패: {e}")
    
    def get_status(self):
        """현재 상태 조회"""
        try:
            summary = self.integrated_monitoring.get_system_summary()
            
            print("\n" + "="*50)
            print("🔍 하린코어 모니터링 상태")
            print("="*50)
            
            # 기본 상태
            monitoring_status = summary.get('monitoring_status', {})
            print(f"📡 모니터링 활성화: {'✅ 활성' if monitoring_status.get('active') else '❌ 비활성'}")
            print(f"🏥 시스템 건강도: {summary.get('system_health', 0.0):.3f}")
            print(f"🚨 활성 알림: {summary.get('active_alerts', 0)}개")
            
            # 연결된 시스템
            print("\n🔗 연결된 시스템:")
            connected_systems = monitoring_status.get('connected_systems', {})
            for system_name, connected in connected_systems.items():
                status = "✅ 연결됨" if connected else "❌ 연결되지 않음"
                print(f"  - {system_name}: {status}")
            
            # 메트릭 요약
            print("\n📈 메트릭 요약:")
            metrics_summary = summary.get('metrics_summary', {})
            for metric_type, data in metrics_summary.items():
                if isinstance(data, dict) and 'average' in data:
                    print(f"  - {metric_type}: {data['average']:.3f}")
            
            # 시스템별 상세 정보
            if 'cognitive_summary' in summary:
                cognitive = summary['cognitive_summary']
                if cognitive.get('status') == 'success':
                    print("\n🧠 인지 시스템:")
                    cognitive_state = cognitive.get('cognitive_state', {})
                    for metric, value in cognitive_state.items():
                        print(f"  - {metric}: {value:.3f}")
            
            if 'emotional_summary' in summary:
                emotional = summary['emotional_summary']
                if emotional.get('status') == 'success':
                    print("\n💭 감정 시스템:")
                    emotional_data = emotional.get('emotional_summary', {})
                    for metric, value in emotional_data.items():
                        if isinstance(value, (int, float)):
                            print(f"  - {metric}: {value:.3f}")
                        else:
                            print(f"  - {metric}: {value}")
            
            if 'action_summary' in summary:
                action = summary['action_summary']
                if action.get('status') == 'success':
                    print("\n⚡ 행동 시스템:")
                    action_stats = action.get('action_stats', {})
                    for metric, value in action_stats.items():
                        if isinstance(value, (int, float)):
                            print(f"  - {metric}: {value:.3f}")
                        else:
                            print(f"  - {metric}: {value}")
            
            print("="*50)
            
        except Exception as e:
            print(f"❌ 상태 조회 실패: {e}")
    
    def generate_dashboard(self, dashboard_type: str = "all"):
        """대시보드 생성"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_dir = Path("data/monitoring")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            if dashboard_type in ["all", "integrated"]:
                dashboard_path = data_dir / f"integrated_dashboard_{timestamp}.html"
                self.integrated_monitoring.generate_integrated_dashboard(str(dashboard_path))
                print(f"✅ 통합 대시보드 생성: {dashboard_path}")
            
            if dashboard_type in ["all", "emotional"] and self.integrated_monitoring.emotion_system:
                emotional_path = data_dir / f"emotional_dashboard_{timestamp}.html"
                self.integrated_monitoring.emotion_system.generate_emotional_dashboard(str(emotional_path))
                print(f"✅ 감정 대시보드 생성: {emotional_path}")
            
            if dashboard_type in ["all", "action"] and self.integrated_monitoring.action_system:
                action_path = data_dir / f"action_dashboard_{timestamp}.html"
                self.integrated_monitoring.action_system.generate_action_dashboard(str(action_path))
                print(f"✅ 행동 대시보드 생성: {action_path}")
            
            if dashboard_type in ["all", "performance"]:
                performance_path = data_dir / f"performance_dashboard_{timestamp}.html"
                self.integrated_monitoring.monitoring_system.generate_performance_dashboard(str(performance_path))
                print(f"✅ 성능 대시보드 생성: {performance_path}")
            
        except Exception as e:
            print(f"❌ 대시보드 생성 실패: {e}")
    
    def export_data(self, filepath: str = None):
        """데이터 내보내기"""
        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"data/monitoring/harin_monitoring_data_{timestamp}"
            
            self.integrated_monitoring.export_all_data(filepath)
            print(f"✅ 데이터 내보내기 완료: {filepath}")
            
        except Exception as e:
            print(f"❌ 데이터 내보내기 실패: {e}")
    
    def create_report(self, filepath: str = None):
        """모니터링 리포트 생성"""
        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"data/monitoring/harin_monitoring_report_{timestamp}.md"
            
            self.integrated_monitoring.create_monitoring_report(filepath)
            print(f"✅ 모니터링 리포트 생성 완료: {filepath}")
            
        except Exception as e:
            print(f"❌ 리포트 생성 실패: {e}")
    
    def show_alerts(self):
        """알림 조회"""
        try:
            alerts = self.integrated_monitoring.monitoring_system.get_active_alerts()
            
            if not alerts:
                print("✅ 활성 알림이 없습니다.")
                return
            
            print(f"\n🚨 활성 알림 ({len(alerts)}개):")
            print("-" * 50)
            
            for alert in alerts:
                level_icon = {
                    "info": "ℹ️",
                    "warning": "⚠️",
                    "error": "❌",
                    "critical": "🚨"
                }.get(alert.level.value, "❓")
                
                timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{level_icon} [{timestamp}] {alert.message}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ 알림 조회 실패: {e}")
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n🎮 하린코어 모니터링 대화형 모드")
        print("명령어를 입력하세요 (help로 도움말 확인)")
        
        while True:
            try:
                command = input("\n📝 명령어 > ").strip().lower()
                
                if command in ["quit", "exit", "q"]:
                    if self.running:
                        self.stop_monitoring()
                    print("👋 모니터링 CLI를 종료합니다.")
                    break
                
                elif command in ["help", "h"]:
                    self.show_help()
                
                elif command in ["start", "s"]:
                    self.start_monitoring()
                
                elif command in ["stop", "st"]:
                    self.stop_monitoring()
                
                elif command in ["status", "st"]:
                    self.get_status()
                
                elif command in ["dashboard", "d"]:
                    self.generate_dashboard()
                
                elif command in ["export", "e"]:
                    self.export_data()
                
                elif command in ["report", "r"]:
                    self.create_report()
                
                elif command in ["alerts", "a"]:
                    self.show_alerts()
                
                elif command.startswith("dashboard "):
                    dashboard_type = command.split(" ", 1)[1]
                    self.generate_dashboard(dashboard_type)
                
                else:
                    print("❓ 알 수 없는 명령어입니다. 'help'를 입력하여 도움말을 확인하세요.")
                
            except KeyboardInterrupt:
                print("\n👋 모니터링 CLI를 종료합니다.")
                if self.running:
                    self.stop_monitoring()
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
    
    def show_help(self):
        """도움말 표시"""
        help_text = """
📚 하린코어 모니터링 CLI 도움말

기본 명령어:
  start, s          - 모니터링 시작
  stop, st          - 모니터링 중지
  status, st        - 현재 상태 조회
  dashboard, d      - 모든 대시보드 생성
  export, e         - 데이터 내보내기
  report, r         - 모니터링 리포트 생성
  alerts, a         - 활성 알림 조회
  help, h           - 이 도움말 표시
  quit, exit, q     - 종료

대시보드 옵션:
  dashboard all     - 모든 대시보드 생성
  dashboard integrated - 통합 대시보드만 생성
  dashboard emotional  - 감정 대시보드만 생성
  dashboard action     - 행동 대시보드만 생성
  dashboard performance - 성능 대시보드만 생성

예시:
  > start           # 모니터링 시작
  > status          # 상태 확인
  > dashboard       # 대시보드 생성
  > export          # 데이터 내보내기
  > quit            # 종료
"""
        print(help_text)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="하린코어 모니터링 CLI")
    parser.add_argument("command", nargs="?", choices=[
        "start", "stop", "status", "dashboard", "export", "report", "alerts", "interactive"
    ], help="실행할 명령어")
    parser.add_argument("--type", choices=["all", "integrated", "emotional", "action", "performance"], 
                       default="all", help="대시보드 타입")
    parser.add_argument("--file", help="출력 파일 경로")
    
    args = parser.parse_args()
    
    cli = MonitoringCLI()
    
    if args.command == "start":
        cli.start_monitoring()
    elif args.command == "stop":
        cli.stop_monitoring()
    elif args.command == "status":
        cli.get_status()
    elif args.command == "dashboard":
        cli.generate_dashboard(args.type)
    elif args.command == "export":
        cli.export_data(args.file)
    elif args.command == "report":
        cli.create_report(args.file)
    elif args.command == "alerts":
        cli.show_alerts()
    elif args.command == "interactive":
        cli.interactive_mode()
    else:
        # 대화형 모드로 시작
        cli.interactive_mode()


if __name__ == "__main__":
    main() 