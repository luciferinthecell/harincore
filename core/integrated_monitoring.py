"""
하린코어 통합 모니터링 시스템 - Integrated Monitoring System
모든 하린코어 시스템의 통합 모니터링 및 시각화
"""

import os
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# 하린코어 시스템 import
from core.monitoring_system import MonitoringSystem, MetricType, SystemSnapshot
from core.cognitive_cycle import CognitiveCycle
from core.emotion_system import EmotionSystem
from core.action_system import ActionSystem

# 시각화 관련 import
try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("시각화 라이브러리가 없습니다. 텍스트 기반 모니터링만 사용 가능합니다.")


class IntegratedMonitoringSystem:
    """통합 모니터링 시스템"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 모니터링 시스템 초기화
        self.monitoring_system = MonitoringSystem()
        
        # 하린코어 시스템들
        self.cognitive_cycle = None
        self.emotion_system = None
        self.action_system = None
        
        # 모니터링 설정
        self.monitoring_active = False
        self.monitoring_interval = self.config.get('monitoring_interval', 5.0)  # 5초
        self.dashboard_update_interval = self.config.get('dashboard_update_interval', 60.0)  # 1분
        
        # 데이터 저장 경로
        self.data_dir = Path("data/monitoring")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 모니터링 스레드
        self.monitoring_thread = None
        self.last_dashboard_update = 0
        
        print("통합 모니터링 시스템 초기화 완료")
    
    def connect_systems(self, cognitive_cycle: CognitiveCycle = None, 
                       emotion_system: EmotionSystem = None, 
                       action_system: ActionSystem = None):
        """하린코어 시스템들 연결"""
        try:
            # 시스템 연결
            if cognitive_cycle:
                self.cognitive_cycle = cognitive_cycle
                cognitive_cycle.set_monitoring_system(self.monitoring_system)
                print("인지 사이클이 모니터링 시스템에 연결되었습니다.")
            
            if emotion_system:
                self.emotion_system = emotion_system
                emotion_system.set_monitoring_system(self.monitoring_system)
                print("감정 시스템이 모니터링 시스템에 연결되었습니다.")
            
            if action_system:
                self.action_system = action_system
                action_system.set_monitoring_system(self.monitoring_system)
                print("행동 시스템이 모니터링 시스템에 연결되었습니다.")
            
            # 모니터링 시스템 시작
            self.monitoring_system.start_monitoring()
            
        except Exception as e:
            print(f"시스템 연결 오류: {e}")
    
    def start_monitoring(self):
        """통합 모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = asyncio.create_task(self._monitoring_loop())
            print("통합 모니터링 시작")
    
    def stop_monitoring(self):
        """통합 모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.cancel()
        self.monitoring_system.stop_monitoring()
        print("통합 모니터링 중지")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 시스템 상태 수집
                await self._collect_system_states()
                
                # 대시보드 업데이트
                current_time = time.time()
                if current_time - self.last_dashboard_update >= self.dashboard_update_interval:
                    await self._update_dashboards()
                    self.last_dashboard_update = current_time
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_states(self):
        """시스템 상태 수집"""
        try:
            # 인지 사이클 상태
            if self.cognitive_cycle:
                cognitive_summary = self.cognitive_cycle.get_monitoring_summary()
                self._record_cognitive_metrics(cognitive_summary)
            
            # 감정 시스템 상태
            if self.emotion_system:
                emotional_summary = self.emotion_system.get_emotional_monitoring_summary()
                self._record_emotional_metrics(emotional_summary)
            
            # 행동 시스템 상태
            if self.action_system:
                action_summary = self.action_system.get_action_monitoring_summary()
                self._record_action_metrics(action_summary)
            
        except Exception as e:
            print(f"시스템 상태 수집 오류: {e}")
    
    def _record_cognitive_metrics(self, summary: Dict[str, Any]):
        """인지 메트릭 기록"""
        if summary.get('status') == 'success':
            cognitive_state = summary.get('cognitive_state', {})
            
            # 인지 메트릭 기록
            for metric_name, value in cognitive_state.items():
                self.monitoring_system._add_metric_point(
                    MetricType.COGNITIVE, 
                    metric_name, 
                    value, 
                    time.time()
                )
    
    def _record_emotional_metrics(self, summary: Dict[str, Any]):
        """감정 메트릭 기록"""
        if summary.get('status') == 'success':
            emotional_summary = summary.get('emotional_summary', {})
            
            # 감정 안정성 기록
            if 'emotional_stability' in emotional_summary:
                self.monitoring_system._add_metric_point(
                    MetricType.EMOTIONAL,
                    'emotional_stability',
                    emotional_summary['emotional_stability'],
                    time.time()
                )
            
            # 메모리 효율성 기록
            if 'total_memories' in emotional_summary:
                memory_efficiency = min(1.0, emotional_summary['total_memories'] / 1000.0)
                self.monitoring_system._add_metric_point(
                    MetricType.MEMORY,
                    'memory_efficiency',
                    memory_efficiency,
                    time.time()
                )
    
    def _record_action_metrics(self, summary: Dict[str, Any]):
        """행동 메트릭 기록"""
        if summary.get('status') == 'success':
            action_stats = summary.get('action_stats', {})
            
            # 성공률 기록
            if 'successful_actions' in action_stats and 'total_actions' in action_stats:
                success_rate = action_stats['successful_actions'] / max(action_stats['total_actions'], 1)
                self.monitoring_system._add_metric_point(
                    MetricType.PERFORMANCE,
                    'success_rate',
                    success_rate,
                    time.time()
                )
            
            # 평균 실행 시간 기록
            if 'avg_execution_time' in action_stats:
                normalized_time = min(1.0, action_stats['avg_execution_time'] / 10.0)  # 10초 기준 정규화
                self.monitoring_system._add_metric_point(
                    MetricType.PERFORMANCE,
                    'avg_execution_time',
                    normalized_time,
                    time.time()
                )
            
            # 협업 효율성 기록
            if 'collaborative_actions' in action_stats and 'total_actions' in action_stats:
                collaboration_rate = action_stats['collaborative_actions'] / max(action_stats['total_actions'], 1)
                self.monitoring_system._add_metric_point(
                    MetricType.COLLABORATION,
                    'collaboration_rate',
                    collaboration_rate,
                    time.time()
                )
    
    async def _update_dashboards(self):
        """대시보드 업데이트"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 통합 대시보드 생성
            dashboard_path = self.data_dir / f"integrated_dashboard_{timestamp}.html"
            self.generate_integrated_dashboard(str(dashboard_path))
            
            # 시스템별 대시보드 생성
            if self.emotion_system:
                emotional_dashboard_path = self.data_dir / f"emotional_dashboard_{timestamp}.html"
                self.emotion_system.generate_emotional_dashboard(str(emotional_dashboard_path))
            
            if self.action_system:
                action_dashboard_path = self.data_dir / f"action_dashboard_{timestamp}.html"
                self.action_system.generate_action_dashboard(str(action_dashboard_path))
            
            # 성능 대시보드 생성
            performance_dashboard_path = self.data_dir / f"performance_dashboard_{timestamp}.html"
            self.monitoring_system.generate_performance_dashboard(str(performance_dashboard_path))
            
            print(f"대시보드 업데이트 완료: {timestamp}")
            
        except Exception as e:
            print(f"대시보드 업데이트 오류: {e}")
    
    def generate_integrated_dashboard(self, filepath: str):
        """통합 대시보드 생성"""
        if not VISUALIZATION_AVAILABLE:
            print("시각화 라이브러리가 없어 통합 대시보드를 생성할 수 없습니다.")
            return
        
        try:
            # 서브플롯 생성
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    '시스템 건강도', '인지 상태', 
                    '감정 상태', '메모리 효율성',
                    '행동 성능', '협업 효율성'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 시스템 건강도
            system_metrics = self.monitoring_system.metrics.get(MetricType.SYSTEM, [])
            if system_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in system_metrics]
                values = [mp.value for mp in system_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="시스템 건강도", line=dict(color='green')),
                    row=1, col=1
                )
            
            # 인지 상태
            cognitive_metrics = self.monitoring_system.metrics.get(MetricType.COGNITIVE, [])
            if cognitive_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in cognitive_metrics]
                values = [mp.value for mp in cognitive_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="인지 상태", line=dict(color='blue')),
                    row=1, col=2
                )
            
            # 감정 상태
            emotional_metrics = self.monitoring_system.metrics.get(MetricType.EMOTIONAL, [])
            if emotional_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in emotional_metrics]
                values = [mp.value for mp in emotional_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="감정 상태", line=dict(color='purple')),
                    row=2, col=1
                )
            
            # 메모리 효율성
            memory_metrics = self.monitoring_system.metrics.get(MetricType.MEMORY, [])
            if memory_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in memory_metrics]
                values = [mp.value for mp in memory_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="메모리 효율성", line=dict(color='orange')),
                    row=2, col=2
                )
            
            # 행동 성능
            performance_metrics = self.monitoring_system.metrics.get(MetricType.PERFORMANCE, [])
            if performance_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in performance_metrics]
                values = [mp.value for mp in performance_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="행동 성능", line=dict(color='red')),
                    row=3, col=1
                )
            
            # 협업 효율성
            collaboration_metrics = self.monitoring_system.metrics.get(MetricType.COLLABORATION, [])
            if collaboration_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in collaboration_metrics]
                values = [mp.value for mp in collaboration_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="협업 효율성", line=dict(color='brown')),
                    row=3, col=2
                )
            
            # 레이아웃 설정
            fig.update_layout(
                title="하린코어 통합 모니터링 대시보드",
                height=1200,
                showlegend=True
            )
            
            # HTML 파일로 저장
            fig.write_html(filepath)
            print(f"통합 대시보드 생성 완료: {filepath}")
            
        except Exception as e:
            print(f"통합 대시보드 생성 실패: {e}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """시스템 전체 요약 조회"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "monitoring_status": {
                    "active": self.monitoring_active,
                    "connected_systems": {
                        "cognitive_cycle": self.cognitive_cycle is not None,
                        "emotion_system": self.emotion_system is not None,
                        "action_system": self.action_system is not None
                    }
                },
                "metrics_summary": self.monitoring_system.get_metrics_summary(),
                "active_alerts": len(self.monitoring_system.get_active_alerts()),
                "system_health": self._calculate_overall_system_health()
            }
            
            # 시스템별 요약 추가
            if self.cognitive_cycle:
                summary["cognitive_summary"] = self.cognitive_cycle.get_monitoring_summary()
            
            if self.emotion_system:
                summary["emotional_summary"] = self.emotion_system.get_emotional_monitoring_summary()
            
            if self.action_system:
                summary["action_summary"] = self.action_system.get_action_monitoring_summary()
            
            return summary
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _calculate_overall_system_health(self) -> float:
        """전체 시스템 건강도 계산"""
        try:
            health_factors = []
            
            # 각 시스템의 건강도 수집
            if self.cognitive_cycle:
                cognitive_summary = self.cognitive_cycle.get_monitoring_summary()
                if cognitive_summary.get('status') == 'success':
                    health_factors.append(cognitive_summary.get('system_health', 0.5))
            
            if self.emotion_system:
                emotional_summary = self.emotion_system.get_emotional_monitoring_summary()
                if emotional_summary.get('status') == 'success':
                    emotional_data = emotional_summary.get('emotional_summary', {})
                    if 'emotional_stability' in emotional_data:
                        health_factors.append(emotional_data['emotional_stability'])
            
            if self.action_system:
                action_summary = self.action_system.get_action_monitoring_summary()
                if action_summary.get('status') == 'success':
                    action_stats = action_summary.get('action_stats', {})
                    if 'successful_actions' in action_stats and 'total_actions' in action_stats:
                        success_rate = action_stats['successful_actions'] / max(action_stats['total_actions'], 1)
                        health_factors.append(success_rate)
            
            # 모니터링 시스템 건강도
            metrics_summary = self.monitoring_system.get_metrics_summary()
            if metrics_summary:
                system_metrics = metrics_summary.get('system', {})
                if 'average' in system_metrics:
                    health_factors.append(system_metrics['average'])
            
            # 평균 계산
            if health_factors:
                return sum(health_factors) / len(health_factors)
            else:
                return 0.5
                
        except Exception as e:
            print(f"시스템 건강도 계산 오류: {e}")
            return 0.5
    
    def export_all_data(self, filepath: str):
        """모든 데이터 내보내기"""
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "system_summary": self.get_system_summary(),
                "monitoring_data": self.monitoring_system.export_monitoring_data(filepath + "_monitoring.json")
            }
            
            # 시스템별 데이터 내보내기
            if self.emotion_system:
                self.emotion_system.export_emotional_data(filepath + "_emotional.json")
            
            if self.action_system:
                self.action_system.export_action_data(filepath + "_action.json")
            
            # 통합 데이터 저장
            with open(filepath + "_integrated.json", 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"모든 데이터 내보내기 완료: {filepath}")
            
        except Exception as e:
            print(f"데이터 내보내기 실패: {e}")
    
    def create_monitoring_report(self, filepath: str):
        """모니터링 리포트 생성"""
        try:
            summary = self.get_system_summary()
            
            report = f"""
# 하린코어 모니터링 리포트
생성 시간: {summary.get('timestamp', 'Unknown')}

## 시스템 상태
- 모니터링 활성화: {summary.get('monitoring_status', {}).get('active', False)}
- 전체 시스템 건강도: {summary.get('system_health', 0.0):.3f}
- 활성 알림: {summary.get('active_alerts', 0)}개

## 연결된 시스템
"""
            
            connected_systems = summary.get('monitoring_status', {}).get('connected_systems', {})
            for system_name, connected in connected_systems.items():
                status = "연결됨" if connected else "연결되지 않음"
                report += f"- {system_name}: {status}\n"
            
            report += "\n## 메트릭 요약\n"
            metrics_summary = summary.get('metrics_summary', {})
            for metric_type, data in metrics_summary.items():
                if isinstance(data, dict) and 'average' in data:
                    report += f"- {metric_type}: {data['average']:.3f}\n"
            
            # 시스템별 상세 정보
            if 'cognitive_summary' in summary:
                cognitive = summary['cognitive_summary']
                if cognitive.get('status') == 'success':
                    report += "\n## 인지 시스템\n"
                    cognitive_state = cognitive.get('cognitive_state', {})
                    for metric, value in cognitive_state.items():
                        report += f"- {metric}: {value:.3f}\n"
            
            if 'emotional_summary' in summary:
                emotional = summary['emotional_summary']
                if emotional.get('status') == 'success':
                    report += "\n## 감정 시스템\n"
                    emotional_data = emotional.get('emotional_summary', {})
                    for metric, value in emotional_data.items():
                        if isinstance(value, (int, float)):
                            report += f"- {metric}: {value:.3f}\n"
                        else:
                            report += f"- {metric}: {value}\n"
            
            if 'action_summary' in summary:
                action = summary['action_summary']
                if action.get('status') == 'success':
                    report += "\n## 행동 시스템\n"
                    action_stats = action.get('action_stats', {})
                    for metric, value in action_stats.items():
                        if isinstance(value, (int, float)):
                            report += f"- {metric}: {value:.3f}\n"
                        else:
                            report += f"- {metric}: {value}\n"
            
            # 파일 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"모니터링 리포트 생성 완료: {filepath}")
            
        except Exception as e:
            print(f"모니터링 리포트 생성 실패: {e}")


# 전역 통합 모니터링 시스템 인스턴스
integrated_monitoring = IntegratedMonitoringSystem() 