"""
하린코어 모니터링 시스템 - Real-time Monitoring & Graph Visualization
실시간 상태 모니터링, 그래프 시각화, 성능 추적 시스템
"""

import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
import threading
import queue

# 시각화 관련 import
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("시각화 라이브러리가 없습니다. 텍스트 기반 모니터링만 사용 가능합니다.")


class MetricType(Enum):
    """메트릭 유형"""
    PERFORMANCE = "performance"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    MEMORY = "memory"
    COLLABORATION = "collaboration"
    SYSTEM = "system"


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """메트릭 데이터 포인트"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """알림"""
    id: str
    level: AlertLevel
    message: str
    metric_type: MetricType
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None


class SystemSnapshot(BaseModel):
    """시스템 스냅샷"""
    timestamp: datetime = Field(default_factory=datetime.now)
    cognitive_state: Dict[str, Any] = Field(default_factory=dict)
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    memory_state: Dict[str, Any] = Field(default_factory=dict)
    agent_states: Dict[str, Any] = Field(default_factory=dict)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    system_health: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphNode(BaseModel):
    """그래프 노드"""
    node_id: str
    node_type: str
    label: str
    position: Tuple[float, float] = (0.0, 0.0)
    properties: Dict[str, Any] = Field(default_factory=dict)
    size: float = 1.0
    color: str = "#1f77b4"


class GraphEdge(BaseModel):
    """그래프 엣지"""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)
    color: str = "#666666"


class CognitiveGraph(BaseModel):
    """인지 그래프"""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    layout: str = "spring"  # spring, circular, hierarchical
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MonitoringSystem:
    """모니터링 시스템"""
    
    def __init__(self):
        self.metrics: Dict[MetricType, List[MetricPoint]] = {
            metric_type: [] for metric_type in MetricType
        }
        self.alerts: List[Alert] = []
        self.snapshots: List[SystemSnapshot] = []
        self.cognitive_graphs: List[CognitiveGraph] = []
        
        # 실시간 모니터링 설정
        self.monitoring_active = False
        self.monitoring_interval = 1.0  # 초
        self.max_metrics_history = 1000
        self.max_snapshots_history = 100
        
        # 알림 임계값
        self.alert_thresholds = {
            MetricType.PERFORMANCE: {"warning": 0.7, "error": 0.5, "critical": 0.3},
            MetricType.EMOTIONAL: {"warning": 0.6, "error": 0.4, "critical": 0.2},
            MetricType.COGNITIVE: {"warning": 0.7, "error": 0.5, "critical": 0.3},
            MetricType.MEMORY: {"warning": 0.8, "error": 0.6, "critical": 0.4},
            MetricType.COLLABORATION: {"warning": 0.6, "error": 0.4, "critical": 0.2},
            MetricType.SYSTEM: {"warning": 0.8, "error": 0.6, "critical": 0.4}
        }
        
        # 모니터링 스레드
        self.monitoring_thread = None
        self.monitoring_queue = queue.Queue()
    
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            print("실시간 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("실시간 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 시스템 스냅샷 생성
                snapshot = self._create_system_snapshot()
                self.snapshots.append(snapshot)
                
                # 메트릭 수집
                self._collect_metrics(snapshot)
                
                # 알림 확인
                self._check_alerts()
                
                # 히스토리 정리
                self._cleanup_history()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"모니터링 오류: {e}")
                time.sleep(self.monitoring_interval)
    
    def _create_system_snapshot(self) -> SystemSnapshot:
        """시스템 스냅샷 생성"""
        # 실제 구현에서는 하린코어의 현재 상태를 가져와야 함
        snapshot = SystemSnapshot(
            cognitive_state={
                "attention_level": 0.8,
                "reasoning_complexity": 0.6,
                "decision_confidence": 0.7
            },
            emotional_state={
                "emotional_balance": 0.75,
                "empathy_level": 0.8,
                "stress_level": 0.3
            },
            memory_state={
                "memory_usage": 0.6,
                "retrieval_efficiency": 0.7,
                "consolidation_progress": 0.5
            },
            agent_states={
                "active_agents": 5,
                "collaboration_efficiency": 0.8,
                "task_distribution": "balanced"
            },
            performance_metrics={
                "response_time": 0.5,
                "accuracy": 0.85,
                "learning_rate": 0.6
            }
        )
        
        # 시스템 건강도 계산
        snapshot.system_health = self._calculate_system_health(snapshot)
        
        return snapshot
    
    def _calculate_system_health(self, snapshot: SystemSnapshot) -> float:
        """시스템 건강도 계산"""
        health_factors = [
            snapshot.performance_metrics.get("accuracy", 0.5),
            snapshot.emotional_state.get("emotional_balance", 0.5),
            snapshot.cognitive_state.get("decision_confidence", 0.5),
            snapshot.memory_state.get("retrieval_efficiency", 0.5),
            snapshot.agent_states.get("collaboration_efficiency", 0.5)
        ]
        
        return sum(health_factors) / len(health_factors)
    
    def _collect_metrics(self, snapshot: SystemSnapshot):
        """메트릭 수집"""
        current_time = time.time()
        
        # 성능 메트릭
        performance_metrics = [
            ("response_time", snapshot.performance_metrics.get("response_time", 0.5)),
            ("accuracy", snapshot.performance_metrics.get("accuracy", 0.5)),
            ("learning_rate", snapshot.performance_metrics.get("learning_rate", 0.5))
        ]
        
        for metric_name, value in performance_metrics:
            self._add_metric_point(MetricType.PERFORMANCE, metric_name, value, current_time)
        
        # 감정 메트릭
        emotional_metrics = [
            ("emotional_balance", snapshot.emotional_state.get("emotional_balance", 0.5)),
            ("empathy_level", snapshot.emotional_state.get("empathy_level", 0.5)),
            ("stress_level", snapshot.emotional_state.get("stress_level", 0.5))
        ]
        
        for metric_name, value in emotional_metrics:
            self._add_metric_point(MetricType.EMOTIONAL, metric_name, value, current_time)
        
        # 인지 메트릭
        cognitive_metrics = [
            ("attention_level", snapshot.cognitive_state.get("attention_level", 0.5)),
            ("reasoning_complexity", snapshot.cognitive_state.get("reasoning_complexity", 0.5)),
            ("decision_confidence", snapshot.cognitive_state.get("decision_confidence", 0.5))
        ]
        
        for metric_name, value in cognitive_metrics:
            self._add_metric_point(MetricType.COGNITIVE, metric_name, value, current_time)
        
        # 메모리 메트릭
        memory_metrics = [
            ("memory_usage", snapshot.memory_state.get("memory_usage", 0.5)),
            ("retrieval_efficiency", snapshot.memory_state.get("retrieval_efficiency", 0.5)),
            ("consolidation_progress", snapshot.memory_state.get("consolidation_progress", 0.5))
        ]
        
        for metric_name, value in memory_metrics:
            self._add_metric_point(MetricType.MEMORY, metric_name, value, current_time)
        
        # 협업 메트릭
        collaboration_metrics = [
            ("active_agents", snapshot.agent_states.get("active_agents", 0) / 10.0),
            ("collaboration_efficiency", snapshot.agent_states.get("collaboration_efficiency", 0.5))
        ]
        
        for metric_name, value in collaboration_metrics:
            self._add_metric_point(MetricType.COLLABORATION, metric_name, value, current_time)
        
        # 시스템 메트릭
        system_metrics = [
            ("system_health", snapshot.system_health),
            ("overall_performance", sum(snapshot.performance_metrics.values()) / len(snapshot.performance_metrics))
        ]
        
        for metric_name, value in system_metrics:
            self._add_metric_point(MetricType.SYSTEM, metric_name, value, current_time)
    
    def _add_metric_point(self, metric_type: MetricType, metric_name: str, value: float, timestamp: float):
        """메트릭 포인트 추가"""
        metric_point = MetricPoint(
            timestamp=timestamp,
            value=value,
            metadata={"metric_name": metric_name}
        )
        
        self.metrics[metric_type].append(metric_point)
        
        # 히스토리 제한
        if len(self.metrics[metric_type]) > self.max_metrics_history:
            self.metrics[metric_type] = self.metrics[metric_type][-self.max_metrics_history:]
    
    def _check_alerts(self):
        """알림 확인"""
        current_time = time.time()
        
        for metric_type, thresholds in self.alert_thresholds.items():
            if not self.metrics[metric_type]:
                continue
            
            # 최근 메트릭의 평균값 계산
            recent_metrics = self.metrics[metric_type][-10:]  # 최근 10개
            avg_value = sum(mp.value for mp in recent_metrics) / len(recent_metrics)
            
            # 임계값 확인
            if avg_value <= thresholds["critical"]:
                self._create_alert(AlertLevel.CRITICAL, metric_type, avg_value, current_time)
            elif avg_value <= thresholds["error"]:
                self._create_alert(AlertLevel.ERROR, metric_type, avg_value, current_time)
            elif avg_value <= thresholds["warning"]:
                self._create_alert(AlertLevel.WARNING, metric_type, avg_value, current_time)
    
    def _create_alert(self, level: AlertLevel, metric_type: MetricType, value: float, timestamp: float):
        """알림 생성"""
        alert = Alert(
            id=f"alert_{int(timestamp)}",
            level=level,
            message=f"{metric_type.value} 메트릭이 {level.value} 수준입니다. (값: {value:.3f})",
            metric_type=metric_type,
            timestamp=timestamp
        )
        
        self.alerts.append(alert)
        print(f"알림 생성: {alert.message}")
    
    def _cleanup_history(self):
        """히스토리 정리"""
        # 스냅샷 히스토리 제한
        if len(self.snapshots) > self.max_snapshots_history:
            self.snapshots = self.snapshots[-self.max_snapshots_history:]
        
        # 오래된 알림 해결
        current_time = time.time()
        for alert in self.alerts:
            if not alert.resolved and current_time - alert.timestamp > 3600:  # 1시간 후 자동 해결
                alert.resolved = True
                alert.resolution_time = current_time
    
    def get_metrics_summary(self, metric_type: Optional[MetricType] = None, 
                          time_window: Optional[float] = None) -> Dict[str, Any]:
        """메트릭 요약 조회"""
        summary = {}
        
        if metric_type:
            metric_types = [metric_type]
        else:
            metric_types = list(MetricType)
        
        current_time = time.time()
        
        for mt in metric_types:
            if not self.metrics[mt]:
                continue
            
            # 시간 필터 적용
            if time_window:
                filtered_metrics = [
                    mp for mp in self.metrics[mt] 
                    if current_time - mp.timestamp <= time_window
                ]
            else:
                filtered_metrics = self.metrics[mt]
            
            if not filtered_metrics:
                continue
            
            values = [mp.value for mp in filtered_metrics]
            summary[mt.value] = {
                "count": len(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else 0.0
            }
        
        return summary
    
    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 조회"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """알림 해결"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolution_time = time.time()
                break
    
    def create_cognitive_graph(self, snapshot: SystemSnapshot) -> CognitiveGraph:
        """인지 그래프 생성"""
        graph = CognitiveGraph()
        
        # 노드 생성
        nodes = [
            GraphNode(
                node_id="attention",
                node_type="cognitive",
                label="주의",
                position=(0.0, 1.0),
                properties={"level": snapshot.cognitive_state.get("attention_level", 0.5)},
                size=snapshot.cognitive_state.get("attention_level", 0.5) * 2 + 1
            ),
            GraphNode(
                node_id="reasoning",
                node_type="cognitive",
                label="추론",
                position=(1.0, 0.5),
                properties={"complexity": snapshot.cognitive_state.get("reasoning_complexity", 0.5)},
                size=snapshot.cognitive_state.get("reasoning_complexity", 0.5) * 2 + 1
            ),
            GraphNode(
                node_id="decision",
                node_type="cognitive",
                label="결정",
                position=(0.0, 0.0),
                properties={"confidence": snapshot.cognitive_state.get("decision_confidence", 0.5)},
                size=snapshot.cognitive_state.get("decision_confidence", 0.5) * 2 + 1
            ),
            GraphNode(
                node_id="emotion",
                node_type="emotional",
                label="감정",
                position=(-1.0, 0.5),
                properties={"balance": snapshot.emotional_state.get("emotional_balance", 0.5)},
                size=snapshot.emotional_state.get("emotional_balance", 0.5) * 2 + 1,
                color="#ff7f0e"
            ),
            GraphNode(
                node_id="memory",
                node_type="memory",
                label="기억",
                position=(0.5, -0.5),
                properties={"efficiency": snapshot.memory_state.get("retrieval_efficiency", 0.5)},
                size=snapshot.memory_state.get("retrieval_efficiency", 0.5) * 2 + 1,
                color="#2ca02c"
            )
        ]
        
        # 엣지 생성
        edges = [
            GraphEdge(
                edge_id="attention_reasoning",
                source_id="attention",
                target_id="reasoning",
                edge_type="influences",
                weight=0.8
            ),
            GraphEdge(
                edge_id="reasoning_decision",
                source_id="reasoning",
                target_id="decision",
                edge_type="leads_to",
                weight=0.9
            ),
            GraphEdge(
                edge_id="emotion_decision",
                source_id="emotion",
                target_id="decision",
                edge_type="influences",
                weight=0.7
            ),
            GraphEdge(
                edge_id="memory_reasoning",
                source_id="memory",
                target_id="reasoning",
                edge_type="supports",
                weight=0.6
            )
        ]
        
        graph.nodes = nodes
        graph.edges = edges
        graph.metadata = {
            "timestamp": snapshot.timestamp.isoformat(),
            "system_health": snapshot.system_health
        }
        
        return graph
    
    def save_graph_visualization(self, graph: CognitiveGraph, filepath: str):
        """그래프 시각화 저장"""
        if not VISUALIZATION_AVAILABLE:
            print("시각화 라이브러리가 없어 그래프를 저장할 수 없습니다.")
            return
        
        try:
            # NetworkX 그래프 생성
            G = nx.DiGraph()
            
            # 노드 추가
            for node in graph.nodes:
                G.add_node(node.node_id, **node.properties)
            
            # 엣지 추가
            for edge in graph.edges:
                G.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
            
            # 레이아웃 계산
            if graph.layout == "spring":
                pos = nx.spring_layout(G)
            elif graph.layout == "circular":
                pos = nx.circular_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # 그래프 그리기
            plt.figure(figsize=(12, 8))
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                   node_size=1000, font_size=10, font_weight='bold',
                   arrows=True, edge_color='gray', arrowsize=20)
            
            plt.title("하린코어 인지 그래프")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"그래프 시각화 저장 완료: {filepath}")
            
        except Exception as e:
            print(f"그래프 시각화 저장 실패: {e}")
    
    def generate_performance_dashboard(self, filepath: str):
        """성능 대시보드 생성"""
        if not VISUALIZATION_AVAILABLE:
            print("시각화 라이브러리가 없어 대시보드를 생성할 수 없습니다.")
            return
        
        try:
            # 서브플롯 생성
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('시스템 건강도', '성능 메트릭', '감정 상태', '메모리 사용량'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 시스템 건강도
            system_metrics = self.metrics.get(MetricType.SYSTEM, [])
            if system_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in system_metrics]
                values = [mp.value for mp in system_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="시스템 건강도", line=dict(color='green')),
                    row=1, col=1
                )
            
            # 성능 메트릭
            performance_metrics = self.metrics.get(MetricType.PERFORMANCE, [])
            if performance_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in performance_metrics]
                values = [mp.value for mp in performance_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="성능", line=dict(color='blue')),
                    row=1, col=2
                )
            
            # 감정 상태
            emotional_metrics = self.metrics.get(MetricType.EMOTIONAL, [])
            if emotional_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in emotional_metrics]
                values = [mp.value for mp in emotional_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="감정 상태", line=dict(color='orange')),
                    row=2, col=1
                )
            
            # 메모리 사용량
            memory_metrics = self.metrics.get(MetricType.MEMORY, [])
            if memory_metrics:
                timestamps = [datetime.fromtimestamp(mp.timestamp) for mp in memory_metrics]
                values = [mp.value for mp in memory_metrics]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name="메모리", line=dict(color='red')),
                    row=2, col=2
                )
            
            # 레이아웃 설정
            fig.update_layout(
                title="하린코어 실시간 모니터링 대시보드",
                height=800,
                showlegend=True
            )
            
            # HTML 파일로 저장
            fig.write_html(filepath)
            print(f"성능 대시보드 생성 완료: {filepath}")
            
        except Exception as e:
            print(f"성능 대시보드 생성 실패: {e}")
    
    def export_monitoring_data(self, filepath: str):
        """모니터링 데이터 내보내기"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                metric_type.value: [
                    {
                        "timestamp": mp.timestamp,
                        "value": mp.value,
                        "metadata": mp.metadata
                    }
                    for mp in metric_points
                ]
                for metric_type, metric_points in self.metrics.items()
            },
            "alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "metric_type": alert.metric_type.value,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                    "resolution_time": alert.resolution_time
                }
                for alert in self.alerts
            ],
            "snapshots": [
                {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "cognitive_state": snapshot.cognitive_state,
                    "emotional_state": snapshot.emotional_state,
                    "memory_state": snapshot.memory_state,
                    "agent_states": snapshot.agent_states,
                    "performance_metrics": snapshot.performance_metrics,
                    "system_health": snapshot.system_health
                }
                for snapshot in self.snapshots
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"모니터링 데이터 내보내기 완료: {filepath}")


# 전역 모니터링 시스템 인스턴스
monitoring_system = MonitoringSystem() 