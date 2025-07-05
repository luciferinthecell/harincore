"""
하린코어 통합 추론 엔진
메타 인지와 추론 시스템을 통합하여 지능적인 사고 처리를 수행
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os

# 하위 시스템 import - 개선된 방식
def safe_import(module_path, class_name=None):
    """안전한 import 함수"""
    try:
        if class_name:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(module_path)
    except ImportError as e:
        print(f"⚠️ {module_path}를 찾을 수 없습니다: {e}")
        return None
    except AttributeError as e:
        print(f"⚠️ {module_path}.{class_name}을 찾을 수 없습니다: {e}")
        return None

# 드리프트 시스템
DriftMonitor = safe_import('reasoning.drift_system.drift_monitor', 'DriftMonitor')

# 감정 시스템
RhythmEngine = safe_import('reasoning.emotion_system.rhythm_emotion_engine', 'RhythmEngine')
EmotionTrace = safe_import('reasoning.emotion_system.rhythm_emotion_engine', 'EmotionTrace')
regulate = safe_import('reasoning.emotion_system.rhythm_governor', 'regulate')

# 메타 인지 시스템
Metacognition = safe_import('reasoning.meta_cognition_system.metacognition', 'Metacognition')
evaluate_path = safe_import('reasoning.meta_cognition_system.meta_evaluator', 'evaluate_path')

# 전문가 시스템
get_crew = safe_import('reasoning.expert_system.expert_system', 'get_crew')
ExpertRouter = safe_import('reasoning.expert_system.expert_router', 'ExpertRouter')
form_crew = safe_import('reasoning.expert_system.crew_formation_engine', 'form_crew')

# 사고 시스템
ThoughtProcessor = safe_import('reasoning.thought_system.thought_processor', 'ThoughtProcessor')
diversify_thought = safe_import('reasoning.thought_system.thought_diversifier', 'diversify_thought')
process_thought_graph = safe_import('reasoning.thought_system.thought_diversifier', 'process_thought_graph')

# 기타 시스템 import
try:
    from .adaptive_loop import AdaptiveReasoningLoop
    from .harin_reasoner import generate_paths, select_best_path
    from .hypothesis_evaluator import HypothesisEvaluator
    from .feedback_engine import FeedbackEngine
    from .identity_manager import IdentityManager
    from .reloop_trigger import ReloopTrigger
    from .value_shift_tracker import ValueShiftTracker
except ImportError:
    print("⚠️ 일부 reasoning 모듈을 찾을 수 없습니다. 기본 기능으로 실행합니다.")
    AdaptiveReasoningLoop = generate_paths = select_best_path = HypothesisEvaluator = None
    FeedbackEngine = IdentityManager = ReloopTrigger = ValueShiftTracker = None


class IntegratedReasoningEngine:
    """통합 추론 엔진 - 메타 인지와 추론 시스템을 통합"""
    
    def __init__(self):
        """통합 추론 엔진 초기화"""
        self.session_state = {
            "current_emotion": "중립",
            "rhythm_scores": {"truth": 0.7, "resonance": 0.7, "responsibility": 0.7},
            "user_context": {},
            "reasoning_history": [],
            "meta_evaluations": [],
            "expert_contributions": [],
            "thought_chains": []
        }
        
        # 하위 시스템 초기화
        self._initialize_subsystems()
        
        print("🚀 통합 추론 엔진 초기화 완료")
    
    def _initialize_subsystems(self):
        """하위 시스템들을 초기화"""
        # 드리프트 시스템
        if DriftMonitor:
            self.drift_monitor = DriftMonitor()
        else:
            self.drift_monitor = None
        
        # 감정 시스템
        if RhythmEngine and EmotionTrace:
            self.rhythm_engine = RhythmEngine()
            self.emotion_trace = EmotionTrace()
        else:
            self.rhythm_engine = self.emotion_trace = None
        
        # 메타 인지 시스템
        if Metacognition:
            self.metacognition = Metacognition()
        else:
            self.metacognition = None
        
        if evaluate_path:
            self.meta_evaluator = evaluate_path
        else:
            self.meta_evaluator = None
        
        # 전문가 시스템
        if ExpertRouter:
            self.expert_router = ExpertRouter()
        else:
            self.expert_router = None
        
        if form_crew:
            self.crew_engine = form_crew
        else:
            self.crew_engine = None
        
        # 사고 시스템
        if ThoughtProcessor:
            self.thought_processor = ThoughtProcessor()
        else:
            self.thought_processor = None
        
        if diversify_thought:
            self.thought_diversifier = diversify_thought
        else:
            self.thought_diversifier = None
        
        # 기타 시스템
        if AdaptiveReasoningLoop:
            self.adaptive_loop = AdaptiveReasoningLoop()
        else:
            self.adaptive_loop = None
        
        if FeedbackEngine:
            self.feedback_engine = FeedbackEngine()
        else:
            self.feedback_engine = None
        
        if IdentityManager:
            self.identity_manager = IdentityManager()
        else:
            self.identity_manager = None
        
        if ReloopTrigger:
            self.reloop_trigger = ReloopTrigger()
        else:
            self.reloop_trigger = None
        
        if ValueShiftTracker:
            self.value_tracker = ValueShiftTracker()
        else:
            self.value_tracker = None
    
    def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """사용자 입력을 처리하여 통합된 추론 결과 반환"""
        try:
            # 1. 입력 전처리 및 컨텍스트 업데이트
            processed_input = self._preprocess_input(user_input, context)
            
            # 2. 감정 및 리듬 분석
            emotion_analysis = self._analyze_emotion_and_rhythm(processed_input)
            
            # 3. 드리프트 감지
            drift_analysis = self._detect_drift(emotion_analysis)
            
            # 4. 전문가 크루 형성
            expert_crew = self._form_expert_crew(processed_input, emotion_analysis)
            
            # 5. 사고 처리 및 다양화
            thought_analysis = self._process_thoughts(processed_input, expert_crew)
            
            # 6. 메타 인지 평가
            meta_evaluation = self._evaluate_meta_cognition(thought_analysis)
            
            # 7. 최적 경로 선택
            best_path = self._select_optimal_path(thought_analysis, meta_evaluation)
            
            # 8. 피드백 및 상태 업데이트
            self._update_session_state(processed_input, emotion_analysis, meta_evaluation)
            
            # 9. 결과 통합
            result = self._integrate_results(
                processed_input, emotion_analysis, drift_analysis,
                expert_crew, thought_analysis, meta_evaluation, best_path
            )
            
            return result
            
        except Exception as e:
            print(f"❌ 추론 처리 중 오류: {e}")
            return self._create_fallback_response(user_input, str(e))
    
    def _preprocess_input(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """입력 전처리"""
        return {
            "raw_input": user_input,
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
            "processed": True
        }
    
    def _analyze_emotion_and_rhythm(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """감정 및 리듬 분석"""
        if not self.rhythm_engine or not self.emotion_trace:
            return {"emotion": "중립", "rhythm": {"truth": 0.7, "resonance": 0.7, "responsibility": 0.7}}
        
        # 감정 분석 (간단한 키워드 기반)
        input_text = processed_input["raw_input"].lower()
        emotion = "중립"
        
        if any(word in input_text for word in ["좋아", "감사", "행복"]):
            emotion = "기쁨"
        elif any(word in input_text for word in ["싫어", "화나", "분노"]):
            emotion = "분노"
        elif any(word in input_text for word in ["슬퍼", "우울", "불안"]):
            emotion = "슬픔"
        
        # 리듬 업데이트
        rhythm_scores = {
            "truth": 0.7 + (0.1 if emotion == "기쁨" else -0.1 if emotion == "분노" else 0),
            "resonance": 0.7 + (0.1 if emotion == "기쁨" else -0.1 if emotion == "슬픔" else 0),
            "responsibility": 0.7 + (0.1 if emotion == "중립" else -0.05)
        }
        
        self.rhythm_engine.update(rhythm_scores["truth"], rhythm_scores["resonance"], rhythm_scores["responsibility"])
        self.emotion_trace.push(emotion)
        
        return {
            "emotion": emotion,
            "rhythm": rhythm_scores,
            "dominant_emotion": self.emotion_trace.dominant_emotion(),
            "recent_emotions": self.emotion_trace.recent()
        }
    
    def _detect_drift(self, emotion_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """드리프트 감지"""
        if not self.drift_monitor:
            return {"drift": False, "status": "monitor_unavailable"}
        
        # 리듬 점수를 드리프트 모니터에 전달
        rhythm_score = emotion_analysis["rhythm"]["truth"]
        drift_result = self.drift_monitor.push(rhythm_score)
        
        return {
            "drift": drift_result.get("drift", False),
            "stdev": drift_result.get("stdev", 0),
            "mean": drift_result.get("mean", 0),
            "status": drift_result.get("status", "unknown")
        }
    
    def _form_expert_crew(self, processed_input: Dict[str, Any], emotion_analysis: Dict[str, Any]) -> List[Any]:
        """전문가 크루 형성"""
        if not get_crew:
            return []
        
        # 간단한 의도 및 키워드 추출
        input_text = processed_input["raw_input"].lower()
        intent = "일반"
        keywords = []
        
        if any(word in input_text for word in ["어떻게", "왜", "무엇"]):
            intent = "질문"
        elif any(word in input_text for word in ["해줘", "해봐", "해야"]):
            intent = "명령"
        
        if any(word in input_text for word in ["창의", "새로운", "혁신"]):
            keywords.append("창의")
        if any(word in input_text for word in ["왜", "반문", "의문"]):
            keywords.append("반문")
        
        emotion = emotion_analysis["emotion"]
        
        # 전문가 크루 생성
        crew = get_crew(intent, emotion, keywords)
        
        return crew
    
    def _process_thoughts(self, processed_input: Dict[str, Any], expert_crew: List[Any]) -> Dict[str, Any]:
        """사고 처리 및 다양화"""
        if not self.thought_processor:
            return {"thoughts": [], "diversified": False}
        
        try:
            # 사고 처리
            thought_result = self.thought_processor.process_user_input(
                processed_input["raw_input"],
                context=str(processed_input["context"]),
                knowledge=""
            )
            
            # 사고 다양화
            if self.thought_diversifier:
                diversified_thoughts = self.thought_diversifier(
                    processed_input["raw_input"]
                )
                thought_result["diversified_thoughts"] = diversified_thoughts
            
            return thought_result
            
        except Exception as e:
            print(f"⚠️ 사고 처리 중 오류: {e}")
            return {"thoughts": [], "diversified": False, "error": str(e)}
    
    def _evaluate_meta_cognition(self, thought_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """메타 인지 평가"""
        if not self.metacognition:
            return {"trust_score": 0.5, "evaluation": "metacognition_unavailable"}
        
        try:
            # 사고 계획을 메타 인지로 평가
            plan = {
                "steps": thought_analysis.get("thoughts", []),
                "arguments": {"pros": [], "cons": []},
                "search_queries": []
            }
            
            meta_evaluation = self.metacognition.evaluate_trust_score(plan)
            reflection = self.metacognition.reflect(meta_evaluation)
            
            return {
                "trust_score": meta_evaluation["trust_score"],
                "signals": meta_evaluation["signals"],
                "comments": meta_evaluation["comments"],
                "reflection": reflection
            }
            
        except Exception as e:
            print(f"⚠️ 메타 인지 평가 중 오류: {e}")
            return {"trust_score": 0.5, "evaluation": "error", "error": str(e)}
    
    def _select_optimal_path(self, thought_analysis: Dict[str, Any], meta_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """최적 경로 선택"""
        if not generate_paths or not select_best_path:
            return {"path": "default", "score": 0.5}
        
        try:
            # 사고들을 경로로 변환
            thoughts = thought_analysis.get("thoughts", [])
            memories = []  # 실제로는 메모리 시스템에서 가져와야 함
            
            paths = generate_paths([str(t) for t in thoughts], memories)
            best_path = select_best_path(paths)
            
            return best_path
            
        except Exception as e:
            print(f"⚠️ 경로 선택 중 오류: {e}")
            return {"path": "default", "score": 0.5, "error": str(e)}
    
    def _update_session_state(self, processed_input: Dict[str, Any], emotion_analysis: Dict[str, Any], meta_evaluation: Dict[str, Any]):
        """세션 상태 업데이트"""
        self.session_state["current_emotion"] = emotion_analysis["emotion"]
        self.session_state["rhythm_scores"] = emotion_analysis["rhythm"]
        self.session_state["reasoning_history"].append({
            "input": processed_input["raw_input"],
            "timestamp": processed_input["timestamp"],
            "emotion": emotion_analysis["emotion"],
            "trust_score": meta_evaluation.get("trust_score", 0.5)
        })
        
        # 히스토리 크기 제한
        if len(self.session_state["reasoning_history"]) > 50:
            self.session_state["reasoning_history"] = self.session_state["reasoning_history"][-50:]
    
    def _integrate_results(self, processed_input: Dict[str, Any], emotion_analysis: Dict[str, Any], 
                          drift_analysis: Dict[str, Any], expert_crew: List[Any], 
                          thought_analysis: Dict[str, Any], meta_evaluation: Dict[str, Any], 
                          best_path: Dict[str, Any]) -> Dict[str, Any]:
        """결과 통합"""
        return {
            "input": processed_input["raw_input"],
            "timestamp": processed_input["timestamp"],
            "emotion_analysis": emotion_analysis,
            "drift_analysis": drift_analysis,
            "expert_crew_size": len(expert_crew),
            "thought_analysis": thought_analysis,
            "meta_evaluation": meta_evaluation,
            "best_path": best_path,
            "session_state": {
                "current_emotion": self.session_state["current_emotion"],
                "reasoning_history_count": len(self.session_state["reasoning_history"])
            },
            "system_status": "integrated"
        }
    
    def _create_fallback_response(self, user_input: str, error: str) -> Dict[str, Any]:
        """폴백 응답 생성"""
        return {
            "input": user_input,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "fallback": True,
            "response": "죄송합니다. 현재 시스템에 일시적인 문제가 있습니다."
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            "subsystems": {
                "drift_monitor": self.drift_monitor is not None,
                "rhythm_engine": self.rhythm_engine is not None,
                "emotion_trace": self.emotion_trace is not None,
                "metacognition": self.metacognition is not None,
                "expert_router": self.expert_router is not None,
                "thought_processor": self.thought_processor is not None,
                "adaptive_loop": self.adaptive_loop is not None,
                "feedback_engine": self.feedback_engine is not None,
                "identity_manager": self.identity_manager is not None
            },
            "session_state": self.session_state,
            "active_subsystems": sum([
                self.drift_monitor is not None,
                self.rhythm_engine is not None,
                self.metacognition is not None,
                self.thought_processor is not None
            ])
        }


# 사용 예시
if __name__ == "__main__":
    engine = IntegratedReasoningEngine()
    
    # 테스트 입력 처리
    test_input = "안녕하세요! 오늘 날씨가 정말 좋네요."
    result = engine.process_input(test_input)
    
    print("🔍 통합 추론 엔진 테스트 결과:")
    print(f"입력: {result['input']}")
    print(f"감정: {result['emotion_analysis']['emotion']}")
    print(f"신뢰도: {result['meta_evaluation']['trust_score']}")
    print(f"전문가 크루: {result['expert_crew_size']}명")
    
    # 시스템 상태 확인
    status = engine.get_system_status()
    print(f"\n📊 시스템 상태:")
    print(f"활성 하위 시스템: {status['active_subsystems']}개")
    print(f"세션 히스토리: {status['session_state']['reasoning_history_count']}개") 