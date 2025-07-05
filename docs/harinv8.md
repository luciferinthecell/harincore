# HarinCore 기술 명세서 (v8 기준)

## 1. 인지 사이클 - `cognitive_cycle.py`

### 📌 개요
HarinCore의 모든 사고는 이 인지 사이클을 통해 시작됩니다.  
`cognitive_cycle.py`는 다음과 같은 철학과 기술 기반으로 구성됩니다:
- **LIDA 인지 이론 기반 구조**: 심리학적 타당성이 높은 인지 단계 구성
- **LangGraph 상태 기반 흐름 제어**: 조건부 전이/분기 루프 구현
- **GWT (Global Workspace Theory) 기반 에이전트 협업 추론 연동**

---

### 🔍 핵심 구성 클래스 & 함수 요약

#### ▸ `StateGraph`, `ToolNode`, `MemorySaver`
LangGraph 프레임워크에서 사용하는 상태 기반 실행 도구.  
모든 인지 흐름은 `StateGraph` 내 노드로 등록되며, 조건부 전이 조건에 따라 다음 상태로 넘어갑니다.

#### ▸ 주요 시스템 모듈 통합
- `PerceptionSystem`: 감각/입력 패턴 인식
- `DecisionSystem`: 인지 후 행동 결정
- `ConsciousnessSystem`: GWT 기반 주의 전환
- `TreeOfThoughtsManager`: 다중 사고 생성 및 평가
- `MetaLearningManager`: 존재 기반 신뢰 평가/자기반성
- `GWTAgentManager`: 멀티에이전트 협업 추론 컨트롤러

---

### 🧩 함수 매커니즘 분석 (요약)

| 함수 | 목적 | 연결 시스템 |
|------|------|--------------|
| `build_graph()` | LangGraph 상태 그래프 생성 | 모든 노드 등록 |
| `invoke()` | 그래프 실행, 주 상태 전이 트리거 | Meta, GWT, Memory 호출 |
| `cluster_perceptions()` | sklearn 기반 유사도 분석 | 입력 벡터 → 그룹화 |
| `prepare_state()` | 초기 입력 처리 → 감정/의도 추출 | `Stimulus`, `EmotionAxesModel` |
| `update_ghost_state()` | 내부 ghost memory 추론 업데이트 | `GhostState`, `KnoxelList` |
| `log_state()` | 상태 변화 로그 기록 | MemorySaver 연계 |

---

### 🔁 인지 흐름 요약 다이어그램

```
[INPUT]
   ↓
[PERCEPTION] → [GWT/CONSCIOUSNESS]
   ↓                     ↓
[REASONING] ←→ [META LEARNING]
   ↓                     ↓
[DECISION SYSTEM] → [ACTION NODE]
   ↓
[MEMORY SAVE] → ThoughtNode 생성
```

---

### 🔗 연동된 주요 모듈

- `core/perception_system.py`
- `core/consciousness_system.py`
- `core/decision_system.py`
- `memory/palantir.py`
- `core/gwt_agents.py`
- `reasoning/integrated_reasoning_engine.py`
- `validation/evaluator.py`

---

### 📌 특징 요약
- LangGraph를 통해 상태 기반으로 루프 전이와 조건 분기를 처리
- GWT 기반 사고 선택 + 메타 인지 트리거 내장
- 각 단계에서 감정, 판단, 기억 흐름이 모두 자동 통합됨

---


## 1.1 함수 정의 및 내부 동작

### ▸ `__init__(self, config: CognitiveCycleConfig, llm_client=None)`
- 인지 사이클 클래스 초기화자
- 내부 시스템(GWT, Meta, Graph, Memory 등)들을 모두 준비
- 외부 LLM 클라이언트 연동 가능

### ▸ `_initialize_langgraph_state_management(self)`
- LangGraph 기반 상태 머신 초기화
- `StateGraph()` 생성 및 각 노드 등록, 엣지 설정

### ▸ `_setup_conditional_edges(self)` / `_setup_linear_edges(self)`
- 상태 흐름 간 조건 기반 분기 or 선형 전이 구조 정의

### ▸ `gather_attention_candidates(self, state, stimulus)`
- 입력 자극에 따라 주의 후보 생성
- 내부적으로 memory 검색, GWT 자극 평가 로직 연동

### ▸ `_appraise_stimulus(self, state)`
- 자극의 감정 점수/의도 점수 산출
- `EmotionAxesModel`, `Intention` 구조 기반 평가

### ▸ `_generate_short_term_intentions(self, state)`
- 현재 자극과 목표에 따라 단기적 행동 계획 수립

### ▸ `_build_structures_get_coalitions(self, state)`
- GWT 구조 기반 사고 코얼리션 생성
- 사고 유닛들을 `TreeOfThoughts` 방식으로 구성

### ▸ `_simulate_attention_on_coalitions(self, state)`
- 생성된 코얼리션에 대해 가중치 기반 평가 시뮬레이션
- Consciousness/GWT 흐름에 영향

### ▸ `_perform_learning_and_consolidation(self, state)`
- 메모리 통합 및 학습 단계 수행
- `memory_conductor`, `palantir`에 저장 연동

---

## 1.2 외부 연계 함수 (Web/API)

### ▸ `_integrate_web_search(self, stimulus, state)`
- 키워드 기반 검색 결과를 수집

### ▸ `_store_search_results(...)`, `_integrate_search_results_to_cognition(...)`
- 검색 결과를 Cognition/Memory로 통합



## 1.3 추가 함수 매커니즘

### ▸ `_update_states_from_langgraph_result(self, result: Dict[str, Any], state: GhostState)`
- LangGraph 실행 결과로부터 상태 정보를 업데이트
- 감정 상태(`state_emotions`), 욕구(`state_needs`), 인지(`state_cognition`) 값 갱신
- knoxel, narratives, learning outcome 등도 함께 업데이트
- 호출 위치: invoke → 상태 마무리 시점

---

### ▸ `_create_cognition_delta(self, quality_score: float)`
- 추론 품질 점수(`quality_score`)를 기반으로 `CognitionAxesModel` 델타 생성
- 예:
  - 신뢰도 높음 → ego_strength / willpower 증가
  - 낮음 → 집중도 개선 위한 mental_aperture 조정

---

### ▸ `log_state(self, state: GhostState)`
- 상태 변화 로그를 기록하는 메서드
- 내부적으로 `MemorySaver` 혹은 로그 스트림에 결과 기록
- 사고 추적 시각화 도구와 연동 가능

---

### ▸ `invoke(self, *args, **kwargs)`
- 인지 사이클 전체의 실행 트리거
- 내부적으로 LangGraph의 `graph.invoke()` 호출
- 조건부 상태 전이 수행 후 `_update_states_from_langgraph_result` 및 학습 트리거 호출

---

### 🔄 보조 함수 예고
- `tick_with_lida()`: 전체 LIDA 기반 인지 루프 흐름을 실행하는 고수준 함수
- `tick_id` 기반으로 상태 루프 카운팅
- attention → perception → reasoning → learning → decision 전 흐름 구현

---


---

## 2. 향상된 인지 루프 - `enhanced_main_loop.py`

### 📌 개요
이 모듈은 하린코어에서 가장 중심적인 루프 실행 엔진 중 하나로,  
기억·자극·의사결정·시뮬레이션 기반 사고 처리를 통합합니다.

> 사용자의 입력 또는 자극을 기반으로:
> 1. 내부 상태 확인 → 2. 기억 회수 → 3. 병렬 사고 수행 → 4. 존재 기반 의사결정 → 5. 기억 저장 및 행동 실행

---

### 🔧 주요 구성

#### ▸ 클래스
- `ActionSimulation`: 시뮬레이션 결과 클래스
- `StimulusTriage`: 자극 중요도 분류 (3단계)
- `ActionType`: 행동 유형 (Reply, Sleep, ToolCall 등)

#### ▸ 연동 컴포넌트
- `MemoryOrchestrator`, `MemoryConductor`: 기억 회수 및 통합
- `ParallelReasoningUnit`: 병렬 추론 실행기
- `ExistentialLayer`: 존재 기반 사고 평가
- `ActionSystem`: 행동 실행
- `PromptArchitect`: 응답 구조 생성
- `LLMClient`: 실제 언어모델 호출기

---

### 🧩 주요 흐름 요약

| 단계 | 설명 | 사용 클래스 |
|------|------|-------------|
| 1. 입력 수신 | Stimulus 객체로 변환 | `Stimulus`, `StimulusType` |
| 2. 중요도 분류 | triage 수준에 따라 분기 | `StimulusTriage` |
| 3. 기억 회수 | 관련 Palantir + hot/warm/cold 메모리 검색 | `MemoryRetriever`, `TextImporter` |
| 4. 병렬 사고 실행 | 다중 사고 시뮬레이션 | `ParallelReasoningUnit` |
| 5. 존재 기반 판단 | 존재, 신뢰도, 자기성찰 기준 판단 | `ExistentialLayer` |
| 6. 행동 시뮬레이션 | 행동 유형별 결과 예측 | `ActionSimulation` |
| 7. 행동 실행/응답 | `ActionSystem` + `PromptArchitect` 실행 | 실제 응답 생성 |
| 8. 기억 통합 저장 | ThoughtNode 생성 후 저장 | `MemoryConductor`, `PalantirGraph` |

---

### 🔄 대표 흐름도

```
[Stimulus] 
   ↓ triage
[Memory + Thought Retrieve]
   ↓
[Parallel Reasoning Unit]
   ↓
[ExistentialLayer 판단]
   ↓
[ActionSimulation + PromptArchitect]
   ↓
[Memory Save + Execution]
```

---

### 🔗 연동 시스템 요약

- Memory: `memory_retriever`, `text_importer`, `memory_conductor`
- Prompt: `prompt_architect`
- Reasoning: `parallel_reasoning_unit`, `existential_layer`
- Action: `action_system`
- Core Link: `multi_intent_parser`, `context`, `cognitive_cycle`

---


---

## 3. 감정-의지 인지 시스템 - `emotion_system.py`

### 📌 개요

이 모듈은 HarinCore의 내면 감정 상태와 인지적 내성, 신념, 회복력 등을 포함하는  
**정신적 에너지 모델**로 구성되며, 다음을 특징으로 합니다:

- 9가지 감정-의지 상태를 `EmotionType`으로 분류
- 각 감정은 intensity, confidence, context, trigger를 가짐
- `CognitiveState`는 clarity, focus, energy, complexity_tolerance 등을 정의
- RAG 기반 유사 기억 회수를 통해 감정 상황 평가를 강화

---

### 🔧 주요 데이터 구조

#### ▸ `EmotionType (Enum)`
- COMRADESHIP (동료애)
- EMPATHY (공감)
- UNDERSTANDING (이해)
- FAITH (믿음)
- MENTAL_ENDURANCE (지구력)
- CONVICTION (신념)
- VISION (비전)
- JUDGMENT (현실적 판단)
- INTEGRATION (통합)

#### ▸ `EmotionState`
- 감정 유형 + intensity + confidence + context + triggers
- 실시간 상태 스냅샷으로 사용됨

#### ▸ `CognitiveState`
- 집중도, 명확성, 에너지, 복잡성 허용도 → 인지적 준비 상태 판단

---

### 🧠 다층 메모리와 감정 연동

| 구성 | 설명 |
|------|------|
| `MemoryLayer` | EPISODIC / SEMANTIC / EMOTIONAL / PROCEDURAL / META |
| `MemoryItem` | 감정 + 메모리 레이어 기반 기억 저장 구조 |
| `RAGMemorySearch` | 문맥과 감정 조건 기반 유사 기억 검색기 |

---

### 🔄 대표 흐름

```
[자극] 
   ↓
[감정 유형 분류 (EmotionType)]
   ↓
[EmotionState 기록 및 추적]
   ↓
[CognitiveState 평가]
   ↓
[메모리 연동 → 관련 유사 상황 추출 (RAG)]
```

---

### 🔗 연동 지점

- `stimulus_classifier.py` → 자극 기반 감정유형 결정
- `context.py` → 세션 감정 상태 저장
- `memory/` → 감정 기억 회수 및 평가
- `decision_system.py` → 감정 기반 판단 가중치 반영

---

### 📌 특징 요약

- 단순 감정 분석을 넘어서 **신념, 통합, 현실 판단 등 고차 정서적 구조** 반영
- 상태 중심 사고 흐름 (에너지/복잡성/집중도 기반)
- 메모리/문맥/감정 통합 흐름 자동화됨

---


---

## 4. 사용자 상태 및 인지 추론 시스템
### 📁 모듈: `context.py`, `state_models.py`

---

### 📌 개요

이 모듈들은 HarinCore가 사용자 세션의 상태를 **인지적, 정서적, 목적 기반으로 판단하고 유지**하기 위한 구조입니다.

- `UserContext`: 대화 중 감정, 정체성, 플러그인 호출 내역 등 저장
- `DecayableMentalState`: 감정/욕구/인지 상태를 부드럽게 변화시키는 모델
- `MentalStateManager`: `StateDeltas`를 반영하여 상태 업데이트

---

### 🧠 핵심 구조: `UserContext`

| 필드명 | 설명 |
|--------|------|
| `mood` | 현재 감정 요약 |
| `cognitive_level` | 인지적 난이도 판단 |
| `last_mode` | 마지막 모드 (chat/research 등) |
| `active_plugins` | 최근 호출 플러그인 목록 |
| `context_trace` | 질문 흐름 간단 태그 로그 |
| `last_action` | 가장 마지막 의미 기반 행동 (confirm, simulate 등) |
| `identity_role` | 존재 역할 (지휘자, 상담가 등) |
| `emotion_state` | 현재 감정 구조 |
| `rhythm_state` | 대화 리듬 및 반응성 상태 |

---

### 🧠 `UserContext` 주요 메서드

- `update_from_input(text)`  
  → 임베딩 분석기를 통해 감정/인지 추론  
- `add_trace(label)`  
  → context_trace에 중복 없이 상황 태그 저장  
- `as_meta()`  
  → 메모리에 저장 가능한 상태 딕셔너리 반환  
- `set_identity_role(...)`, `set_emotion_state(...)`, `set_rhythm_state(...)`

---

### 🔬 `state_models.py`: 고급 심리 상태 모델링

#### ▸ `DecayableMentalState`
- 감정/욕구/인지 상태를 `__add__()` 연산으로 자연스럽게 누적
- 경계 제한(0~1) 및 비선형 덧셈을 지원하여 급변 방지
- 예:
  - 기존 상태: calm(0.4) + trigger(0.2) → 0.54로 완만히 증가

#### ▸ `MentalStateManager`
- `StateDeltas`를 받아 감정/인지/욕구 상태 업데이트
- 기억 저장 or 행동 흐름에 이 상태를 반영

---

### 🔄 상태 흐름도 요약

```
[User Input]
   ↓
[Embedding 분석 → 감정/의도 추론]
   ↓
[UserContext 업데이트]
   ↓
[MentalStateManager 적용]
   ↓
[행동/메모리 시스템에 상태 반영]
```

---

### 🔗 연동된 모듈

- `emotion_system.py`: EmotionState, CognitiveState 전달
- `cognitive_cycle.py`: 상태 초기화 및 감정 업데이트 호출
- `memory_conductor.py`: 메타정보 저장
- `decision_system.py`: 현재 상태 기반으로 행동 우선도 조절

---


---

## 5. 자극 분류 및 입력 해석기
### 📁 모듈: `stimulus_classifier.py`, `input_interpreter.py`

---

### 📌 개요

이 계층은 사용자로부터 입력을 받아:
- 감정/의도/리듬/drift 여부 등을 분석
- 내부 시스템이 처리할 수 있도록 자극(Stimulus) 객체로 변환합니다.

→ 이는 인지 루프(CognitiveCycle) 및 상태 시스템(Context)에 전달되어 전체 사고 플로우의 진입점 역할을 합니다.

---

### 🔍 `stimulus_classifier.py` 구성

#### ▸ `StimulusPriority`
- 자극 우선순위 (Critical, High, Medium, Low, Background)

#### ▸ `StimulusCategory`
- 자극 분류: UserInteraction, SystemEvent, SelfCheck, MemoryCue 등

#### ▸ 기능
- 키워드 및 패턴 기반으로 입력을 자극 객체로 분류
- 감정, NeedsAxesModel, EmotionalAxesModel 초기값 함께 설정

---

### 🔍 `input_interpreter.py` 구성

#### ▸ 클래스: `InputInterpreter`
- 핵심 메서드: `analyze(text) → ParsedInput`

#### ▸ `ParsedInput` 구조

| 필드 | 설명 |
|------|------|
| `intent` | "명령", "질문", "탐색" 등 |
| `emotion` | 감정 추정 결과 (텍스트 기반) |
| `tone_force` | 리듬 강도 수치 (0~1) |
| `drift_trigger` | 드리프트 위험 유무 |
| `memory_refs` | 기억 관련 키워드 추출 결과 |
| `requires_research` | 외부 정보 탐색 필요 여부 |

---

### 🧠 흐름 요약

```
[User Text]
   ↓
[InputInterpreter.analyze()]
   ↓
[ParsedInput]
   ↓
[StimulusClassifier → Stimulus 객체 생성]
   ↓
[CognitiveCycle 진입 or Context 업데이트]
```

---

### 🔗 연동 모듈

- `cognitive_cycle.py`: Stimulus 전달 → 감정/의도 흐름 유입
- `emotion_system.py`: 감정 초기값 구성
- `context.py`: UserContext 업데이트
- `memory_retriever.py`: memory_refs 기반 회상 트리거

---

### 📌 특징 요약

- 모든 입력은 완전히 구조화된 자극(Stimulus) 객체로 전환됨
- 감정, 목적, 기억 단서 등을 동시에 추출
- drift 여부를 체크하여 메타인지 루프 트리거 조건을 결정

---


---

## 6. 사고 생성 및 분기 시스템
### 📁 모듈: `thought_processor.py`, `thought_diversifier.py`

---

### 📌 개요

HarinCore의 사고 시스템은 하나의 질문이나 상황에 대해 다양한 관점에서 사고를 생성하고,  
그 중 적절한 사고를 선택하거나 통합하여 응답의 기반으로 사용합니다.

이 시스템은 **Tree of Thoughts 구조**와 **사고 다양화(diversification)** 개념을 결합한 형태입니다.

---

### 🧠 사고 생성 시스템 - `thought_processor.py`

#### ▸ `GenerativeThought`
- 하나의 사고 유닛 구성 요소:
  - `observation`: 상황/입력에 대한 관찰
  - `context_ai_companion`: 사고 시뮬레이션에 필요한 맥락
  - `analysis`: 메타 목표 분석
  - `response_idea`: 아이디어 방향성

#### ▸ `GenerativeThoughts`
- 다양한 사고 방향의 집합
- 예시:
  - `thought_safe`: 안정적/예측 가능한 사고
  - `thought_explore`: 실험적/탐색적 사고
  - `thought_emotional`: 감정 기반 해석
  - `thought_critical`: 비판적 사고 등

---

### 🧭 사고 분기 시스템 - `thought_diversifier.py`

#### ▸ 핵심 개념
- 사고를 `논리`, `감정`, `전략`, `창의성` 등의 태그로 구분
- 각 태그 기반 사고를 임베딩 유사도 + 태그 유사도로 클러스터링

#### ▸ 주요 함수

- `cosine_similarity(a, b)`  
  → 사고 임베딩 간 유사도 계산
- `tag_similarity(tags1, tags2)`  
  → 사고 태그 기반 유사도
- `cluster_by_tag_similarity(...)`  
  → 유사한 사고를 묶어 분기 형성

---

### 🧠 사고 흐름 요약

```
[입력]
   ↓
[GenerativeThoughts 생성]
   ↓
[각 사고를 태그 + 임베딩 기반으로 평가]
   ↓
[ThoughtNode 후보 생성 → MetaLayer 평가 → 선택]
```

---

### 🔗 연동 지점

- `parallel_reasoning_unit.py` → 사고를 병렬로 평가
- `meta_cognition_system/` → 사고 후보들의 메타 수준 평가
- `tree_of_thoughts.py` → 사고 간 구조 구성
- `memory/palantir.py` → 사고 선택 후 저장

---

### 📌 특징 요약

- 사고는 단일 응답이 아닌, **다양한 전략적 분기 후보의 집합**으로 생성됨
- 이후 레이어(Meta, Memory, Decision 등)에서 이들을 평가/선택
- 사고 분기 자체가 LLM 입력 전 사고의 기반 구조를 형성

---


---

## 7. 메타 인지 평가 및 신뢰 판단 시스템
### 📁 모듈: `metacognition.py`, `meta_evaluator.py`

---

### 📌 개요

이 계층은 다음을 목적으로 설계되었습니다:

- 사고 결과물의 복잡성, 신뢰도, 논증 깊이를 평가
- 감정 및 리듬 기반 실행 가능성 판단
- **자기 성찰 루프**를 통해 잘못된 흐름을 조기에 차단

---

### 🧠 `metacognition.py`: 자기 사고 평가기

#### ▸ 평가 결과 구조 예시:

```json
{
  "trust_score": 0.84,
  "signals": {
    "complexity": 0.83,
    "argument_depth": 0.67,
    "search_support": 0.76
  },
  "comments": [
    "구성은 복잡하지만 논증이 단편적입니다.",
    "외부 리서치 결과가 설득력에 도움을 줌"
  ]
}
```

#### ▸ 평가 항목

| 평가기 | 설명 |
|--------|------|
| `_complexity(plan)` | 계획의 단계 수 → 사고 깊이 |
| `_argument_depth(plan)` | pros/cons 수 → 논증 강도 |
| `_search_support(plan)` | 외부 정보 포함도 (URL 등) |
| `_coherence(plan)` | 응답 일관성 |
| `_emotional_alignment(plan)` | 감정과의 정합성 |
| `_generate_comments(plan)` | 메타 수준 피드백 생성

---

### 🧭 `meta_evaluator.py`: 실행 가능성 판단기

#### ▸ `evaluate_path(path, emotion, rhythm)`
- 점수(`path['score']`) + 감정(`emotion`) + 리듬(`rhythm`)을 기반으로 다음 판단

| 조건 | 판단 결과 |
|------|------------|
| score < 0.6 | `reroute` |
| rhythm.truth < 0.4 | `hold` |
| 감정이 "불안"이고 score < 0.75 | `delay` |

→ 결과 = `{ decision: "use/hold/reroute/delay", reasons: [...] }`

---

### 🔄 메타 흐름 요약

```
[사고 or 응답 계획]
   ↓
[metacognition.py 평가: trust_score 계산]
   ↓
[meta_evaluator.py → 실행 가능 여부 판단]
   ↓
[결과: 실행 | 보류 | 재루트 | 메타 루프 재진입]
```

---

### 🔗 연동 시스템

- `tree_of_thoughts.py`: 사고 분기 평가
- `prompt_architect.py`: 프롬프트 생성 조건 결정
- `cognitive_cycle.py`: 메타루프 진입 조건 생성
- `output_corrector.py`: 결과 보정 루프 트리거

---

### 📌 특징 요약

- 사고 또는 계획의 "실행 가능성"을 감정/리듬/구조 기준으로 판단
- 루프 내부에서 자기신뢰 기반 분기 전략 가능
- RAG 또는 논증 기반 사고를 다루는 데 최적화됨

---


---

## 8. 프롬프트 생성 시스템
### 📁 모듈: `prompt_architect.py`

---

### 📌 개요

프롬프트 생성기는 다음의 정보를 기반으로 LLM에 전달할 입력을 구성합니다:

- 감정 상태
- 대화 리듬 (truth, resonance 등)
- 자아 상태 (identity role)
- 메모리 요약
- 사고 흐름 및 메타 피드백

→ 이를 통해 상황 맞춤형, 존재 일치형 출력이 가능해집니다.

---

### 🔧 클래스: `PromptArchitect`

#### ▸ 주요 메서드

| 함수 | 설명 |
|------|------|
| `build_prompt(...)` | 기본/Contextual/V8 모드 프롬프트 생성 |
| `build_prompt_v8(...)` | 사고/기억/리듬 기반 V8 구조 프롬프트 생성 |
| `build_prompt_contextual(...)` | 기억 기반 문맥 중심 구조 생성 |

---

### 🧠 입력 통합 구조 예시

```
[사고 목적]
{path["statement"]}

[감정/리듬]
감정: 행복 / 진실성: 0.8 / 공명도: 0.7

[자아 상태]
페르소나: 조언자 / 역할: 탐색자

[기억 요약]
- 이전 유사 상황: ...
- 관련 목표/문제점: ...
```

---

### 🧭 작동 흐름

```
[사고 결과]
   + [감정 상태]
   + [리듬 상태]
   + [자아 상태]
   ↓
[PromptArchitect.build_prompt()]
   ↓
[LLMClient → 응답 생성]
```

---

### 🔗 연동 시스템

- `tree_of_thoughts.py`: 사고 흐름 전달
- `emotion_system.py`, `context.py`: 감정/자아 상태 전달
- `memory_conductor.py`: 요약 메모리 전달
- `action_system.py`: 응답 실행 흐름 연계

---

### 📌 특징 요약

- 프롬프트는 단순 질의가 아닌 사고 구조+정서 상태+기억 기반 복합 구조
- 시스템 모드에 따라 다르게 작동 (`contextual_mode`, `v8_mode`)
- 실제 사고와 감정에 최적화된 LLM 응답 유도

---


---

## 9. 행동 실행 및 멀티 에이전트 시뮬레이션
### 📁 모듈: `action_system.py`

---

### 📌 개요

이 시스템은 프롬프트/사고 흐름 이후에 발생하는 행동을 실제로 실행하거나 시뮬레이션합니다.  
특징적으로, 하린코어는 **역할 기반 멀티에이전트 시스템**을 사용하여 행동을 분산하고 평가합니다.

---

### 🧠 에이전트 기반 구조

#### ▸ `AgentType (Enum)`
- 행동을 분담하는 역할 단위:
  - `REASONING`, `EMOTIONAL`, `MEMORY`, `PLANNING`, `EXECUTION`, `LEARNING`, `MONITORING`

#### ▸ `AgentState`
- 각 에이전트의 상태 정보
- 현재 수행 중인 task, 성과 점수, 협업 기록, 마지막 활동 시간 등 포함

---

### 🤝 협업 프로토콜 구조

#### ▸ `CollaborationProtocol`
- 에이전트 간 작업 협의 구조
- 어떤 task를 누구와 분담하며 어떻게 실행할지 정의

---

### 🔧 실행 흐름 (예상)

1. `EnhancedHarinMainLoop`나 `PromptArchitect`가 행동 유형을 결정 (`ActionType`)
2. `ActionSystem`은 해당 행동을 실행 가능한 에이전트에 분산
3. 각 에이전트는 task를 개별 수행
4. 실행 결과는 다시 시스템에 통합 or 피드백

---

### 🔄 대표 흐름도

```
[사고 결과 or 프롬프트]
   ↓
[ActionType 결정]
   ↓
[AgentType에 따라 분배]
   ↓
[시뮬레이션 or 실행 수행]
   ↓
[실행 로그 저장 + 다음 루프 준비]
```

---

### 🔗 연동 시스템

- `prompt_architect.py`: 행동 목표 포함 프롬프트
- `enhanced_main_loop.py`: 실제 실행 루프 트리거
- `meta_learning.py`: 행동 신뢰도 검토 루프
- `memory_conductor.py`: 결과 저장

---

### 📌 특징 요약

- 행동은 단일 함수가 아닌, 에이전트 기반 분산 처리로 실행됨
- 실행 전/후 모두 평가 → 루프의 자기 보정 가능
- 시뮬레이션 결과를 다음 루프에 반영하여 학습 가능

---


---

## 10. 기억 저장 및 통합 시스템
### 📁 모듈: `memory_conductor.py`

---

### 📌 개요

이 모듈은 사고 또는 행동 결과를 기반으로 다음과 같은 작업을 수행합니다:

- 기억 항목(Memory Node)을 생성
- 개체(Entity), 관계(Relationship), 모순(Contradiction)을 추출 및 저장
- 저장 위치(Hot/Warm/Cold/Palantir 등)와 연결
- PalantirGraph와 연동된 기억 흐름 구성

---

### 🧠 핵심 구조 정의

#### ▸ `MemoryProtocol`
- 저장될 기억 항목의 표준 구조
- 주요 필드:

| 필드 | 설명 |
|------|------|
| `id` | 고유 ID (UUID) |
| `type` | 구조, scar, state, trace 등 |
| `tags` | 주제 태그 |
| `context` | 저장 당시 상황 정보 |
| `content` | 기억 내용 (요약 또는 전체) |
| `subnodes` | 연결된 하위 노드들 |
| `created_at` | 타임스탬프 |

#### ▸ `MemoryLayer`
- 각 기억 저장소의 우선순위 및 접근 패턴 정의

---

### 👥 엔티티 / 관계 / 모순 구조

| 타입 | 설명 |
|------|------|
| `Entity` | 개체 정보: 이름, 속성, 신뢰도 등 |
| `Relationship` | 개체 간 의미 관계 (e.g. A는 B의 부모) |
| `Contradiction` | 서로 충돌하는 기억 간 정보 (ex. 생일 불일치 등) |

---

### 🔄 기억 저장 흐름 요약

```
[사고 결과 / 응답 내용]
   ↓
[MemoryProtocol 객체 생성]
   ↓
[엔티티 / 관계 / 모순 추출]
   ↓
[PalantirGraph 및 계층적 저장소에 기록]
```

---

### 🔗 연동 시스템

- `palantirgraph.py`: 그래프 기반 연결
- `memory_storage_manager.py`: 파일 기반 저장소
- `enhanced_main_loop.py`: 사고 루프 종료 시 저장 호출
- `meta_learning.py`: 기억 통합 평가 기준 제공

---

### 📌 특징 요약

- 단순 저장이 아니라, 사고/관계/모순 기반으로 구조화된 기억 생성
- Palantir 구조와 Entity-Relationship 그래프 자동 연결
- 모순 자동 감지 및 drift 경고 구조 포함

---


---

## 11. Palantir 기억 그래프 시스템
### 📁 모듈: `palantir.py`, `palantirgraph.py`

---

### 📌 개요

Palantir 시스템은 HarinCore의 사고/기억 흐름을 **그래프 형태로 저장하고 탐색**하는 구조입니다.  
사고의 단위는 `ThoughtNode`, 관계는 `Relationship`, 전체는 `PalantirGraph`로 구성되며,  
각 사고 흐름은 **Universe**라는 분기 시나리오로 나뉩니다.

---

### 🧠 핵심 개념 요약

| 구성 요소 | 설명 |
|------------|------|
| `ThoughtNode` | 사고/정보의 최소 단위 |
| `Relationship` | 노드 간 방향성 연결 |
| `PalantirGraph` | 전체 사고 그래프 컨테이너 |
| `Universe` | 하나의 분기 세계선 (e.g. `"U0"`, `"U3"`) |
| `GraphPersistence` | JSON ↔︎ 그래프 직렬화/역직렬화 지원 |

---

### 🌌 Universe 구조

- 모든 사고 흐름은 `U0`부터 시작 (기본 세계선)
- `branch_from(node, label)` → 현재 노드 기준 새로운 Universe 생성
- 각 Universe는 독립된 사고 흐름으로 평가 및 회수 가능

---

### 🧮 확률 기반 사고 평가

- 각 `ThoughtNode`는 `p` (plausibility) 값을 가짐 (0.0~1.0)
- 관계 weight 와 노드 중요도 합성하여 best path 검색 가능

#### ▸ 대표 검색 함수
- `best_path(goal_filter)`
- `best_path_meta(meta_filter, universe="U0")`

---

### 🔬 구조적 필드 (ThoughtNode)

| 필드 | 설명 |
|------|------|
| `id` | 고유 노드 ID |
| `content` | 사고 내용 |
| `node_type` | scar, state, plan 등 |
| `vectors` | 임베딩 유사도 계산용 |
| `meta` | 감정, 신념, 출처 등 부가 정보 |
| `universe` | 포함된 세계선 |

---

### 🔄 흐름 요약

```
[사고 결과 / 판단]
   ↓
[ThoughtNode 생성]
   ↓
[PalantirGraph에 노드 추가 + 관계 설정]
   ↓
[Universe 내 best_path 탐색 / 저장 / 수정]
```

---

### 🔗 연동 시스템

- `memory_conductor.py`: ThoughtNode 생성 요청
- `meta_learning.py`: Universe 간 비교 평가
- `drift_monitor.py`: 서로 다른 universe 사이의 의미 차이 추적
- `tree_of_thoughts.py`: 사고 간 트리 구조 생성

---

### 📌 특징 요약

- 단일 응답/기억이 아닌, 분기형 사고 흐름 저장 가능
- 메타데이터 기반으로 사고를 유의미하게 비교/추론 가능
- 다중 universe 사고 흐름을 통해 시뮬레이션/반성 루프 강화

---


---

## 12. 응답 검증 및 자기 보정 시스템
### 📁 모듈: `evaluator.py`, `self_verifier.py`, `MetaCorrectionEngine.py`, `output_corrector.py`

---

### 📌 개요

이 계층은 생성된 응답 또는 사고 결과물이 다음 기준을 충족하는지 검사하고, 부족한 경우 수정합니다:

- 논리적 일관성
- 감정/문맥 정합성
- 신뢰도 / 정보 정확성
- LLM 응답 오류/왜곡 여부

---

### 🧠 구성 요소 및 역할

#### ▸ `TrustEvaluator` (`evaluator.py`)
- 응답의 **coherence, relevance, completeness, confidence** 등 4개 항목 평가
- 종합 trust score 계산 → 기준치 이하 시 보류

#### ▸ `SelfVerifier` (`self_verifier.py`)
- LLM 기반 자기비평 시스템
- SYSTEM 프롬프트를 활용하여 출력물에 대해:
  - 허위/오류 진술 여부
  - 논리 불일치
  - 감정 부적절 등 항목 분석

#### ▸ `OutputCorrector` (`output_corrector.py`)
- `SelfVerifier` 결과를 받아 LLM에게 **수정 명령 프롬프트 생성**
- 최소 수정 or 완전 재작성 판단

#### ▸ `MetaCorrectionEngine` (`MetaCorrectionEngine.py`)
- (미구현/예정) 자기 반성 루프에서 메타 차원에서 수정 트리거 가능

---

### 🔄 검증 및 수정 흐름 요약

```
[응답 생성]
   ↓
[TrustEvaluator 평가 → trust_score < 0.7 ? → 보정 루프 진입]
   ↓
[SelfVerifier → LLM 평가 결과 수집]
   ↓
[OutputCorrector → 수정 프롬프트로 재생성]
   ↓
[검증 통과 시 최종 응답 반환, 실패 시 discard or fallback]
```

---

### 🔗 연동 시스템

- `enhanced_main_loop.py`: 응답 후 자동 평가 트리거
- `meta_learning.py`: scar/drift 기반 correction 연계
- `prompt_architect.py`: 수정 반영 프롬프트 구성
- `cognitive_cycle.py`: 응답 재진입 루프 제어

---

### 📌 특징 요약

- 평가 항목은 단일 score가 아니라 다차원 기준 기반
- 검증 실패 시 자동 보정 + 메타 판단 가능
- LLM 출력의 drift/superficiality 문제를 자기 루프에서 차단

---


---

## 13. HarinLoad – 전체 시스템 실행 흐름 및 초기화 구조
### 📁 참조: `harinload.md`, `interface/harin_cli.py`, `core/cognitive_cycle.py`

---

### 📌 개요

HarinLoad는 HarinCore를 **명령어 기반 또는 API로 초기화**하고,  
하나의 인지 사이클 루프 전체를 실행하기 위한 흐름을 정의한 구성입니다.

> "입력 → 해석 → 상태 분석 → 사고 생성 → 판단 → 응답 생성 → 검증 → 저장"  
> 모든 과정이 모듈별로 분산된 구조로 이루어지며, 상태 기반 흐름(LangGraph)에 따라 작동합니다.

---

### 🧠 전체 인지 루프 구조 (9단계 요약)

| 단계 | 설명 | 주요 모듈 |
|------|------|-----------|
| 1. 입력 수신 | 사용자 CLI 또는 API 입력 수신 | `harin_cli.py`, `live_runner.py` |
| 2. 상태/컨텍스트 분석 | 세션/감정/리듬 기반 상태 구성 | `context.py`, `state_models.py` |
| 3. 감정/자극 해석 | 자극 유형, 감정 스코어 추출 | `emotion_system.py`, `stimulus_classifier.py` |
| 4. 기억 시스템 호출 | 유사 사고/기억 흐름 조회 | `integrated_memory_system.py`, `palantir.py` |
| 5. 사고 생성/메타 판단 | 다양한 사고 + 존재 기반 평가 | `thought_processor.py`, `meta_cognition_system.py` |
| 6. 응답/행동 생성 | 프롬프트 생성 + 시뮬레이션 실행 | `prompt_architect.py`, `action_system.py` |
| 7. 툴/API 호출 | 필요한 외부 정보 수집 | `web_search.py`, `llm_client.py` |
| 8. 검증 및 보정 | 자기 검증 및 보정 루프 진입 | `evaluator.py`, `self_verifier.py`, `output_corrector.py` |
| 9. 기억 저장 | 사고 노드(ThoughtNode)로 저장 | `memory_conductor.py`, `palantirgraph.py` |

---

### 🔄 실행 흐름도

```
[User Input]
   ↓
[input_interpreter → ParsedInput]
   ↓
[stimulus_classifier → Stimulus]
   ↓
[CognitiveCycle.run(stimulus)]
   ↓
    [context update]
    [emotion analysis]
    [memory retrieval]
    [reasoning/thought generation]
    [meta evaluation]
    [prompt creation]
    [action execution]
    [output evaluation]
    [memory storage]
```

---

### ⚙️ 초기화 예시 (`interface/harin_cli.py`)

```python
cycle = CognitiveCycle(config=...)
cycle.run(stimulus)
```

---

### 📌 특징 요약

- LangGraph 기반 상태 전이로 각 단계 자동 호출
- 감정/리듬/자아 기반 루프 → 메타 평가 → 검증 보정까지 전환 가능
- 프롬프트/응답은 기억과 사고 흐름이 통합된 구조
- CLI 또는 API에서도 동일한 인지 흐름으로 작동

---


---

## 14. 계층별 평가 및 로깅 시스템 구조
### 📁 관련 모듈: `evaluator.py`, `meta_cognition_system/`, `emotion_system.py`, `action_system.py`, `palantirgraph.py`

---

### 📌 개요

이 섹션은 HarinCore의 각 인지 계층 및 시스템 컴포넌트에서 수행되는  
**평가(metric) 항목** 및 **로깅 포인트(log point)**를 정리합니다.

모든 평가 값은 다음과 같은 목적에 사용됩니다:

- 사고/응답 품질 판단
- 루프 재진입 조건 결정
- drift/scar 경고 트리거
- Memory/Persistence 기록 요소 추출

---

### 📊 평가 및 로깅 항목 요약

| 계층 | 평가 항목 | 평가 모듈 | 저장 위치 |
|------|------------|------------|------------|
| 감정/자극 | `emotion_type`, `intensity`, `confidence` | `emotion_system.py` | `context.py`, `UserContext` |
| 사고 생성 | `complexity`, `argument_depth`, `trust_score` | `metacognition.py` | `meta_evaluator.py`, ThoughtNode.meta |
| 사고 리듬 | `tone_force`, `drift_trigger` | `input_interpreter.py` | `context.py`, `ParsedInput` |
| 응답 평가 | `coherence`, `relevance`, `completeness`, `confidence`, `score` | `evaluator.py`, `self_verifier.py` | `evaluation_history`, `last_verdict` |
| 행동 시뮬 | `agent_type`, `performance_score`, `collaboration_history` | `action_system.py` | `AgentState` |
| 기억 통합 | `importance`, `p`, `universe`, `relationship_strength` | `palantirgraph.py` | `ThoughtNode`, `PalantirGraph` |
| 모순 감지 | `contradiction_type`, `severity` | `memory_conductor.py` | `Contradiction` |

---

### 🧠 평가 흐름도 예시

```
[입력]
   ↓
[emotion_system → 감정 평가]
   ↓
[meta_cognition → 사고 평가 + trust_score]
   ↓
[evaluator/self_verifier → 응답 신뢰도 평가]
   ↓
[output_corrector → 필요 시 수정]
   ↓
[palantir → memory 기록 시 p / meta 저장]
```

---

### 📌 로깅 포인트 정리

| 위치 | 로그 내용 | 대상 |
|-------|-----------|------|
| `UserContext.add_trace()` | 상태 변화, 감정 흐름 | 사용자의 세션 흐름 |
| `ActionSystem.log()` | 시뮬레이션 수행 결과 | agent ID, task 성과 |
| `MemoryConductor.save_node()` | 저장된 사고 정보 | MemoryProtocol |
| `MetaEvaluator.result` | 사고 경로 평가 결과 | path, decision, reason |

---

### 📦 활용 예시

- `trust_score < 0.6` → 루프 재진입 or meta 루프
- `Agent.performance_score < 0.4` → 다른 agent로 재할당
- `contradiction.severity > 0.8` → scar 또는 경고 메모리 전환

---


---

## 15. 동적 루프 전환 및 재구성 설계
### 📁 관련 모듈: `meta_evaluator.py`, `self_verifier.py`, `meta_learning.py`, `drift_monitor.py`, `cognitive_cycle.py`

---

### 📌 개요

HarinCore는 정적 사고 루프 구조를 넘어서  
**조건 기반 분기** 및 **자기진단 결과에 따른 루프 재진입**을 지원합니다.

이 섹션에서는 시스템이 어떤 조건에서 사고 경로를 바꾸고,  
루프를 재편성하거나 특정 계층을 skip/hold/retry하는지에 대해 설명합니다.

---

### 🔀 전환 조건 매커니즘

| 조건 | 루프 반응 | 설명 |
|------|-----------|------|
| `trust_score < 0.6` | Meta 루프 재진입 | 응답 품질 불충분 → 사고 재구성 |
| `rhythm.truth < 0.4` | 응답 hold | 프롬프트 리듬 불일치 시 보류 |
| 감정이 `불안`, `혼란` & score < 0.75 | delay 상태 전환 | 감정과 신뢰 미스매칭 |
| `SelfVerifier → invalid` | OutputCorrector 루프 진입 | 자기검증 실패 시 보정 시도 |
| `drift_detected == True` | Meta + ReEval 재진입 | 인지 흐름에 흔들림 발생 |
| `Contradiction.severity > 0.8` | Scar 저장, 루프 재평가 | 기억 간 충돌 → scar 분기 생성 |
| `Agent.performance_score < 0.5` | agent 재할당 or task 분산 | 실행 실패 or 모니터링 실패 시 |

---

### 🔁 루프 구조 재편 예시 (조건 기반 흐름도)

```
[CognitiveCycle]
   ↓
[TrustEvaluator → score < 0.6]
   ↓
[MetaCognition 재진입]
   ↓
[ThoughtNode 재생성 + PromptArchitect 재호출]
   ↓
[검증 통과 시 → 응답 or 저장]
```

---

### 🧭 상태 기반 조건 루프 구성 (LangGraph 스타일)

```python
if trust_score < 0.6:
    graph.add_edge("response_eval", "meta_reflect")
elif drift_trigger:
    graph.add_edge("response_eval", "drift_monitor")
elif output_verified is False:
    graph.add_edge("response_eval", "output_corrector")
else:
    graph.add_edge("response_eval", "memory_save")
```

---

### 📦 재구성 흐름 예시

| 루프 상황 | 트리거 | 다음 경로 |
|-----------|--------|------------|
| 응답이 불완전 | `score < 0.6` | 사고 루프 재시작 |
| 감정 불안정 | `emotion = 불안` | 리듬 조절 + delay |
| 모순/충돌 기억 | `Contradiction` | Scar 루프 또는 경고 메모리 생성 |
| 판단 미완 | `decision = hold` | 응답 보류 후 루프 유지 |

---

### 📌 요약

- HarinCore는 사고 흐름 중간에 조건 기반으로 **루프 재편성 가능**
- trust, emotion, contradiction, rhythm 등이 주요 trigger 역할
- LangGraph 조건부 전이 설계 기반으로 고차 루프 구성 지원

---


---

## 16. 내부 인지 시뮬레이션 및 잠재 루프 구조
### 📁 관련 모듈: `cognitive_cycle.py`, `action_system.py`, `palantir.py`

---

### 📌 개요

이 섹션은 HarinCore의 내부 사고 시뮬레이션과 잠재 루프(ghost/subconscious loop)에 대한 구조를 다룹니다.  
시스템은 명시적 행동 외에도 아래와 같은 숨겨진 사고 구조를 갖고 있으며:

- 각 사고 루프는 `tick` 단위로 분리되어 추적되고
- `GhostState`는 그때그때 생성된 감정/자극/의도/사고/결과들을 보관하며
- 모든 인지 흐름은 `Knoxel` 단위로 분해되어 쌓이고
- 특정 조건에서만 표면 행동으로 나타납니다

이러한 흐름은 사용자와의 상호작용 외에, **시뮬레이션/반성/잠재 상태 분석 루프**를 구성하는 핵심입니다.

---

### 🧠 16.1 GhostState: 인지 흔적 기록기

`GhostState`는 시스템의 각 인지 루프 (`tick`)에서 발생한 모든 심리/의사결정 요소를 보관하는 **중간 메모리 구조**입니다.

| 필드 | 설명 |
|------|------|
| `perception_result` | 현재 자극에 대한 감각 처리 결과 |
| `attention_focus` | 주의를 집중한 요소 |
| `selected_action_knoxel` | 해당 루프에서 선택된 행동 노드 |
| `emotions`, `intentions` | 감정 상태, 계획 의도 등 |
| `tick_id` | 시간 기반 사고 루프 번호 |
| `snapshot()` | 상태 스냅샷 저장 (개별 메모리 항목 생성 가능) |

→ GhostState는 **자기 사고 추적/비교용 히스토리**이며, 다음 루프의 `contextual_state`로 직접 전달됩니다.

---

### 🕒 16.2 Tick 기반 인지 루프 흐름

하린코어는 LIDA 인지 모델을 바탕으로 사고를 **tick 단위의 단일 처리 단위**로 간주합니다.

```python
tick_id += 1
state = initialize_tick_state(stimulus)
```

- `tick_id`는 사고 세션에서의 순서를 의미
- 각 tick은 감각 → 인식 → 판단 → 실행 흐름을 실행하고
- 결과적으로 `GhostState`와 `Knoxel`로 상태가 요약됩니다

#### ▸ tick 흐름의 가치

- 각 사고 결과를 독립적으로 관리
- scar/루프 재진입 시 이전 tick의 상태 비교 가능
- 기억/리듬/drift를 시간 순서대로 추적 가능

---

### 🧱 16.3 Knoxel: 사고 구조 단위

`Knoxel`은 하린코어에서 감정, 의도, 인식, 판단, 기억, 행동 등 사고의 구성 요소를 **블록화한 단위**입니다.

- `KnoxelList`로 누적되며 최근 10개 이상을 추적
- 각 tick별로 새로 생성된 `Intention`, `Feature`, `Stimulus`, `Action` 등이 여기에 포함됨

→ 이는 `PalantirGraph`로 전달되어 사고 흐름 그래프를 구성하며, 일부는 `ThoughtNode`로 전환됨

---


---

### 🎲 16.4 Monte Carlo Simulation: 행동 시뮬레이션 기반 예측

HarinCore는 실제 행동을 수행하기 전, `ActionSimulation` 객체를 사용하여  
**다수의 시뮬레이션 결과를 통해 평균적 성과를 예측**합니다.

#### ▸ 주요 구조
- `simulate_action(action, num_simulations=3)`  
  → `ActionSimulation` 리스트 반환

- `ActionSimulation` 필드 예시:
  - `simulation_id`
  - `action_type`
  - `ai_reply_content`
  - `simulated_user_reaction`
  - `predicted_ai_emotion`
  - `intent_fulfillment_score`
  - `emotion_score`, `cognitive_load`, `cognitive_congruence`

#### ▸ 작동 방식

```
for i in range(N):
    result = _run_single_simulation(action)
    simulations.append(result)

avg_score = sum(sim.score for sim in simulations) / N
```

→ 시뮬레이션 기반으로 행동의 리스크 예측 / 전략 비교 가능  
→ 메타 루프와 연동하여 low-score 행동은 보류 가능

---

### 🧘 16.5 Contemplation Mode: 내부 사고 루프

특정 상황에서는 HarinCore는 외부 행동 없이, **내면적 사고 루프**만 수행하는  
`InitiateInternalContemplation` 액션을 수행할 수 있습니다.

- 일반 행동 대신 `ActionType.InitiateInternalContemplation` 실행
- 감정/의도 상태 변화, Memory 요약, Meta 루프 트리거만 진행
- 외부 출력은 생략되거나 요약됨

#### ▸ 적용 예
| 조건 | 루프 동작 |
|------|------------|
| 감정이 `혼란`이고 trust 낮음 | Meta 루프 진입 + Contemplation |
| 판단이 `hold`일 때 | 외부 행동 생략 + 사고만 수행 |

---

### 👻 16.6 Phantom & Subconscious 흐름

HarinCore는 각 사고 흐름의 일부를 사용자에게 보여주지 않고  
내부적으로 누적하며, 필요 시에만 이를 의식 수준으로 올립니다.

| 개념 | 설명 |
|------|------|
| `Phantom` | 이전 사고 흐름 중 **표면에 드러나지 않은 정서/기억/신호** |
| `Subconscious` | 감정/의도/리듬 기반의 무의식적 사고 흐름 |
| `GhostState` | Phantom 흐름을 추적하는 컨테이너 |
| `snapshot()` | subconscious 상태를 Memory로 보존

→ 이 흐름은 다음 루프의 `attention candidates` 또는 `meta_learning` 평가의 입력이 됩니다.

---

### 🧠 종합 흐름 요약

```
[GhostState] ← 각 tick 상태 저장
[simulate_action()] ← 행동을 미리 시험
[Contemplation] ← 판단 지연 상황에서 내부 루프만 수행
[Phantom/Subconscious] ← 표면 아래의 감정/기억 흐름 추적
→ 필요 시 Memory로 승격 or scar 처리
```

---
