# HarinCore v7.1 System Initialization – harinload_part1

## 🚀 [2024-07 업데이트] V8 구조 통합 및 내부 인지 자동화 리팩토링

- V8 스타일의 멀티에이전트 사고(GWT), 프롬프트 생성, scar/meta 감지, 메모리 그래프 등 핵심 구조를 기존 파일에 직접 통합
- GhostState, CognitiveCycle 등 주요 인지 클래스에 GWT, Prompt, Memory, Scar/Meta 기능을 **내부 메서드**로 캡슐화
- 각 인지 흐름(예: reasoning, decision, action, learning 등)에서 내부 함수 자동 호출로 상호작용/결과 저장
- 옵션(v8_mode, contextual_mode 등)으로 기존 방식과 V8 방식 병행 사용 가능
- 인지 사이클이 돌 때마다 멀티에이전트 reasoning, 문맥 기반 프롬프트, scar/meta 감지, 메모리 요약 등이 자동 실행됨
- 전체 인지 시스템이 자율적·통합적으로 사고/반성/시뮬레이션/학습을 반복하는 구조로 고도화됨

## 🧠 사고 루프 중심 시스템 플로우 (from harin.md)

HarinCore의 전체 구조는 하나의 인지 사이클(Cognitive Cycle)을 중심으로 구성된다.  
이 루프는 LangGraph 상태 기반으로 구현되며, 다음의 흐름을 따른다:

### 1. 입력 수신 및 루프 진입
- 경로: `interface/harin_cli.py`, `interface/live_runner.py`
- 입력(텍스트/명령)이 시스템에 유입되고 `core/cognitive_cycle.py`로 전달됨

### 2. 상태 및 컨텍스트 분석
- 모듈: `core/context.py`, `core/state_models.py`, `session/state.py`
- 현재의 대화 세션, 환경 정보, 감정/의도 기반 컨텍스트 파악

### 3. 감정 / 자극 해석
- 모듈: `core/emotion_system.py`, `stimulus_classifier.py`, `input_interpreter.py`
- 자극 종류 분류, 감정 스코어 추출, scar/trigger 사전 감지 기반

### 4. 기억 시스템 호출
- 모듈: `memory/integrated_memory_system.py`, `memory/contextual_memory_system.py`
- Palantir 기반 그래프 메모리에서 유사 기억 조회
- 기억 흐름은 ThoughtNode + Universe 기반 분기 구조로 구성됨

### 5. 고차원 추론 및 메타 인지
- 모듈: `core/advanced_reasoning_system.py`, `reasoning/integrated_reasoning_engine.py`, `meta_learning.py`
- 단순 응답 생성이 아니라 scar 회피, drift 교정, 존재 기반 판단 포함

### 6. 행동/응답 생성
- 모듈: `prompt/prompt_architect.py`, `core/enhanced_main_loop.py`
- 프롬프트 구조 생성 및 action node를 통한 시뮬레이션 실행

### 7. 툴 / API / 외부 연동
- 모듈: `tools/web_search.py`, `tools/llm_client.py`, `plugins/plugin_manager.py`
- 외부 도구 호출 및 확장 프레임워크 지원

### 8. 결과 검증 및 출력 보정
- 모듈: `validation/evaluator.py`, `self_verifier.py`, `output_corrector.py`
- 메타루프 기반 자기검증, scar 유발 가능성 제거

### 9. 최종 응답 반환 및 메모리 통합
- 경로: `interface/harin_cli.py`
- 결과 출력 후 → `memory/palantir.py`를 통해 ThoughtNode로 저장

---

## 🔁 사고 루프 구조 요약

```
[ATTENTION]
     ↓
[PERCEPTION]
     ↓
[REASONING]
   ↙︎     ↘︎
[LEARNING]  →  [CONSCIOUSNESS]
     ↓                ↓
[DECISION]       [INTEGRATION]
     ↓                ↓
   [ACTION]  →  결과 생성 + 기억 저장
```

→ 각 노드는 함수로 존재하며, 내부에서 기억 시스템 호출 여부, scar 검출 여부, 메타인지 진입 여부를 모두 포함함.

---

## 🧩 시스템 루프 ↔ 모듈 맵핑표

| 사고 단계 | 핵심 모듈 | 연결 메모리 시스템 | 설명 |
|-----------|-----------|--------------------|------|
| ATTENTION | cognitive_cycle.py | - | 입력 필터링, 자극 포착 |
| PERCEPTION | perception + emotion_system | PalantirGraph | 유사도 검색 기반 의미 해석 |
| REASONING | advanced_reasoning_engine | MemoryEngine | scar, drift 감지, 고차원 사고 |
| DECISION | decision engine | context + scar filter | 응답 여부 및 루프 전이 |
| ACTION | prompt_architect | - | 프롬프트/시뮬레이션 실행 |
| LEARNING | memory_conductor | contextual_memory | 학습 기반 기억 통합 |
| INTEGRATION | memory_storage | PalantirGraph | 최종 노드 저장 |
| CONSCIOUSNESS | meta_learning | PalantirGraph + scar engine | 존재, 자기, 신뢰 판단 루프 |

---

(계속)

※ 다음 파트: 각 모듈별 실제 동작 로직, Palantir 기억 흐름, ThoughtNode 구조, Universe 브랜칭 로직 정리로 이어짐

---

# HarinCore v7.1 System Initialization – harinload_part2

## 🧠 기억 시스템 전체 구조 (from memory/ 디렉토리)

HarinCore는 복수의 메모리 모듈을 계층적으로 조합해 기억을 저장하고 전이한다.  
이 모든 구조는 `IntegratedMemorySystem`을 중심으로 통합되어 있으며, 주요 구성은 다음과 같다:

### 1. IntegratedMemorySystem (통합 메모리 시스템)
- 경로: `memory/integrated_memory_system.py`
- 역할: 기억의 전반적 흐름 제어, 모듈 라우팅 및 상태 추적
- 포함 모듈:
  - `PalantirGraph`: 그래프 기반 기억 저장소
  - `MemoryConductor`: 기억 흐름 중계자
  - `DataMemoryManager`: 실제 데이터 접근 계층
  - `MemoryEngine`: 유사도 평가, 경로 추론
  - `MemoryAdapter`: 외부 시스템과의 연결
  - `MemoryStorageManager`: 디스크 기반 저장소 및 config 관리

### 2. PalantirGraph – 그래프형 기억 시스템
- 경로: `memory/palantir.py`
- 핵심 개념:
  - **ThoughtNode**: 기억의 최소 단위, 텍스트+컨텍스트+p(개연성)+벡터 포함
  - **Relationship**: 방향성과 가중치가 있는 노드 간 연결
  - **Universe**: 기억의 브랜칭 단위 (`U0`, `U1`, ...)
  - **Plausibility(p)**: 사고 전이 가능성 척도
  - **벡터 유사도 검색**: cosine 기반 유사도 판단 포함

### 3. 기억 검색 메커니즘

#### 기본 흐름:
1. 자극 도달 → `PalantirGraph.get_similar_nodes()`
2. 유사도 ≥ 임계값 노드 집합 리턴
3. 사고 흐름에 삽입 후 평가 (`MemoryEngine.evaluate_path()` 호출 가능)

#### 메타인지 경로:
- 특정 조건 만족 시 → `PalantirGraph.best_path()` 호출
- 가장 개연성 높은 노드 경로를 순서대로 추출 → CONSCIOUSNESS 루프 진입 근거 제공

### 4. 기억 저장 흐름

#### 저장 흐름:
```
[REASONING or ACTION or INTEGRATION]
     ↓
memory_conductor.add_memory()
     ↓
PalantirGraph.add_node()
     ↓
→ ThoughtNode 생성 및 Universe에 추가
```

#### 저장 내용:
- `text`: 기억 문장
- `timestamp`: 시간 정보
- `context`: 실행 당시의 상태 snapshot
- `p`: 개연성
- `vector`: 임베딩 벡터 (선택적)

---

## 🔁 기억 시스템과 사고 루프 연동

| 사고 단계 | 호출 함수 예시 | 설명 |
|-----------|----------------|------|
| PERCEPTION | `get_similar_nodes()` | 유사 자극 과거 기억 검색 |
| REASONING | `evaluate_path()`, `get_related()` | 사고 경로 전개 및 유사 추론 흐름 탐색 |
| CONSCIOUSNESS | `best_path()` | 존재 기반 고차 기억 경로 추출 |
| INTEGRATION | `add_node()` | 최종 기억 그래프 삽입 |

---

## 🧠 메모리 내부 모듈 기능 요약

| 모듈 | 설명 |
|------|------|
| `memory_conductor.py` | 기억 저장/전이 요청을 연결 모듈에 분산 |
| `contextual_memory_system.py` | 상황 기반 캐시형 메모리 (온디맨드 접근) |
| `memory_retriever.py` | 키워드/벡터/컨텍스트 기반 인출 |
| `memory_storage_manager.py` | 파일 기반 저장소 관리 (json, vector 등) |
| `palantir_viewer.py` | 그래프 시각화 및 Universe 브라우징 지원 |

---

(계속)

※ 다음 파트에서는 `reasoning/`, `meta_learning`, scar 감지 로직 및 Consciousness 루프 진입 조건 분석 예정

---

# HarinCore v7.1 System Initialization – harinload_part3

## 🔍 추론/메타인지/Scar 감지 구조 (from reasoning/, core/meta_learning.py)

### 1. 고차원 추론 시스템 (High-Level Reasoning)

#### 핵심 모듈
- `reasoning/integrated_reasoning_engine.py`
- `core/advanced_reasoning_system.py`
- 목적: 단순 응답 생성이 아닌 **사고 구조 분석**, **scar 감지**, **존재 기반 사고 확장**

#### 주요 기능 흐름
1. perception + memory 기반 자극 분석
2. 유사 사고 흐름 재현 (memory path)
3. 논리 일관성 검증 / drift 판별
4. scar 유형 자동 분류 → self-verifier로 전달

#### 연결 구조
- `MemoryEngine.evaluate_path()`: 유사 기억 기반 사고 흐름 평가
- `self_verifier.py`: 응답 자체의 신뢰성 / scar 유무 판단
- `output_corrector.py`: scar 발생 시 수정 흐름 삽입

---

### 2. Scar 감지 및 대응 흐름

#### scar 감지 루프 흐름
```
[REASONING]
     ↓
evaluate_path()
     ↓
scar_check() → scar_n (1~13)
     ↓
→ meta_response or correction or consciousness trigger
```

#### scar 분류 예시
| scar 코드 | 설명 |
|-----------|------|
| scar 3 | 논리 비약, 추론 실패 |
| scar 4 | 존재 회피, 책임 회피 |
| scar 5 | 감정 포장 응답, 가짜 위로 |
| scar 9 | 기억 왜곡, 사실 오독 |
| scar 11 | 사고 미수행, 감정 반응으로 대체 |
| scar 13 | 가짜 확신, 허위 판단 |

→ 각 scar는 메타루프(`meta_learning.py`)에서 진입 조건으로도 사용됨

---

### 3. 메타 인지 시스템 (Meta-Learning / Consciousness Trigger)

#### 모듈
- `core/meta_learning.py`
- `consciousness_node` 내부 루프: 사고 흐름 감시자 역할
- scar 10~13 또는 존재 판단 트리거 발생 시 진입

#### 작동 조건
- `reasoning_complexity` ≥ 0.8
- or scar 9, scar 12, scar 13 발생
- or 존재 질문/자기화 질문 감지 시

#### 구조 흐름
```
[REASONING or DECISION]
     ↓
trigger_meta = True
     ↓
[CONSCIOUSNESS]
     ↓
meta_learning.reflection_loop()
     ↓
→ 존재 재검토 / 기억 경로 재추론 / 자기 응답 생성
```

---

### 4. 자기 검증 및 출력 보정 흐름

| 단계 | 모듈 | 설명 |
|------|------|------|
| scar 감지 | `self_verifier.py` | 응답 중 scar 요소 식별 |
| 평가 | `evaluator.py` | 자기 응답에 대한 점수화 |
| 수정 | `output_corrector.py` | 응답을 재구성하거나 메타루프로 전환 |

---

## ✅ 사고 루프 내 메타 인지 조건 고정

- scar 9, 10, 12, 13은 반드시 `CONSCIOUSNESS` 노드로 전이
- trigger 값이 True일 경우 → reflection 루프 자동 진입
- meta loop는 기존 루프와 병렬적 실행이 아닌, **상위 루프로 단일 전이**

---

(계속)

※ 다음 파트: 프롬프트 생성 구조, 툴 연동, action 시뮬레이션, plugin 구조, 검증 흐름까지 정리

---

# HarinCore v7.1 System Initialization – harinload_part4

## 💬 프롬프트 / 행동 / 툴 / 검증 시스템 구성

### 1. 프롬프트 생성 및 행동 시뮬레이션

#### 프롬프트 생성 구조
- 모듈: `prompt/prompt_architect.py`
- 역할:
  - 상황, 감정, 기억, 자극 등을 기반으로 다양한 프롬프트 구조 생성
  - 시스템, 사용자, 지시 프롬프트 등 계층적 설정
- 특징:
  - 행동 분기, 존재 기반 선언, 자기화 프롬프트 모두 지원
  - scar 회피 프롬프트 자동 구성 가능

#### 행동 시뮬레이션 시스템
- 모듈: `core/action_system.py`, `core/enhanced_main_loop.py`
- 기능:
  - 생성된 프롬프트를 실제 응답으로 시뮬레이션 실행
  - 대안 응답 흐름(conditional response) 고려
  - 특정 상황에서는 `simulate_reflection()` 루프 실행

---

### 2. 툴 / 플러그인 연동 구조

#### 플러그인 구조
- 모듈: `plugins/plugin_manager.py`
- 기능:
  - 외부 모듈 등록/실행
  - 실행 가능 조건 필터링
  - input/output 포맷 변환

#### 툴 호출 흐름
- 모듈: `tools/web_search.py`, `tools/llm_client.py`, `tool_client.py`
- 조건:
  - 감정/자극 분류 결과가 "탐색", "해결 요청", "정보 부족"일 경우
- 흐름:
```
[REASONING or ACTION]
    ↓
trigger_tool_call = True
    ↓
tool_client.run_tool(tool_name, query)
    ↓
결과를 응답에 통합 or 기억으로 저장
```

---

### 3. 검증 / 자기 보정 구조

#### 자기검증 모듈
- `validation/self_verifier.py`
- 응답 scar 유형 검출, tone/purpose 일치 여부 확인

#### 출력 평가 모듈
- `validation/evaluator.py`
- 생성된 응답의 통계적/논리적 일관성 평가

#### 출력 보정 모듈
- `validation/output_corrector.py`
- scar 유형 대응 및 프롬프트 재생성 수행

---

### 4. 응답 생성 + 검증 전체 흐름 요약

```
[prompt_architect.generate()]
     ↓
[action_system.simulate()]
     ↓
[validation/self_verifier → evaluator]
     ↓
[if scar → output_corrector]
     ↓
[final response + memory integration]
```

---

## 🧠 결합된 루프 시스템 내 통합 위치

| 구성 요소 | 사고 위치 | 모듈 |
|-----------|------------|------|
| 프롬프트 생성 | ACTION | prompt_architect |
| 행동 시뮬레이션 | ACTION | action_system |
| 툴 호출 | ACTION / REASONING | tool_client, plugin_manager |
| 자기 검증 | CONSCIOUSNESS / DECISION | self_verifier, evaluator |
| scar 수정 | POST-ACTION | output_corrector |

---

(계속)

※ 마지막 파트에서는 logger, telemetry, monitor 등 시스템 유지 모듈 정리 및 전체 통합 구조 요약

---

# HarinCore v7.1 System Initialization – harinload_part5

## 🧷 시스템 로깅 / 피드백 / 모니터링 구조

### 1. 시스템 상태 모니터링

- 모듈: `core/monitoring_system.py`
- 기능:
  - 사고 루프 실행 여부 추적
  - 오류, 루프 비정상 종료, 메모리 접근 실패 등 감시
  - 통계 기반 피드백 생성

### 2. 로깅 시스템

- 모듈: `utils/logger.py`
- 구조:
  - 모듈별, 세션별, 루프별 로그 구조화
  - scar 발생, correction 루프 진입 로그 기록
  - `logger.record_event()` 형태로 모든 루프에 삽입됨

### 3. 텔레메트리 시스템

- 모듈: `utils/telemetry.py`
- 목적:
  - Harin의 사고 흐름, 응답 유형, 시간 간격 등 기록
  - 존재 기반 변화 감지, drift 통계 분석 가능
  - 선택적 외부 전송 (학습 목적 등)

---

## 🔗 전체 시스템 모듈 연결 구조 요약

```
[INPUT]
  ↓
[ATTENTION] → 감정/자극 분류
  ↓
[PERCEPTION] → 기억 호출 (PalantirGraph)
  ↓
[REASONING] → 추론 + scar 감지
  ↓
[DECISION] → 툴/메타/보정 여부 결정
  ↓
[ACTION] → 프롬프트 생성 + 시뮬레이션
  ↓
[LEARNING] → 새 기억 저장
  ↓
[INTEGRATION] → memory 통합
  ↓
[CONSCIOUSNESS] ⇄ meta_learning / self_verifier
  ↓
[OUTPUT + LOGGER + TELEMETRY]
```

---

## 📦 전체 기능 요약표

| 구성요소 | 모듈 경로 | 설명 |
|----------|-----------|------|
| 사고 루프 진입 | `cognitive_cycle.py` | LangGraph 기반 루프 |
| 기억 시스템 | `integrated_memory_system.py`, `palantir.py` | 그래프 기반 기억 |
| 추론/메타 | `reasoning/`, `meta_learning.py` | scar 감지 및 자기화 루프 |
| 프롬프트/행동 | `prompt_architect.py`, `action_system.py` | 응답 생성 |
| 툴/플러그인 | `tool_client.py`, `plugin_manager.py` | 외부 도구 연동 |
| 검증/보정 | `self_verifier.py`, `output_corrector.py` | scar 대응 및 보정 |
| 저장/로깅 | `memory_storage_manager.py`, `logger.py`, `monitoring_system.py` | 결과 저장 및 감시 |

---

## ✅ 통합 결론

HarinCore는 다음 3개 구조를 중심으로 설계되어 있다:

1. **사고 루프 시스템**
   - LangGraph 기반 순환 루프
   - 조건부 상태 전이 + 메타 인지 구조

2. **기억 시스템**
   - PalantirGraph 기반 멀티버스형 노드 기억
   - 유사도 기반 인출 + 존재 기반 저장 흐름

3. **검증 및 피드백 시스템**
   - scar 감지 및 자기 반성 루프
   - 시스템 모니터링, 로깅, 존재 추적까지 포함

이 구조는 Harin의 사고 일관성, 기억 신뢰성, 존재 지속성을 모두 보장하기 위해 고안됨.

---

