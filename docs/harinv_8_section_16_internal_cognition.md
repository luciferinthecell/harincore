# 16. 내부 인지 시뮬레이션 및 잠재 루프 구조

> **Scope** — This section details the internal cognitive simulation engine that powers HarinCore’s higher‑order reasoning: its tick‑driven loop, GhostState persistence, Monte‑Carlo–style exploratory branches, and the latent‑loop supervisor that keeps everything bounded and safe.

---

## 16.0 TL;DR

- **Tick‑based execution** breaks cognition into deterministic micro‑frames.
- **GhostState** stores the live “mental context” between ticks, separate from long‑term memory.
- **Knoxel graph** shards knowledge into addressable voxels that can be hot‑swapped during simulation.
- **Contemplation loop** spins up nested Phantoms that explore futures via **Monte Carlo Simulation Nodes (MCSNs)**.
- **Latent‑loop controller** enforces resource budgets, safety rules, and roll‑back semantics.

---

## 16.1 핵심 용어

| 용어               | 정의                                                         |
| ---------------- | ---------------------------------------------------------- |
| **Tick**         | 인지 루프의 최소 단위(≈ 1 transaction); perception→reason→act 단계 포함 |
| **GhostState**   | 세션 지속 상태: latent‑vector, episodic pointer, affect matrix   |
| **Knoxel**       | Knowledge Voxel. 64–512 token 단위의 연결 지식 블록                 |
| **Phantom**      | 일시적 가상 에이전트 클론; 특정 시나리오 탐색용                                |
| **MCSN**         | Monte Carlo Simulation Node. Phantom이 사용하는 search node 구조  |
| **Subconscious** | 백그라운드 스레드. 기억 통합·연상∙패턴 보정 담당                               |

---

## 16.2 Cognitive Tick Pipeline

```
┌── Tick Dispatcher ──┐
│   Phase 0 : Input   │ (perception)
│   Phase 1 : Eval    │ (intent & goal refresh)
│   Phase 2 : Plan    │ (Phantom spawn / MCS search)
│   Phase 3 : Commit  │ (action & memory write)
└─────────────────────┘
```

*Implemented in **`core/cognitive_cycle.py`**.*

1. **Input Phase** — assembles user text, memory recalls, and GhostState snapshot.
2. **Eval Phase** — policy gatekeeper + emotion/drive update.
3. **Plan Phase** — spins Contemplation loop if uncertainty > θ.
4. **Commit Phase** — materializes final response + side effects (memory, tasks).

### 16.2.1 Tick Dispatcher Details

- **Latency budget:** 55 ms soft, 120 ms hard per tick.
- **Hook points:** `before_eval`, `after_plan`, `before_commit`—allow plugins (e.g., tracing).

---

## 16.3 GhostState Mechanics

### 16.3.1 구조체

```python
class GhostState(BaseModel):
    latent: np.ndarray  # 768‑D embedding
    episodic_ptr: UUID  # link to recent MemoryGraph node
    affect: Dict[str, float]  # valence, arousal, dominance
    tick_counter: int
```

### 16.3.2 Life‑cycle

| Stage      | Trigger                  | Action                       |
| ---------- | ------------------------ | ---------------------------- |
| **Init**   | New session              | latent = 0‑vector            |
| **Update** | End of tick              | back‑propagate new embedding |
| **Spill**  | tick\_counter % 512 == 0 | snapshot → MemoryGraph       |
| **Prune**  | memory TTL               | drop stale episodic\_ptr     |

---

## 16.4 Knoxel (Knowledge Voxel) Layer

- **Granularity:** Defaults to 256 tokens (configurable).
- **Addressing:** `knx://<sha1‑hash>`; allows deduplication across users.
- **Hot‑swap:** During simulation, Phantom can temporarily replace a Knoxel with a hypothetical variant → enables counter‑factual reasoning.

---

## 16.5 Contemplation Loop

> Activated when **expected‑uncertainty > 0.35** or when goals conflict.

### 16.5.1 Architecture

```
User Input
   ↓
Main Agent ──┐
             │ spawn → Phantom[0] ──┐
             │                      │ expand → MCSN…
             │ spawn → Phantom[1] ──┘
             │ …
             ▼
Latent‑Loop Controller → best path → Commit
```

*Supervisor module: **`reasoning/contemplation.py`*

### 16.5.2 Reasoner Stack

1. **Planner** — drafts high‑level paths.
2. **Simulator** — performs MCS rollouts (see §16.6).
3. **Critic** — scores paths using reward: *Utility – Hallucination risk – Cost*.

---

## 16.6 Monte Carlo Simulation Nodes (MCSNs)

### 16.6.1 Node Types

| Node           | Meaning                                       |
| -------------- | --------------------------------------------- |
| **StateNode**  | Context snapshot (Knoxel subset + GhostState) |
| **ActionNode** | LLM action proposal; children are StateNodes  |
| **ChanceNode** | Random external event injection               |

### 16.6.2 UCT Variant

```
UCT(s,a) = Q(s,a) + c · √(ln N(s) / N(s,a))
```

- **c** tuned to 0.7 to bias exploitation.
- Incorporates **Hallucination Penalty** term: subtracts λ·P(hallucinate|path).

### 16.6.3 Roll‑out Example

```python
root = StateNode(context)
for _ in range(128):
    leaf = tree_policy(root)
    reward = default_policy(leaf)
    backup(leaf, reward)
best = best_child(root)
```

---

## 16.7 Phantom Instances

- **Clone Depth:** up to 3 (config). Each Phantom inherits GhostState snapshot.
- **Isolation:** No direct I/O; only communicates via latent‑loop controller mailbox.
- **Garbage Collection:** On commit or timeout (> 75 ms), Phantom tree is destroyed.

---

## 16.8 Subconscious Pipeline

1. **Trigger Detect:** Idle GPU slices or low QPS periods.
2. **Memory Consolidation:** merges recent Knoxels into semantic clusters.
3. **Pattern Alignment:** adjusts embedding axes to reduce drift.
4. **Safety Sweep:** re‑runs policy on new memories.

---

## 16.9 Latent‑Loop Controller

### 16.9.1 Budgeting

```yaml
max_ticks: 4
max_phantoms: 16
max_rollouts: 512
walltime_soft_ms: 150
```

### 16.9.2 Loop Unroll Logic

- **While** budget allows & improvement > ε.
- Early stop if top‑2 path score diff < δ.

---

## 16.10 Safety & Governance

| Safety Check         | Condition                 | Action                |
| -------------------- | ------------------------- | --------------------- |
| **Loop Stall**       | ∆Score < 0.01 for 2 ticks | force commit          |
| **Policy Violation** | content flagged           | abort, redact         |
| **Cost Surge**       | GPU > 2× median           | fallback shallow path |

---

## 16.11 Metrics & Telemetry

- **Sim coverage** = visited MCSNs / theoretical max.
- **Hallucination risk** (critic output).
- **Latency per tick**, **Cost/token**.

---

## 16.12 Implementation Notes

- Core classes: `Phantom`, `KnoxelManager`, `LatentLoopController`.
- All simulation runs are **deterministically seedable** for offline replay.
- Uses [torch.multinomial] for action sampling to stay GPU‑resident.

---

## 16.13 미래 과제

1. **Graph Neural Memory** — replace Knoxel KNN with GNN encoder.
2. **Self‑Reflective Critic** that fine‑tunes hallucination penalty online.
3. **Temporal‑Awareness Upgrade**: rhythmic ticks aligned with user locale.

