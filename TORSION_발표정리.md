# TORSION — 발표 정리 (교수님 보고용)

> **한 줄 정의**
> **TORSION은 "의미론적 결함(Semantic Fault)"을 시험 신호(excitation signal)로 주입하여,
> 자율주행 파이프라인 내부에서 오류가 어느 표현 경계에서 증폭·감쇠되는지를 경험적으로 식별(System Identification)하는 프레임워크다.**
>
> - 결함 주입(Fault Injection) = **도구**
> - 오류 전파(Error Propagation) = **관찰 대상**
> - 특성화(Characterization) = **결과**
> - (경험적) 시스템 식별 = **방법론**

---

## 1. 배경 & 기존 연구의 한계

자율주행 시스템은 여러 **중간 표현(intermediate representation)**을 거쳐 동작한다:

```
Sensor → Perception → Tracking → Prediction → Planning → Control
                 (object-set)   (cost-map / occupancy / BEV feature)
```

**기존 연구의 세 갈래와 각각의 한계:**

| 접근 | 무엇을 하나 | 한계 |
|---|---|---|
| **저수준 Fault Injection** (bit-flip, stuck-at) | 하드웨어/비트 수준 결함 주입 | 표현의 **의미(semantics)** 수준 결함을 못 다룸. 구조가 깨져 비현실적 |
| **Adversarial Attack** (PGD 등) | 최악의 입력 섭동 | 최악·비해석적. 표현의 **구조적 유효성(contract)**을 위반 |
| **안전성 평가** | fault → collision (블랙박스) | 결과만 봄. **어디서·왜** 오류가 위험으로 커지는지 설명 못 함 |

**공백:** 표현 스택에 **구조는 보존하면서 의미만 비트는(contract-preserving)** 통제된 결함을 주입하고,
그 오류가 표현 경계를 넘어 **어떻게 전파되는지 체계적으로 규명**하는 프레임워크가 없다.

> **TORSION의 위치:** 이 공백을 메운다. semantic fault를 excitation으로 써서 파이프라인의 전파 특성을 식별한다.
> (전통 fault injection과 adversarial attack 사이에 있는 **새로운 관점 = 표현 공간의 의미론적 섭동**)

---

## 2. 핵심 개념

### 2.1 Semantic Fault & Representation Contract
- **Semantic Fault** = 표현의 **구조적 계약(개수·클래스·track ID, cost 값 범위, 주행가능 위상, 텐서 형상)은 보존**하고
  **의미만 왜곡**하는 결함. (예: 탐지된 차량을 여전히 "유효한 1대"로 두되 위치·속도·방향만 비틈 → tracking drift / calibration error를 모사)
- **핵심:** 계약을 지키므로 결함이 붙은 표현은 여전히 "정상적인" 표현이다 → 시스템이 정상 입력처럼 처리 → 전파를 관찰할 수 있다.
- 계약을 **위반**하는 결함(Gaussian noise 등)은 **비교용 baseline**으로만 사용.

### 2.2 왜 "System Identification"인가
제어/시스템 이론에서 시스템을 알기 위해 **알려진 신호(excitation)를 넣고 출력(response)을 관찰**한다.
우리는 semantic fault를 excitation으로 주입하고, 각 표현 경계에서의 응답(gain)을 측정한다.
특히 **결함 크기를 sweep**하며 응답을 재는 것은 문자 그대로 system identification이다.

> **정직성 원칙:** 우리는 **경험적 gain·선형성**을 식별하지, 형식적 parametric 모델(Laplace/state-space)을 세우지 않는다.
> → 논문에서 **"empirical / SI-inspired characterization"**, 강한 용어("transfer function")는 **"propagation response"**로 표현.

---

## 3. 방법

### 3.1 통합 파이프라인 (측정의 척추)
```
Semantic Fault
   │  주입 지점을 바꿔가며(excitation location)
[object] → [prediction] → [cost-map] → [plan] → [control] → Safety
```
- 세 표현(object-set / cost-map / prediction)에 표현별 자연스러운 연산자로 주입
- 하류(planner·control·safety)는 **공유** → 공정 비교
- 결함 크기는 하류 plan 편차(budget)로 **matched** → 표현 간 비교 가능

### 3.2 정의한 전파 지표 (우리 프레임워크 내 분석 도구)
| 지표 | 의미 |
|---|---|
| **Interface Gain** | 경계 전후 오차 비 (>1 증폭, <1 감쇠) |
| **CIS** (Critical Interface Score) | 주입 경계의 안전 임계도 (안전저하 / 주입크기) |
| **FAR / reach-safety** | 안전 저하 정도 / 결함이 safety까지 도달한 비율 |

> ⚠️ 이 지표들은 **우리가 이 연구를 위해 정의한 도구**다. "정의했고 이를 통해 관찰했다"까지 주장하며, "일반 법칙"으로 과장하지 않는다.

---

## 4. 무엇을 했나 (실험 전체)

| 단계 | 내용 | 목적 |
|---|---|---|
| Phase A | 전파 지표 정의·측정 (CIS, gain) | WHAT: 어느 경계가 임계적인가 |
| Phase B | **Prediction 단계 추가** (현실적 파이프라인) | 실 자율주행 체인 반영 |
| Phase A+ | **메커니즘 규명 (M1/M2/M3)** | WHY: 왜 증폭·감쇠하나 |
| Task 1 | 진폭 sweep → 선형/비선형 응답 분류 | 시스템 식별(전달 특성) |
| Task 2 | Failure Taxonomy (대표 전파 경로) | 결함→실패 경로 |
| Task 3 | planner 선택 argmin↔softmax 통제 | 인과 격리 |
| ① | 구조가 다른 planner (potential-field) | planner 일반성 |
| ② | **nuPlan 실데이터 open-loop** (실 지도+실 agent) | 실데이터 일반화 |
| 보조 | CARLA 450 에피소드 실측 / 실제 InterFuser BEV | 실물리·실모델 검증 |

---

## 5. 결과

### 5.1 측정 (WHAT) — 어느 경계가 임계적인가 (합성, 30 seeds)
| 시나리오 | object 주입 | prediction 주입 | cost-map 주입 |
|---|---:|---:|---:|
| leading_vehicle | 0.510 | 0.032 | **1.377** |
| pedestrian_crossing | 1.226 | 1.025 | 0.950 |
| **전체** | 0.684 | 0.340 | **1.021** |

→ **cost-map 경계가 가장 안전 임계적**. prediction 임계도는 **시나리오 의존**(보행자 횡단에서 지배적).

### 5.2 설명 (WHY) — 세 가지 메커니즘 ⭐ (이 연구의 핵심)
| | 경계 | 구조적 원인 | 정량 근거 |
|---|---|---|---|
| **M1** | 래스터화 (object→cost) = **감쇠기** | 커널·격자로 뭉개는 many-to-one 투영 | 국소 Jacobian J = 0.02~0.04 **≪ 1** |
| **M2** | cost→plan = **증폭기** | **sampling-argmin 결정경계 스위칭** (미분이 아니라 이산 점프) | 결정여유 최소 구간에서 plan 편차 **27배**, 승자 교체율 **24배**, Spearman 0.52 |
| **M3** | object→prediction = **증폭기** | 등속 예측이 속도 오차를 시간축으로 적분 | ∂pred/∂v = t, 해석해=실측 정확 일치, horizon에 비례 |

> **의의:** 단순히 "관찰했다"가 아니라 **왜 그런지를 구조적으로 설명**. SCI 상위권이 요구하는 지점.

### 5.3 시스템 특성화 (Task 1, 2)
- **Propagation Response:** cost→plan은 **비선형**(입력 크기에 따라 gain 변함), plan→control·object→prediction은 **선형**.
  → **파이프라인 = 선형 전달소자 사이에 planner(argmin)라는 비선형 스위칭 소자.**
- **Failure Taxonomy (810 runs):** 결함의 **76~85%가 planner 스위칭을 경유**; 지배적 유발 실패 = **급제동(phantom brake)**;
  충돌률 gradient = **cost-map 11.9% > object 5.2% > prediction 0.4%**.

### 5.4 인과·일반성 (Task 3, ①)
- **argmin↔softmax:** 선택을 부드럽게 하면 스위칭 **−36%**, object/cost 충돌 **0으로 소거**
  → **이산 선택 = 위험한 증폭기 / 연속 선택 = 안전** (설계에 actionable).
- **구조가 다른 planner(potential-field):** cost-map 최고 임계·planner 비선형은 **planner 불변**;
  그러나 "planner-switch 게이트웨이"는 **argmin 계열 특이적**(게이트웨이율 0.85→0.14).

### 5.5 실데이터 일반화 (nuPlan open-loop, 실 지도+실 agent, **10,440 runs**) ⭐
| 주장 | 실데이터 재현 | 근거 |
|---|---|---|
| **래스터화 감쇠** (object→cost ≪1) | ✅ **견고** | 전 시나리오 gain 0.007~0.065 |
| **cost-map = 최고 안전임계** | ✅ **견고** | 전 시나리오 재현 |
| planner 증폭 | ⚠️ 부분 | car-following만 재현(2.68), 밀집 장면 약함 |
| planner-switch 게이트웨이 | ❌ 미재현 | argmin-flip 0.89→**0.02~0.23** (밀집장면 특이) |
| directed > random | ❌ 미재현 | (CARLA와 동일: 견고한 건 일관성) |

→ **핵심 특성화(래스터화=감쇠기, cost-map=최고 임계 인터페이스)는 실 nuPlan에서 일반화된다.** 나머지는 희소·합성 장면 현상.

### 5.6 실물리·실모델 보조 검증
- **CARLA 450 에피소드:** directed 결함이 실제 폐루프 안전 저하(leading 2.0m에서 100% 충돌 threshold).
  견고한 명제 = **directed는 일관·저분산(min-TTC std 0~0.32), random은 산발·고분산(0.74~0.81)**.
  ("directed가 충돌을 더 낸다"는 **거짓** — 정직하게 보고. 견고한 건 일관성/해석가능성)
- **실제 InterFuser BEV feature:** directed displacement > geometric twist, 세 표현 전부(해상도↑ 1.41→1.91배).

---

## 6. 기여 (Contributions)

1. 표현 간 **semantic fault propagation을 정식화**한다.
2. **semantic fault를 시험 신호(probe)로 삼아 자율주행 파이프라인을 특성화**한다 (empirical system identification).
3. **임계 전파 경계를 규명하고, 왜 증폭·감쇠하는지 구조적으로 설명**한다 (M1 투영 Jacobian / M2 argmin 결정경계 / M3 적분).
4. **planner 선택이 안전 증폭 관문**임을 보인다 — *단, argmin 계열 + 희소 장면 조건에 한정*(보편 아님, 명시).

---

## 7. 정직한 한계 (발표에서 먼저 말할 것)

- **"planner-switch 게이트웨이"는 삼중 제약**: argmin 계열 특이적 + 희소 장면 특이적 → **"법칙"이 아니라 "특정 조건에서 관찰된 메커니즘"**으로 서술.
- **방어 가능한 일반화 핵심** = 래스터화=감쇠기 + cost-map=최고 임계 인터페이스 (합성+실데이터 모두 성립).
- 전파 지표(Gain/FAR/CIS)는 **우리가 정의한 분석 도구** → "정의+관찰"까지만 주장.
- 합성 planner/시나리오는 단순화; nuPlan은 open-loop(안전지표 = plan 편차 + 실 agent와 min-distance·TTC), 실데이터 효과 크기는 작음.
- gaussian baseline은 계약 한계 초과로 실패 → random 방향 baseline으로 대체(값을 지어내지 않고 실패를 기록).
- interface-gain 절대값은 정규화 가정에 의존 → **부호·순서만 주장**.

> **핵심 태도:** 순진한 가설("directed가 더 위험")이 틀렸음을 인정하고, 일관성·메커니즘·일반화로 재정립하는 **정직한 연구 아크**.

---

## 8. 결론 & 향후

**결론:** semantic fault를 excitation으로 쓰면 자율주행 파이프라인의 **오류 전파 특성을 식별**할 수 있고,
**래스터화 경계는 오류를 감쇠, planner(argmin) 경계는 오류를 증폭**하며, **cost-map이 가장 안전 임계적인 표현 경계**임을
합성·실데이터에서 일관되게 보였다. 각 거동의 **구조적 이유(M1/M2/M3)**까지 설명했다.

**향후:**
- 학습 기반 planner로 특성 일반성 추가 검증
- nuPlan closed-loop 시뮬 / 멀티-location(Boston·Pittsburgh·Singapore)
- 논문화: excitation/SI 서사, Figure 1 = propagation graph, Figure = failure taxonomy tree

**Target venue:** IEEE T-ITS / T-IV (제어·시스템 성향) 1순위, DSN/ISSRE/RESS 2순위.

---

## 부록 A. 재현성
- 코드: `torsion/analysis/{propagation,mechanism,transfer_function,failure_taxonomy}.py`,
  `torsion/scenarios/{unified_pipeline,costmap_runner,predict}.py`,
  `torsion/data/{nuplan_adapter,nuplan_map,_geometry}.py`, `scripts/run_*.py`
- 데이터: `results/metrics/*.csv` / 상세 수치: `TORSION_results_summary.md` §4.1~4.14
- **전체 테스트 108 passed / 1 skipped**, 하위호환 유지(합성 flag off / argmin → 기존 결과 100% 재현)
- nuPlan 처리 = 경량 스택(shapely + pyproj, devkit 불필요)

## 부록 B. 발표용 한 장 그림 아이디어 (Figure 1)
```
                     [gain]        [gain]        [gain]        [gain]
  Semantic Fault → object ──2.09──▶ prediction ──0.004──▶ cost-map ──10~40──▶ plan ──2.3──▶ control → Safety
                           (증폭)         (감쇠: 래스터화)      (증폭: argmin)       (선형)
   ▲ excitation                                    ▲ 감쇠기            ▲ 증폭기 = 위험 관문
```
- 각 화살표에 gain 수치 → 이게 "전파 그래프 = 식별된 시스템 특성"
- 감쇠(래스터화) / 증폭(argmin) 대비가 한눈에
