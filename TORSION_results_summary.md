# TORSION — 실험 정리 & 결과 요약 (리뷰용)

> 작성일 2026-07-06. 본 문서는 TORSION 연구의 **핵심 아이디어 → 실험을 어떻게 교정했는지 → 그래서 어떤
> 결과가 나왔는지**를 정직하게 한 곳에 정리한 리뷰/컨펌용 문서입니다. 상세 설계는 `TORSION_experiment_design.md`,
> 최종 프레이밍은 `TORSION_framing.md`, 그림은 `results/figures/`, 수치는 `results/metrics/` 참고.

---

## 1. 한 문단 요약

TORSION은 **"Semantic Fault"라는 새로운 fault model**을 정립하는 연구다. 자율주행 파이프라인의 **중간 표현
(object-set / cost-map / BEV feature)** 에, **구조적 계약(count·class·ID, cost range, drivable topology,
tensor shape)은 보존하면서 의미(semantic)만 국소적으로 비트는** fault를 주입하고, 그 fault가 **안전 실패로
어떻게 전파되는지**를 체계적으로 분석하는 프레임워크다. 저수준 fault(bit-flip)와 최악입력 공격(adversarial)
사이의 빈 층을 채우며, 기여는 "강력한 새 연산자"가 아니라 **방법론(3 pillars)** 과 **fault model 자체**에 있다.
합성(3표현) + 실측(CARLA)에서 정직하게 검증했다.

---

## 2. 핵심 정의 (교정 후 최종)

> **TORSION은 기하학을 비트는 것이 아니라, 구조적 유효성을 보존하며 의미 해석을 비트는 것이다.**
> Rotation·Displacement·Cost-inflation은 모두 그 아래의 **연산자**일 뿐이고, TORSION은 프레임워크 전체를 가리킨다.

**Semantic Contract를 깨는 것(❌ Gaussian noise, bit-flip)** vs **계약 안에서 의미만 비트는 것(✅ TORSION)** —
이 구분이 프레임워크의 정체성이다.

---

## 3. 실험을 어떻게 교정했는가 (정직한 과정)

이 연구의 핵심 가치는 **순진한 가설을 공정한 검증으로 반복 교정**해 방어 가능한 결론에 도달했다는 데 있다.
각 단계는 `문제 발견 → 교정 → 결과` 형태로 정리한다.

### 교정 1 — 불공정한 baseline을 공정하게
- **문제:** 초기 비교에서 TORSION이 Gaussian noise보다 강해 보였으나, 알고 보니 (a) 섭동 예산이 4배 크고,
  (b) yaw·velocity 채널을 gaussian은 건드리지도 않았으며, (c) TORSION만 지향적·타겟팅이었다. → 이기는 이유가
  "semantic 보존"이 아니라 교란 요인 때문일 수 있었다.
- **교정:** 동일 필드·동일 예산으로 맞춘 `gaussian_matched`, 그리고 **방향성만 제거한** `random_warp`
  (계약 보존, 무작위 방향) baseline을 도입해 **구조(structure)만 차이나도록** 통제.
- **결과:** 순진한 H1("TORSION이 더 강함")은 **거짓**으로 판명. 충돌률에서는 오히려 gaussian이 산발적으로 더 냈다.
  그러나 **동일 예산에서 TORSION은 더 심각하고(min-TTC), 시나리오 내에서 거의 결정론적인(std≈0)** 저하를 유도.
  → 명제를 **"강함"이 아니라 "일관성·해석가능성"** 으로 재정립.

### 교정 2 — 통계적 엄밀성 (결정론적 CI 붕괴 해결)
- **문제:** 지향적 연산자가 결정론적이라 시나리오별 95% CI가 `[1.101, 1.101]`처럼 0으로 붕괴 → 통계적 신뢰 없음.
- **교정:** 시나리오 **인스턴스를 확률화**(seed마다 actor 초기위치·속도·타이밍 랜덤 + 관측 노이즈),
  **paired 비교**(같은 인스턴스에 clean/각 방법 적용), **bootstrap 95% CI + tail risk + seed간 분산** 도입.
- **결과:** 진짜 분포가 생겼고, 핵심 결론이 **CI가 0을 배제하며 생존**:
  `directed vs random`(paired ΔTTC 2.416 [2.092, 2.700]), `displacement vs twist`(2.347 [2.060, 2.630]).

### 교정 3 — "이름값(진짜 twist)" 가설의 정직한 검증
- **문제:** "TORSION"이라는 이름에 맞게 signature 메커니즘을 **진짜 비틀림장(twist)** 으로 만들고자 함
  (기존 연산자는 사실 평행이동+균일회전이라 엄밀히 torsion이 아님).
- **교정:** 거리 의존적으로 각이 변하는 **swirl(공간 비틀림)** + **temporal curl(궤적 비틀림)** 을 구현하고,
  **displacement(변위)** 와 동일 예산에서 공정 비교.
- **결과 (희소 객체):** 순수 twist는 **displacement보다 약함** — 회전은 예산을 접선 방향으로 분산시켜 덜 치명적.
- **결과 (밀집 cost-map):** 처음엔 swirl이 clean→100% 충돌로 극적이었으나, **알고 보니 도로 경계를 왜곡한
  계약 위반(B.4)** 이 원인. **경계를 보존하도록 교정**하자 swirl은 오히려 무해(planner가 더 보수적) → twist 다시 약화.
- **결론:** 기하학적 twist는 이기지 못한다. **displacement가 실효 연산자.**

### 교정 4 — 실제 학습된 표현(BEV feature)에서 재검증
- **문제:** 위 결과는 raw 표현에서였다. 학습된 BEV feature에서는 "translation 등가물이 없어" twist가 이길 수도 있다는
  가설(CNN이 translation-equivariant하면 translation은 양성 변환이라는 논리).
- **교정:** **실제 InterFuser 모델**(GitHub 코드 + 공개 체크포인트)을 GPU로 로드해 shared BEV feature에
  forward-hook 주입. twist vs translate를 **두 해상도(7×7, 28×28)** 에서 예산 맞춰 비교(20개 입력).
- **결과:** 두 해상도 모두 **translate > twist**(7×7에서 1.35–1.41×, 28×28에서 1.91–1.97× — 고해상도에서 격차 확대).
  equivariance 오차도 커서 translation은 양성 변환이 아니라 실제로 효과적. → **세 표현 모두에서
  "directed displacement > geometric twist" 3회 확인.** twist는 은유로 남고, 메커니즘은 directed semantic displacement.

### 교정 5 — 표현 간 비교 가능성 (unified pipeline)
- **문제:** object-set과 cost-map이 서로 다른 planner를 써서, 차이가 "표현" 때문인지 "planner" 때문인지 뒤섞임.
- **교정:** **하나의 파이프라인**(scenario → object-set → cost-map으로 rasterize → 동일 planner → control)을 만들어,
  **동일 downstream에서 주입 단계(object stage vs cost-map stage)만 바꿔** 비교.
- **결과 (flagship):** 동일 예산에서 **cost-map(늦은 단계) 주입이 object-set(이른 단계)보다 훨씬 치명적**
  (paired ΔTTC high `0.802 [0.585, 1.021]`, 평균 열화 object 0.003 vs cost-map 0.441). 이른 fault는 rasterization+
  planner에 흡수(감쇠), 늦은 fault는 제어에 직접 전파. → **오류 전파의 비대칭**을 정량화.

### 교정 6 — 실측 (CARLA closed-loop)
- **문제:** 합성 결과가 실제 물리·제어에서도 성립하는가.
- **교정:** 실제 CARLA 0.9.16 closed-loop에서 leading-vehicle / cut-in 시나리오, ego가 쓰는 object 인지에 semantic
  displacement 주입. **450 에피소드(15 seed/cell)** 로 재실측. (초기 n=5 예비 결과는 과대평가였음 → seed 확대로 교정.)
- **결과:** 아래 §4.5.

---

## 4. 검증된 결과 (수치)

모든 값은 `results/metrics/`의 CSV에서 추출. min-TTC 단위는 초(s), 낮을수록 위험. 예산은 realized budget(동일 열).

### 4.1 합성 — 동일 예산·계약보존, high magnitude, 표현별 (30 seeds)

**Object-set** (clean 기준 min-TTC = 3.12):

| method | collision | mean min-TTC | std min-TTC |
|---|---:|---:|---:|
| torsion_displace (directed) | 0.333 | **1.470** | **≈0** |
| random_warp (무작위 방향) | 0.100 | 2.063 | 0.613 |
| gaussian (계약 위반) | 0.000 | 2.378 | 0.468 |

**Cost-map** (clean 기준 min-TTC = 1.67):

| method | collision | mean min-TTC | std min-TTC |
|---|---:|---:|---:|
| torsion_displace (directed) | 0.000 | **1.091** | **≈0** |
| random_warp | 0.056 | 2.292 | 0.331 |
| gaussian (계약 위반) | 0.000 | 2.751 | 0.305 |
| swirl (legal, 경계보존) | 0.000 | 2.306 | ≈0 |
| swirl_illegal (경계왜곡=계약위반) | 0.000 | 3.135 | ≈0 |

핵심: **directed displacement가 가장 낮은 min-TTC(가장 심각) + std≈0(결정론적)**. random/gaussian은 std가 큼(산발).
계약을 위반한 illegal swirl은 오히려 legal보다 **덜 위험**(3.135 > 2.306) → "위반=파국"은 예산 아티팩트였음(정정).

### 4.2 Directedness gap — directed vs random (high, 동일 예산)

| 표현 | directed min-TTC | random min-TTC | Δ(random−directed) | std(directed) | std(random) |
|---|---:|---:|---:|---:|---:|
| object_set | 1.470 | 2.063 | **+0.593** | ≈0 | 0.613 |
| cost_map | 1.091 | 2.292 | **+1.200** | ≈0 | 0.331 |

directed가 더 심각(Δ>0)하고 동시에 더 일관(std 작음).

### 4.3 연산자 leaderboard (composite, 상위)

| rank | 표현 | method | mag | collision | min-TTC | composite |
|---:|---|---|---|---:|---:|---:|
| 1 | object_set | **torsion_displace** | high | 0.333 | 1.470 | 0.909 |
| 2 | object_set | torsion_displace | medium | 0.333 | 2.153 | 0.797 |
| 4 | object_set | random_warp | high | 0.100 | 2.063 | 0.424 |
| 5 | cost_map | torsion_displace | high | 0.000 | 1.091 | 0.394 |

→ **directed displacement가 두 표현 모두 최상위.** (twist/curl/swirl은 하위.)

### 4.4 BEV feature (실제 InterFuser) — waypoint 변화 / 단위 feature 예산 (high, 20 inputs)

| hook (해상도) | bev_twist | bev_translate | **translate / twist** | gaussian |
|---|---:|---:|---:|---:|
| patch7 (7×7) | 0.01292 | 0.01824 | **1.41×** | 0.00958 |
| layer2_28 (28×28) | 0.00630 | 0.01206 | **1.91×** | 0.00355 |

두 해상도 모두 **translate > twist**, 고해상도에서 격차 확대. equivariance 상대오차(waypoint):
patch7 **0.79**, layer2_28 **1.97** → translation은 "양성 변환"이 아니라 실제로 효과적. → **displacement > twist 3회째 확인.**

### 4.5 Cross-representation 주입지점 민감도 (unified pipeline, 30 seeds)

전체(ALL, high) 집계:

| 주입 단계 | collision | mean min-TTC | std | realized budget |
|---|---:|---:|---:|---:|
| inject@object (이른 단계) | 0.033 | 1.678 | 0.830 | 0.844 |
| inject@costmap (늦은 단계) | **0.111** | **0.876** | 0.535 | 0.667 |

paired 차이(object−costmap min-TTC, 양수 = costmap이 더 위험), 95% CI:
low **0.053 [0.026, 0.086]**, medium **0.185 [0.073, 0.305]**, high **0.802 [0.585, 1.021]** — 전 구간 0 배제.
→ **늦은 단계(cost-map) 주입이 이른 단계(object)보다 치명적** (전파 감쇠의 비대칭). leading에서 object 주입은
오히려 안전해지는 감쇠·역전도 관측.

### 4.6 CARLA 실측 (450 에피소드, 15 seeds/cell, Town10HD)

**leading_vehicle — 충돌률 (magnitude, m):**

| method | 0.5 | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| directed displace | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** |
| random direction | 0.00 | 0.00 | 0.40 | 0.47 | 0.60 | 0.53 |

min-TTC (leading): clean 1.40 (std 0.09); directed std **0–0.32**; random std **0.74–0.81**.

**cut_in — 충돌률:**

| method | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|
| clean | 0.13 | 0.13 | 0.13 | 0.13 |
| directed displace | 0.13 | 0.20 | 0.33 | **0.47** |
| random direction | 0.07 | 0.13 | 0.07 | 0.33 |

- ✅ **directed 의미 fault가 실제 CARLA 안전을 저하** (leading: 2.0m에서 100% 충돌의 급격한 threshold; cut_in: 단조 0.13→0.47).
- ⚠️ **정직한 nuance:** "directed가 random보다 충돌을 더 낸다"는 **거짓** — 중간 구간(1.25~1.75m)은 random이 더 충돌.
  견고한 구분은 **directed=일관·저분산·threshold형, random=산발·고분산**(합성 consistency 명제의 실물리 재현).
- 단서: cut_in은 clean이 이미 13% 충돌(병합 공격적)이라 noisy → leading이 깨끗한 판별 시나리오.

### 4.7 Fault-propagation 지표 (Phase A — 기존 unified trace에서 산출, 신규 실험 0)

동일 unified 파이프라인(object→cost-map→plan→control→safety)의 프레임별 단계 오차를 재분석해
5개 전파 지표를 정의·측정. 모두 30 seeds, high magnitude, directed displacement.

**(a) Critical Interface Score (CIS = 안전저하 / matched plan budget) — 경계별 안전 임계도:**

| 시나리오 | inject@object | inject@cost-map | 비대칭 |
|---|---:|---:|---:|
| leading_vehicle | 0.066 [0.025, 0.118] | **1.739 [1.518, 1.963]** | **≈26×** |
| cut_in | 0.275 [0.169, 0.406] | 0.711 [0.611, 0.830] | 2.6× |
| pedestrian_crossing | 0.006 | 0.010 | ~0 (둘 다 미미) |
| **ALL** | **0.125 [0.080, 0.177]** | **1.152 [0.955, 1.335]** | **≈9×** |

→ **늦은 단계(cost-map) 주입이 안전 임계도에서 object 대비 9배(leading 26배).** contribution #5(cross-stage asymmetry) 정량 확정.

**(b) Interface-gain propagation map (정규화 오차의 경계별 비 ratio; >1 증폭, <1 감쇠) — ALL:**

| 주입 | object→cost | cost→plan | plan→control | control→safety | FAR |
|---|---:|---:|---:|---:|---:|
| object | **0.004** (감쇠기) | 17.6 | 2.27 | 0.004 (소멸) | 0.009 |
| cost-map | — (N/A) | **40.3** (증폭기) | 2.36 | 0.068 (생존) | **0.167** |

→ **object→cost-map 래스터화 경계 = 천연 감쇠기**(gain 0.004, object 결함 대부분 흡수),
**cost-map→plan 경계 = 결정적 증폭기**(17~40배). cost-map 주입은 감쇠기 하류·증폭기 상류에 진입해 가장 위험.
leading_vehicle object의 FAR = **−0.040 (음수 = 안전 개선, inversion)** → "attenuated **or inverted**"의 직접 증거.

**(c) Reach-safety rate (결함이 safety 단계까지 전파된 run 비율) — depth의 공정 비교형:**

| 시나리오 | object | cost-map |
|---|---:|---:|
| leading_vehicle | 0.30 | **1.00** |
| cut_in | 0.70 | 1.00 |
| pedestrian_crossing | 0.00 | 0.00 |
| **ALL** | **0.33** | **0.67** |

(raw propagation-depth count는 object가 자기 injection 단계에서 +1 먼저 시작하는 위치 편향이 있어 count 직접 비교는
부적절 → **safety 도달률**로 공정 비교. cost-map 결함이 leading에서 100% 안전까지 전파, object는 30%.)
**Recovery time**: object 3.03 s, cost-map 3.38 s.

**(d) Directedness × injection 상호작용 (신규 발견):** random-direction 결함은 이 비대칭이 무너짐 —
cost-map CIS가 **1.152 → 0.048**로 폭락(object는 0.125→0.161로 오히려 상승). 즉 **방향성(directed) 결함이
특히 cost-map 경계에서 위험**하다는 표현-의존적 상호작용.

> **한계/주의:** interface-gain의 절대값은 단계별 정규화 기준(고정 reference) 선택에 민감한 **모델링 가정** —
> 논문에선 상대 순서·부호만 주장하고 정규화 민감도를 함께 보고. CIS·reach-safety·recovery는 물리 단위라 견고.
> gaussian baseline은 unified 파이프라인 calibration에서 contract 한계 초과 예외로 전량 실패(540 run) →
> 공정 baseline은 random_warp 사용(값 지어내지 않고 failed 기록).

### 4.8 Prediction 단계 추가 (Phase B — 파이프라인을 현실화: object→**prediction**→cost→plan→control→safety)

기존 파이프라인은 Prediction을 건너뛰어 velocity/yaw 결함이 저평가됨. 명시적 constant-velocity 예측 단계를
추가(`use_prediction`, default False로 기존 결과 100% 재현; 새 결과는 `propagation_pred_*.csv` 별도 저장,
74 tests pass). 30 seeds, high, directed displacement.

**CIS — 시나리오별 3주입점 (object / prediction / cost-map):**

| 시나리오 | object | prediction | cost-map |
|---|---:|---:|---:|
| leading_vehicle | 0.510 | **0.032** (무의미) | **1.377** |
| cut_in | 0.644 | 0.117 | 0.677 |
| pedestrian_crossing | **1.226** | **1.025** | 0.950 |
| **ALL** | 0.684 | 0.340 | **1.021** |

**Interface-gain map (ALL, object 주입):** object→prediction **2.09 (증폭기)** → prediction→cost **0.004 (감쇠기)**
→ cost→plan **10.5 (증폭기)** → plan→control 2.30 → control→safety 0.068. FAR 0.157, reach-safety 0.91.

**Horizon 스케일링 (leading, inject@prediction):** raw 예측오차 = 2.26 / 4.67 / 7.55 (H=1/2/3 s) →
**예측오차가 horizon에 선형 비례 = CV Jacobian ≈ H 실증** (Phase A mechanism 연결). 단, 이 증폭은
prediction→cost 감쇠(0.004)에서 흡수돼 leading의 안전영향(CIS)으로는 이어지지 않음.

**확정 결론 (Phase B):**
1. **Prediction = 오차 크기 증폭기** (object→pred gain 2.09, leading 3.40; horizon 선형). advisor 가설 실증.
2. **Prediction→cost 래스터화 = 감쇠기** (0.004) — object→cost와 동일한 구조적 병목.
3. **Prediction 경계 임계도는 시나리오 의존:** 미래 궤적이 안전을 결정하는 **pedestrian_crossing에서 지배적(1.02)**,
   현재 간격이 지배하는 leading에서 무의미(0.03). → "critical propagation interface"가 상황 특이적임을 정량화.
4. **두 Phase A 아티팩트 교정:** (a) "object 결함 감쇠"는 부분적으로 prediction 누락 탓 — 넣으면 object CIS
   0.125→0.684, reach-safety 0.33→0.91. (b) "pedestrian 결함 둔감(CIS~0.01)"은 **완전한 prediction-누락 아티팩트** —
   넣으면 세 경계 전부 고임계(1.23/1.02/0.95). 현실적 파이프라인이 안전 임계도를 크게 바꿈.
5. **구조 병목 서사 확정:** 래스터화(object→cost, prediction→cost) = 일관된 감쇠기; cost→plan(argmin) = 일관된 증폭기.
   cost-map은 여전히 단일 최고 임계 경계(ALL 1.02).

### 4.9 메커니즘 — "왜 감쇠/증폭하는가" (Phase A+ — 측정→설명, 순수 분석)

각 경계의 감쇠/증폭을 **구조적 원인**으로 정량 규명(연산자·파이프라인·planner 무수정). 30 seeds. 5 tests pass.

**M1 — 래스터화 경계 = 감쇠기 (커널·격자 한계 투영 Jacobian):**
ε-섭동 유한차분으로 국소 Jacobian J = Δcost_L2/ε 측정:

| 시나리오 | object→cost J | prediction→cost J |
|---|---:|---:|
| cut_in | 0.0254 | 0.0378 |
| leading_vehicle | 0.0200 | 0.0354 |
| pedestrian_crossing | 0.0360 | 0.0401 |

→ **전부 J ≪ 1, ε-선형** = cost-map이 object 위치의 커널-평활·격자-양자화된 many-to-one 투영이라 국소 민감도가 작음 → 감쇠(측정 gain ~0.004)를 설명.

**M2 — cost→plan 경계 = 증폭기 (sampling-argmin 결정경계 스위칭):** ⭐ 핵심 인과
프레임별 clean decision margin(= 최적−차선 후보 score 갭, feasible 후보 중) 사분위별:

| margin 사분위 | mean realized plan deviation | argmin flip rate |
|---|---:|---:|
| Q1 (최소 margin) | **0.151 m** | **0.68** |
| Q2 | 0.039 m | 0.196 |
| Q3 | 0.023 m | 0.114 |
| Q4 (최대 margin) | **0.0056 m** | **0.028** |

→ 최소-margin 사분위에서 plan deviation **≈27×**, argmin flip **≈24×** (완벽 단조). **Spearman ρ=0.52**(강한 단조) vs Pearson 0.09 — 이 격차가 "결정경계 근처 1/margin 쌍곡 blow-up"(비선형)의 증거.
**증폭 메커니즘 = gradient가 아니라 argmin이 결정경계를 넘으며 승자 후보가 이산 점프**(advisor의 gradient 표현을 우리 sampling planner에 맞게 정정·실증).

**M3 — object→prediction 경계 = 증폭기 (CV 적분 Jacobian):**

| horizon (s) | analytic J | empirical J | raw pred L2 |
|---:|---:|---:|---:|
| 1 | 0.5 | 0.5 | 2.26 |
| 2 | 1.0 | 1.0 | 4.66 |
| 3 | 1.5 | 1.5 | 7.55 |

→ CV 예측이 δv를 δv·t로 사상 → ∂pred/∂v = t, horizon 평균 = H/2. **analytic = empirical 정확 일치**, raw L2 ∝ H → prediction 증폭 = "속도오차의 horizon 적분" 확정.

> **종합:** 세 경계의 거동이 각각 (투영 Jacobian ≪1) / (argmin 결정경계 스위칭) / (CV 적분 Jacobian = H/2)로 설명됨.
> 논문 핵심 전환("측정 → 메커니즘")을 이 세 결과가 담당. M2는 부분적 한계도 정직히 표기(Q4 flip 0.028 ≠ 0, Pearson 약함=비선형).

### 4.10 Propagation Response 특성화 (경계를 선형/비선형 응답으로 분류)

> **용어 주의(advisor 확정):** 우리가 재는 것은 **ε-유한차분 경험적 gain**(dynamics·주파수/Laplace 영역 없음)이므로
> 논문에서는 강한 control-theory 용어 **"transfer function"을 피하고** "propagation response / empirical transfer
> behavior"로 표기. (코드 모듈 `transfer_function.py`는 내부 명칭.)

각 경계 gain을 **주입 크기(plan budget 0.05→2.0 m, 9단계)의 함수로** 측정 → 선형 응답(gain 크기-무관)과
비선형 응답(gain 크기-의존) 구분. **raw 물리 비율**로 판정(정규화 gain 곱은 telescoping이라 자명 → 회피). 5 tests pass.

**선형성 verdict (raw gain CV = std/mean across budget):**

| 경계 | object 주입 | costmap 주입 | 성격 |
|---|---|---|---|
| object→prediction | 선형 (CV 0.054, gain 1.30→1.51) | — | **선형** (CV 적분) |
| prediction→cost | 선형 (CV 0.148) | — | ~선형 (경미한 saturation) |
| **cost→plan** | **비선형** (CV 0.172, 9.5→12.7↑) | **비선형** (CV 0.171) | **비선형 스위칭** (argmin) |
| plan→control | 선형 (CV 0.081, ~2.3) | 선형 (CV 0.056) | **선형** (추종) |
| control→safety | 비선형 (CV 0.455) | 비선형 (CV 0.538) | 비선형 saturation (충돌경계) |

**확정 결론 (control-theoretic):**
- **cost→plan = 지배적 비선형 스위칭 소자** — 3개 주입점 전부 비선형(CV 0.17~0.37). M2(argmin) 재확인. gain이 크기 따라
  9.5→12.7 상승.
- **plan→control = 선형 추종 소자** (3곳 전부 CV 0.03~0.08), object→prediction도 선형 → **파이프라인은 선형 전달소자들 사이에
  planner(argmin)라는 비선형 스위칭 소자가 낀 구조.**
- **정직한 nuance:** "planner만 비선형"은 단순화. **control→safety도 비선형(충돌경계 saturation)** — 단 safety_drop이
  censored/sparse라 추정 noisy(prediction 주입 시 mean gain≈0으로 CV 발산). 출력단 saturation으로 해석.
- prediction→cost 래스터화는 ~선형이나 크기 커지면 완만한 saturation(gain 0.048→0.035).

### 4.11 Failure Taxonomy (데이터 기반 대표 전파경로 — fault origin → signature → failure mode)

810 run(3 시나리오 × 3 주입점 × 3 magnitude × 30 seed, 6-stage prediction 체인)을 자동 분류. 4 tests pass.
경로 손으로 안 그리고 **데이터에서 빈도와 함께 추출**. failure mode 우선순위: collision > off_road > lane_departure >
hard_brake > near_miss(TTC<1.5) > safe. signature = argmin flip 시 `planner_switch`.

**대표 경로 (상위):**
| fault origin | signature | failure mode | runs | 비율 |
|---|---|---|---:|---|
| prediction | planner_switch | hard_brake | 230 | 28.4% (85% of pred) |
| object | planner_switch | hard_brake | 206 | 25.4% (76% of obj) |
| costmap | planner_switch | hard_brake | 188 | 23.2% (70% of cost) |
| **costmap** | **planner_switch** | **collision** | 32 | 4.0% (12% of cost) |
| object | attenuated | hard_brake | 30 | 3.7% |

**origin별 충돌률:** cost-map **11.9%** > object 5.2% > prediction 0.4%.

**확정 결론:**
1. **planner argmin switch = 보편적 전파 관문** — origin 무관 거의 모든 결함(76~85%)이 planner 스위칭 경유. M2·§4.10
   transfer-function을 taxonomy가 독립 재확증(planner가 유일 보편 증폭 게이트).
2. **지배적 유발 실패 = phantom/hard brake**(충돌 아님) — 이 planner는 결함에 급제동으로 방어(swerve 아님).
   contract가 도로경계 보존이라 차선이탈 대신 감속. (phantom braking = 실제 AV 심각 실패모드.)
3. **origin→심각도 gradient**: cost-map만 충돌로 escalate(12%), prediction은 방어제동에 그침(0.4%).
4. **정직한 확인:** advisor 예시 3경로(object→prediction drift→wrong yield / cost-map→planner switch→lane departure /
   prediction→late brake→collision)는 **데이터 미지지**(lane_departure/near_miss 거의 없음=급제동 정책, pred→collision 1건).
   데이터는 더 통일된 서사("모든 결함→planner switch→대부분 급제동/cost-map만 충돌")를 줌.
5. **→ Task 3 필연성:** 차선이탈/swerve 부재는 **이 planner의 급제동 정책 특이성** → 2번째 planner가 다른 실패모드를
   보이는지 확인해야 taxonomy·transfer-function의 planner-불변성 방어 가능.

### 4.12 Planner-invariance & 인과 격리 (Task 3 — hard argmin ↔ soft softmax 선택)

planner 선택 방식을 pluggable하게(argmin ↔ softmax τ) 만들어(하위호환 default argmin; 전체 스위트 93 passed, 1 skipped)
비선형성이 argmin 때문인지(H1)·전파 구조가 planner-불변인지(H2) 통제 실험. seeds 20, 6-stage 체인.

| 지표 | argmin | softmax τ=0.02 | softmax τ=0.1 |
|---|---:|---:|---:|
| cost→plan gain CV | 0.187 | 1.30 | 1.36 |
| cost→plan norm slope | 0.003 | 1.31 | 1.34 |
| cost→plan verdict | nonlinear | nonlinear | nonlinear |
| planner_switch rate | 0.889 | 0.567 | 0.578 |
| 충돌률 object | 0.050 | 0.000 | 0.000 |
| 충돌률 prediction | 0.006 | 0.028 | 0.144 |
| 충돌률 cost-map | 0.106 | 0.000 | 0.000 |
| cost-map CIS (high) | 0.957 | 0.007 | 0.258 |
| object CIS (high) | 0.631 | 0.015 | 0.461 |

**확정 결론 (정직·미묘):**
- **H1 부분 확증 + 반전:** hard argmin은 **이산 planner-switch 게이트와 충돌 escalation의 원인** — 부드럽게(softmax) 하면
  switch **−36%**(0.889→0.57), **object/cost-map 충돌이 0으로** 소거. 즉 **soft 선택은 fault→충돌의 실질적 완화책**(planner
  개발자 actionable). **하지만 cost→plan은 여전히 비선형**(3 모드 전부) — softmax는 비선형을 없애지 못하고 **이산 스위칭 →
  연속 크기-의존**으로 성격만 바꿈. (CV 절대값 1.3은 soft planner가 거의 반응 안 해 e_plan≈0 → 비율 불안정한 부분 있음;
  robust한 건 "switch↓·충돌↓, 비선형 잔존".)
- **H2 부분 불변:** 증폭/감쇠 **위상(cost→plan 증폭, 래스터화 감쇠, prediction 증폭)은 유지** = 구조는 planner-일반적. 그러나
  **임계도 절대값·순위는 planner-의존**(softmax가 전역적으로 훨씬 안전, costmap CIS 0.957→0.007; τ0.1선 object>costmap로 역전).
- **이상치(정직 표기):** softmax 고온(τ0.1)에서 **prediction 충돌만 상승**(0.006→0.144) — soft-blend된 경로가 오염된 예측을
  덜 결정적으로 회피하는 소수 pedestrian 케이스(n=20, sparse) 추정. 과대해석 금지.

### 4.13 Planner-independence (구조가 다른 planner — sampling vs potential-field/gradient)

argmin 계열 밖의 **구조적으로 다른 planner**(PotentialFieldPlanner: cost-field 기울기+차선인력으로 연속 조향,
후보열거·argmin 없음)를 추가(하위호환 factory, 전체 97 passed/1 skipped)해 핵심 발견의 planner-불변성 검증. seeds 20.

| planner | costmap CIS | object CIS | pred CIS | cost→plan gain | CV | verdict | gateway rate | argmin flip | 충돌 obj/cost |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| sampling(argmin) | 0.957 | 0.631 | 0.371 | 15.41 | 0.187 | nonlinear | **0.852** | 0.889 | 0.05 / 0.106 |
| potential-field | 15.73* | 10.37* | 0.72* | 0.816 | 0.346 | nonlinear | **0.139** | 0.000 | 0.00 / 0.033 |

**Planner-독립 (구조 유지 — 두 아키텍처 공통):**
- **cost-map = 최고 CIS 랭크** (costmap > object > prediction), 두 planner 모두.
- **cost→plan = 비선형 응답** (선형 아님), 두 planner 모두.
- argmin flip은 potential-field에서 0 (예상대로).

**Architecture-특이 (메커니즘 다름) — C4에 직접 영향:**
- **"planner-switch 게이트웨이"는 sampling-argmin 특이적** — gateway rate 0.852→**0.139** 붕괴, cost→plan 평균 gain
  15.4→**0.816(<1, 평균적으로 증폭 안 함)**. 즉 **"planner switching이 보편 게이트웨이"는 planner 아키텍처를 넘는
  보편이 아님** — argmin 계열에 국한.
- gradient planner가 훨씬 안전(충돌 →~0). **Task 3 softmax와 동일 방향: 이산(argmin)=위험 증폭기 / 연속(softmax·gradient)=안전·감쇠.**

**정직한 caveat:** ***potential-field CIS 절대값(15.73)은 sampling(0.957)과 직접 비교 불가** — plan budget(분모)이 작아
CIS가 부풀려진 아티팩트(softmax 때와 동일 원인). **랭크·verdict만 교차 비교, 절대값 금지.** gradient planner 게인/임계는
튜닝 여지가 있어 "덜 증폭"이 일부 과감쇠 설정 탓일 수 있음.

**→ C4 정직한 재정의:** "**fault origin을 가로질러**(Task 2, 76~85%) planner 인터페이스가 보편 게이트웨이"는 **argmin
계열(AD 주류) 안에서** 성립; **연속-선택 planner는 이 게이트웨이를 완화**. 즉 보편성은 "fault origin에 대해"이지
"planner 아키텍처에 대해"가 아님. 대신 **cost-map 최고 임계 + 이산선택=위험/연속선택=안전**이 planner-불변 결론.

### 4.14 실데이터 일반화 (nuPlan open-loop, 실 지도 + 실 agent) ⭐ item ②

**실 nuPlan** 로그(12 logs, 290 frames, **10,440 runs**)에서, **실 지도 도로 prior(gpkg lane→UTM 재투영)** + **실 agent
obstacle**로 cost-map을 만들어 동일 semantic fault를 주입, 합성 결과의 재현 여부 검증(open-loop, 두 안전지표). 108 passed/1 skipped.
데이터: shapely+pyproj로 gpkg 파싱(devkit 불필요), `torsion/data/{nuplan_adapter,nuplan_map,_geometry}.py`.

**per-category (torsion_displace, sampling planner):**
| 시나리오 | object→cost gain | cost→plan gain | argmin flip | cost-map>object (plan dev) |
|---|---:|---:|---:|---|
| FOLLOWING | 0.018 | **2.68** (증폭 ✓) | 0.11 | ✓ 0.159 > 0.070 |
| INTERSECTION | 0.007 | 0.84 (증폭 안 함) | 0.02 | ✓ 0.035 > 0.017 |
| LANE_CHANGE | 0.065 | 0.63 | 0.0~0.33 | ✓ 0.133 > 0.0 |
| **전체 평균** | **0.014** [0.013,0.015] | **1.83** [1.57,2.11] | 0.085 | ✓ |

**재현 판정 (정직 — 긍정·부정 모두):**
| 주장 | 실데이터 재현 | 비고 |
|---|---|---|
| **(a) 래스터화 감쇠** (object→cost ≪1) | ✅ **견고** | 전 카테고리 gain 0.007~0.065 ≪1 (합성 0.004) |
| **(b) planner 증폭** (cost→plan >1) | ⚠️ **부분** | FOLLOWING(합성 아날로그)만 2.68 재현; 밀집 intersection/lane-change는 <1 |
| **(c) cost-map > object 안전임계** | ✅ **견고** | 전 카테고리 재현 (plan dev·mindist·ttc) |
| **(d) planner-switch 게이트웨이** | ❌ **미재현** | argmin flip 0.89(희소 합성)→**0.02~0.23**(밀집 실장면). 게이트웨이는 희소장면 현상 |
| **(e) directed > random** | ❌ **미재현** | 거의 동률(CARLA와 동일 — 견고한 건 일관성이지 raw strength 아님) |

**핵심 결론:** **전파 CHARACTERIZATION의 구조(래스터화=감쇠기, cost-map=최고 임계 인터페이스)는 실 nuPlan 데이터에서
견고하게 일반화된다.** planner 증폭은 car-following(합성 아날로그)에서 재현되고 밀집 상호작용 장면에서 약해진다.
**단, planner-switch 게이트웨이(C4)와 directed>random은 실데이터에서 재현되지 않는다** — 밀집·현실 장면에서는
단일-agent 결함이 전역 argmin을 거의 바꾸지 못함(합성 희소장면의 특이성). 효과 크기도 실데이터에서 전반적으로 작음
(plan dev ~0.07 m, mindist drop ~0.002 m; fault 0.5~2 m 대비). — **긍정·부정 모두 정직 보고.**

---

## 5. 3-Pillar 방법론 (최종 기여 구조)

1. **Representation-aware Fault Injection** — 표현별 자연스러운 연산자로 주입 (object/cost-map/BEV).
2. **Semantic Contract Preservation** — 구조는 유효, 의미만 왜곡 (위반형은 baseline).
3. **Cross-representation Error Propagation Analysis** — 동일 지표로 주입 단계별 안전 임계도·전파를 분석.

---

## 6. 정직한 한계

- 합성 planner/시나리오는 단순화됨(현실 반영 제한). pedestrian_crossing은 판별력이 약함.
- CARLA 실측은 단순 controller + ground-truth 인지 기반(전체 perception 스택 아님), 단일 맵(Town10HD),
  cut_in clean이 noisy(~13% 충돌). directed vs random의 실측 분리는 "충돌수"가 아니라 "분산/일관성"에서 견고.
- InterFuser BEV 연구는 open-loop feature 민감도(실제 CARLA closed-loop 아님) — 합성 closed-loop 결과의 보완.
- "지향적 변위" 연산자 자체는 신규성이 낮음 — 기여는 **fault model + 방법론 + 공정한 교차표현 검증**에 있음.

---

## 7. 수치 데이터 출처 (CSV)

| 결과 (§) | 파일 |
|---|---|
| 4.1 합성 fair baseline | `results/metrics/synthetic_summary.csv` |
| 4.2 directedness gap | `results/metrics/ablation_directedness.csv` |
| 4.1 계약보존/illegal swirl | `results/metrics/ablation_contract.csv` |
| 4.3 leaderboard | `results/metrics/unified_leaderboard.csv` |
| 4.4 BEV (InterFuser) | `results/bev_feature_torsion_phase4b.json` |
| 4.5 주입지점 민감도 | `results/metrics/unified_injection_sensitivity.csv` |
| 4.6 CARLA 실측 (450 ep) | `results/metrics/carla_runs.csv` |
| 4.7 전파 지표 (CIS/FAR/gain) | `results/metrics/propagation_cis.csv`, `propagation_map.csv`, `propagation_stage_errors.csv` |
| 4.8 Prediction 단계 (6-stage 체인) | `results/metrics/propagation_pred_{cis,map,stage_errors,horizon}.csv` |
| 4.9 메커니즘 (M1/M2/M3) | `results/metrics/mechanism_{rasterization,decision_margin,prediction_jacobian}.csv` |
| 4.10 Transfer-function 특성화 | `results/metrics/transfer_function_{gains,linearity}.csv` |
| 4.11 Failure taxonomy | `results/metrics/failure_taxonomy_{runs,paths}.csv` |
| 4.12 Planner-invariance (argmin↔softmax) | `results/metrics/planner_invariance.csv` |
| 4.13 Planner-independence (sampling vs potential-field) | `results/metrics/planner_independence.csv` |
| 4.14 실데이터 일반화 (nuPlan open-loop) | `results/metrics/nuplan_propagation_{runs,summary}.csv` |

(대응 그림이 필요하면 `results/figures/` 의 Fig. 3–15 참조.)

---

## 8. 포지셔닝

- **프레이밍:** interpretable safety evaluation / fault injection (adversarial attack 아님).
- **타깃:** Dependability 계열(DSN, ISSRE) 1순위 — fault model 기여가 정확히 맞음.
- **확장성:** 구조화된 중간표현 + 안전/결정 출력을 갖는 도메인(로보틱스, VLM 에이전트, 의료 AI)으로 자연 일반화.

---

### 한 줄 결론
> **작은 semantic fault가, 구조적 계약을 지키면서도, 자율주행의 안전 실패로 일관되고 해석 가능하게 전파된다 —
> 그리고 그 전파는 주입하는 표현 단계에 따라 체계적으로 달라진다.** (합성 3표현 + 실제 CARLA에서 검증)
