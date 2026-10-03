# 성능 개선 플래그 설명 (run_final.py)

> 목표: 리더보드 PR-AUC(Average Precision) 0.7075 → 0.74 이상.
> 모델 구조(로지스틱 + 순위변환 + 소프트 라벨)는 그대로 유지하고, 아래 개선을 **독립적인 argparse 플래그**로 추가했다.
> **플래그를 하나도 켜지 않으면 기존 0.7075 파이프라인이 그대로 재현된다**(기본 off).

## ⚠️ 실행/검증 안내 (중요)
- 이 코드는 **정적으로 작성·문법검증(`py_compile`)만** 되어 있다. 작성 환경에 pandas/numpy/scikit-learn/scipy가 없고
  네트워크가 막혀 있어 **파이프라인을 실제로 실행·검증하지 못했다.**
- 따라서 **반드시 로컬(pandas 설치된 환경)에서 아래 명령으로 직접 검증**해야 한다.
- 모든 신규 피처/가중은 **D 이전 데이터만** 사용하도록 작성해 누수(leakage)가 없도록 했으나,
  검증 수치로 최종 확인하라.

```
pip install pandas numpy scikit-learn scipy
```

## 채택 기준 (HANDOFF 명시)
- 18일 창 시간검증(`--F 18 --prev_F 18 --mode latest --validate`)에서 **9일 평균 PR-AUC > 0.570**(현재 최종)이고,
  **대부분의 날 개선 + 큰 하락 없음**이면 채택.
- 검증 +0.009 ≈ 리더보드 +0.002 로 축소 반영됨.
- **KFold shuffle 금지.** 검증은 항상 위 `--validate`(시간순 9일) 로만 한다.
- 최종 제출 파일은 반드시 `--F 26` 으로 생성.

## 기준(baseline) 검증 명령
먼저 현재 최종의 검증 점수를 기록해 비교 기준으로 삼는다.
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate
# → "평균 ROC .... PR 0.570" 근처가 나와야 함 (이게 비교 기준)
```

---

## 개선 1 — 라벨 정합성: `--label_fix`
공식 08-11 라벨과의 1.5% 불일치 원인 중 하나로 의심되는 **상위 20% 선정의 동점(tie) 처리**를 교정한다.
기존 `s >= s.quantile(.8)` 비교는 임계값에 동점이 몰리면 20%보다 많거나 적게 1을 줄 수 있다.
`--label_fix` 는 가장 큰 `s`부터 `ceil(0.2*n)`개만 1로 두어 **정확히 상위 20%**(결정적, 누수 없음)를 선정한다.

검증:
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --label_fix
```
채택: 9일 평균 PR > 0.570 이고 큰 하락 없음. (라벨 생성 자체를 바꾸므로 다른 플래그와 조합 전에 단독으로 먼저 평가 권장.)

---

## 개선 2 — lag=1 오답 교정: `--lag1feat`
상위권 오답의 74%가 lag=1(지난주 정답이었으나 이번엔 안 오른 채널)이다. 이들의 지난주 상승이
**'일회성 1편 히트'인지 '여러 편 고른 상승'인지** 구분하는 피처 2개를 메인 모델에 추가한다
(lag 창 중 **이미 관측된** 부분 `[prev_D, D)` 만 사용 → 누수 없음):
- `obs_conc` = 관측 상위영상 집중도 = 최고 조회수 / 관측 영상 조회수 합 (1편뿐이면 1.0, 고르면 낮음)
- `obs_nrise` = 과거 중앙값 P를 넘긴 관측 영상 수 (여러 편이 고르게 올랐으면 큼)

둘 다 날짜별 순위변환 후(`rk_obs_conc`, `rk_obs_nrise`) 메인 로지스틱 입력에 들어가며 MICE로 결측 보정된다.

검증:
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat
```
채택: 9일 평균 PR > 0.570, lag=1 많은 날의 하락이 없을 것.

---

## 개선 3 — AP 직접 최적화: `--ap_weight <배수>`
상위권(양성) 가중 손실. 양성(y=1) 행의 `sample_weight`를 지정 배수만큼 **추가로 곱한다**.
기존 가중치 체계(소프트 라벨 가중 `w` × 시간감쇠)와 **곱셈 호환**을 유지하므로,
하드 양성(w=1)과 소프트 양성 행(y=1, w=q) 모두 자연스럽게 상위로 끌어올려 상위 정밀도를 높인다.
기본값 1.0 = 변화 없음. 권장 탐색: 1.5, 2.0, 3.0.

검증:
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --ap_weight 2.0
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --ap_weight 1.5
```
채택: 9일 평균 PR > 0.570, 큰 하락 없음. 과하면(예: 3.0↑) 보정이 깨질 수 있으니 값을 쓸어보며 최적점 선택.

---

## 개선 4 — q-모형 통합 및 EM: `--q_uni`, `--q_em <반복>`
소프트 라벨의 핵심인 q-모형(앞 k일만 본 정보 → 최종 정답 확률)을 개선한다.
- `--q_uni` : k별 분리 학습 대신 **k/F(kfrac)를 입력 피처로 넣은 통합 q-모형**(피처 소수).
  모든 관측길이 k의 예제를 한 번에 학습해 데이터 효율·안정성을 높이고 짧은 k에서도 정보를 공유한다.
- `--q_em N` : **완전 EM**. 최종 로지스틱 예측으로 소프트 라벨 q를 N회 갱신하며 재학습(기본 0=안 함).

검증:
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --q_uni
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --q_em 1
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --q_uni --q_em 1
```
채택: 9일 평균 PR > 0.570, 큰 하락 없음. (`--q_em` 은 반복이 많을수록 느려지고 과적합 위험 → 1~2 권장.)

---

## 조합 평가 및 최종 제출
각 플래그를 단독으로 평가해 채택 기준을 넘긴 것만 조합한다. 조합도 반드시 `--validate` 로 재확인:
```
# 예: 개선1 + 개선2 + ap_weight 가 각각 통과했다면 조합 검증
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --label_fix --lag1feat --ap_weight 2.0
```

최종 제출 생성(채택된 플래그만, 반드시 `--F 26`). 기존 2-모델 순위 반반 블렌딩 유지:
```
# a = lb7627 계열, b = latest 계열. 채택된 신규 플래그를 두 명령 모두에 동일하게 붙인다.
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode lb7627 --q_x --soft_k [채택플래그] --out a.csv
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode latest        [채택플래그] --out b.csv
# 최종 = 0.5*rank(a) + 0.5*rank(b)
```
`a`는 C=0.1, `b`는 C=0.015(기본)로 자동 설정된다.

> 주의: HANDOFF 5절 "효과 없던 것"(LightGBM/LambdaRank/survival/`--q_ext`/`--inter`/`--lag_int`/`--sim`/
> 24·48h속도/구독자증가율 등)은 반복하지 말 것. 위 신규 플래그와 섞지 말고 비교 기준도 그것들을 끈 상태로 둘 것.
