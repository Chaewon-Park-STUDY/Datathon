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

# 2차 개선 플래그 (lag=1 오답 교정 심화)

> 1차에서 **`--lag1feat` 만 유효**했다(검증 9일 평균 PR 0.5695→**0.5716**, 리더보드 0.7075→0.7090).
> `--ap_weight`(하락), `--q_uni`/`--q_em`(중립)은 쓰지 않는다.
> 2차는 "lag 창에서 이미 관측된 상승의 '질'을 구분한다"는 통한 방향을 더 깊게 판다.
> 모두 **독립 플래그·기본 off**이며, **반드시 `--lag1feat` 와 함께** 평가한다(관측-질 신호를 재사용/보강하므로).
>
> **비교 기준(baseline for 2차)**: `--lag1feat` 켠 9일 평균 PR = **0.5716**. 이걸 넘겨야 2차 개선이 유효.
> **채택 조건**: `--lag1feat` 위에 얹었을 때 9일 평균 PR > 0.5716 + 대부분의 날 개선 + 큰 하락 없음. KFold shuffle 금지.

## ⚠️ 실행/검증 안내 (2차에도 동일)
- 2차 코드도 작성 환경(pandas/numpy/scikit-learn/scipy 없음, 네트워크 차단)에서 **실행·검증하지 못했다.** `py_compile` 문법검증만 했다.
- 모든 신규 피처는 **lag 창 중 D 이전 [prev_D, D) 관측부분만** 사용 → 누수 없음. 검증 수치로 최종 확인하라.
- 아무 2차 플래그도 켜지 않으면 1차까지의 파이프라인이 그대로 재현된다(기본 off).

## 2차-1 — 관측-질 피처 보강: `--lag1feat2`
`--lag1feat` 의 `obs_conc`(상위영상 집중도) / `obs_nrise`(과거중앙값 초과 편수)에 더해,
"일회성 1편 히트 vs 다편 고른 상승"을 더 잘 가르는 피처 2개를 메인 로지스틱에 추가한다
(모두 `[prev_D, D)` 관측부분만 → 누수 없음, 날짜별 순위변환 후 `rk_obs_second`, `rk_obs_exrise`, MICE 보정):
- `obs_second` = 관측 **2위/1위 조회수 비율** (1편뿐이면 0.0 = 극단 쏠림, 비슷한 2편 이상이면 1 근처 = 고른 상승)
- `obs_exrise` = 과거 중앙값 P 를 **넘긴 관측 영상들의 평균 로그 초과폭** (초과 영상 없으면 0.0) — '얼마나 세게' 넘었나

검증(반드시 `--lag1feat` 동반):
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1feat2
```
채택: 9일 평균 PR > 0.5716, 큰 하락 없음.

## 2차-2 — q-모형에 관측-질 신호 주입: `--lag1q`
소프트 라벨 q-모형(앞 k일만 본 정보 → 최종 정답 확률)은 지금 `obs_conc`/`obs_nrise` 신호를 쓰지 않는다.
`--lag1q` 는 `qfeat()` 에 이 신호(날짜별 순위 `rk_obs_conc`/`rk_obs_nrise`)와 `lag`(7일 전 정답)을 추가한다
(`X` 에서 날짜 d 기준으로 재사용 → `[prev_D, D)` 관측부분만, 누수 없음):
- `q_conc` = `rk_obs_conc` (관측 집중도 순위), `q_nrise` = `rk_obs_nrise` (상승 편수 순위), `q_lag` = `lag`
→ 잘린 날짜들의 소프트 라벨 q 추정 정확도 향상 기대. (`rk_obs_conc` 가 없으면, 즉 `--lag1feat` 를 안 켜면 자동 무시된다.)

검증(반드시 `--lag1feat` 동반):
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q
```
채택: 9일 평균 PR > 0.5716, 큰 하락 없음.

## 2차-3 — lag=1 과잉예측 억제: `--lag1damp`
상위권 오답의 **74%가 lag=1**(지난주 정답이었으나 이번엔 안 오른 채널, 평소 조회수 큼·최근 모멘텀 약함)이다.
이를 직접 겨냥해, lag=1 이면서 관측 상승이 **"집중(1편 쏠림) + 소수 상승"일수록 커지는** 상호작용을 메인 모델에 추가한다:
- `lag_weakobs` = `lag` × `rk_obs_conc` × `(1 − rk_obs_nrise)`
  (lag=1 & 1편 쏠림 집중도 높음 & 상승 편수 적음 → 큰 값). lag 결측(채점 대상 아님)은 0 → 상호작용 0.
→ 모델이 여기에 **음의 계수**를 학습하면, 믿을 만한 '다편 고른 상승' lag=1 은 두고 '일회성 상승' lag=1 만 깎아
  lag=1 과잉예측을 줄인다(믿을 만한 양성은 보존).

검증(반드시 `--lag1feat` 동반):
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1damp
```
채택: 9일 평균 PR > 0.5716, lag=1 많은 날의 하락이 없을 것.

## 2차 조합 평가
각 플래그를 `--lag1feat` 와 1:1 로 평가해 0.5716 을 넘긴 것만 조합한다. 조합도 `--validate` 재확인:
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1feat2 --lag1q --lag1damp
```

## 2차 최종 제출 (채택된 2차 플래그만, 반드시 `--F 26`, 2-모델 순위 반반 블렌딩)
`a`(lb7627 계열, C=0.1) 와 `b`(latest 계열, C=0.015) 를 만든 뒤, **두 출력의 `rank(pct=True)` 평균**으로 블렌딩한다
(1차에서 `pct` 누락으로 제출이 거부된 적 있으니 반드시 `pct=True` 범위 0~1 순위로 블렌딩할 것):
```
# [채택2차] = 1:1 검증을 통과한 2차 플래그들(예: --lag1feat --lag1damp). 두 명령에 동일하게 붙인다.
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode lb7627 --q_x --soft_k --lag1feat [채택2차] --out a.csv
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode latest             --lag1feat [채택2차] --out b.csv
```
블렌딩(파이썬):
```python
import pandas as pd
a = pd.read_csv('a.csv'); b = pd.read_csv('b.csv')
m = a.merge(b, on='row_id', suffixes=('_a', '_b'))
# 반드시 pct=True (0~1 범위) 순위 평균
m['prediction'] = 0.5 * m.prediction_a.rank(pct=True) + 0.5 * m.prediction_b.rank(pct=True)
m[['row_id', 'prediction']].to_csv('submission_blend.csv', index=False)
```

> 주의: HANDOFF 5절 "효과 없던 것"(LightGBM/LambdaRank/survival/`--q_ext`/`--inter`/`--lag_int`/`--sim`/
> 24·48h속도/구독자증가율 등) 과 1차에서 효과 없던 `--ap_weight`/`--q_uni`/`--q_em` 은 2차에서 쓰지 말 것.

---

## (1차) 조합 평가 및 최종 제출
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

---

# 3차 개선 플래그 (라벨/타깃 정합성 + q-모형 통합 개편)

> 1·2차에서 **`--lag1feat`(리더보드 0.7090)** 와 **`--lag1q`(검증 9일 평균 PR 0.5735, 지금까지 최고)** 가 통했다.
> 점수를 크게 움직인 유일한 과거 사례는 **소프트 라벨**(예선 0.68→0.76)이었고, lag1q 가 통한 것으로 보아
> **라벨·q-모형 쪽을 건드릴 때 큰 폭**이 난다. 그래서 3차는 **라벨 정합성 근본 교정 + q-모형 통합 개편**에 집중한다.
> 피처 추가와 달리 라벨/타깃 구조를 바로잡으면 전체 학습이 흔들리던 지점이 풀려 **큰 폭 상승 가능성**이 있다.
> (단, 0.73 도달을 보장하지는 않는다. 검증 +0.009 ≈ 리더보드 +0.002 로 축소 반영되므로, 0.73 엔 검증 PR ~0.66 가 필요하다.)
>
> 모두 **독립 플래그·기본 off**다. **아무 3차 플래그도 켜지 않으면 2차까지의 파이프라인이 그대로 재현된다.**
> 성능용 3차 플래그(`--label_measurable`, `--q_joint`)는 **반드시 `--lag1feat --lag1q` 와 함께** 평가한다.
>
> **비교 기준(baseline for 3차)**: `--lag1feat --lag1q` 9일 평균 PR = **0.5735**(현재 최고 조합). 이걸 넘겨야 3차가 유효.
> **채택 조건**: `--lag1feat --lag1q` 위에 얹었을 때 9일 평균 PR > 0.5735 + 대부분의 날 개선 + 큰 하락 없음. KFold shuffle 금지.

## ⚠️ 실행/검증 안내 (3차에도 동일)
- 3차 코드도 작성 환경(pandas/numpy/scikit-learn/scipy 없음, 네트워크 차단)에서 **실행·검증하지 못했다.** `py_compile` 문법검증만 했다.
- 모든 신규 교정/피처는 **D 이전 데이터만** 사용 → 누수 없음. `--label_measurable` 의 '측정 완료' 판정은
  영상 단위 고정 속성(`published_at + 5일 <= 데이터 끝`)이라 어떤 검증 fold 에서도 동일하게 D 이전 정보로 결정된다.
- **반드시 로컬에서 아래 명령으로 수치 확인하라.**

## 3차-진단 — 라벨 정합성 진단: `--label_diag`
공식 `train_labels.csv`(연습: 08-11, 08-25)와 재구성 라벨 `label()` 을 **채널별로 비교**해, 불일치를 원인별로 분해 출력하고 **즉시 종료**한다(학습/제출 영향 없음). 분해:
- **라벨값 불일치**: 공식·재구성 둘 다 채점 대상인데 0/1 이 다른 수(= 상위 20% 선정/점수 차이)
- **eligibility 불일치**: 한쪽만 채점 대상(과거 롱폼 ≥3편 & P≥100 경계·동점·측정 범위 차이)
- 참고: 과거 창에 **측정 미완**(`published_at + 5일 > 데이터 끝`) 롱폼이 섞인 채널 수

`--label_fix`/`--label_measurable` 조합을 바꿔가며 **불일치율이 줄어드는 조합**을 찾는다:
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --label_diag
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --label_diag --label_fix
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --label_diag --label_measurable
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --label_diag --label_fix --label_measurable
```
(진단은 `--mode`/`--validate` 와 무관하게 동작한다. 재구성 라벨이 공식과 가장 적게 어긋나는 교정 조합을 아래 성능 검증의 후보로 삼는다.)

## 3차-1 — 라벨 정합성 교정(측정 범위): `--label_measurable`
공식 08-11 라벨과의 1.5% 불일치 원인 후보 중 **"view_5d 측정 가능 범위"**를 겨냥한다.
업로드 5일 후 조회수(`view_5d`)가 끝까지 측정된 롱폼, 즉 **`published_at + 5일 <= 데이터 끝`** 인 영상만
과거 점수 `P`·**eligibility**(채점 대상 판정)·미래 점수 `Q` 계산에 쓴다. 측정이 덜 된 최근 영상이
P·eligibility·Q 를 흔드는 것을 막아 라벨을 공식과 정합시킨다. 관측-질 신호(`obs_*`)·구간 상·하한도 같은 기준으로 일관되게 계산된다.
- `--label_fix`(상위 20% 동점 처리)와 **독립**이라 조합 가능(`--label_diag` 로 어느 조합이 불일치를 가장 줄이는지 먼저 확인).
- 누수 없음: 측정 가능 여부는 영상 단위 고정 속성이며 D 이전 정보로 결정된다.

검증(반드시 `--lag1feat --lag1q` 동반):
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --label_measurable
# (불일치 진단에서 --label_fix 도 함께일 때 더 적게 어긋나면 아래도)
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --label_measurable --label_fix
```
채택: 9일 평균 PR > 0.5735, 큰 하락 없음.

## 3차-2 — q-모형 통합 개편: `--q_joint`
`--lag1q` 가 통한 방향(q-모형에 관측-질 신호 주입)을 **심화**한다. 과거에 단독으로 실패한
`--q_uni`(k/F 통합)·`--q_em`(EM)을 **개별이 아니라 하나의 통합 q-모형에서 lag1q 신호와 함께** 작동하도록 재설계한다:
- **k/F(kfrac) 통합**: 모든 관측길이 k 의 예제를 한 번에 학습(데이터 효율·안정성, 짧은 k 정보 공유).
- **관측-질 신호 결합**: `--lag1q` 가 주입하는 `q_conc`(관측 집중도 순위)·`q_nrise`(상승 편수 순위)·`q_lag`(7일 전 정답)을 그대로 쓰되,
- **상호작용**: `kfrac × q_conc`, `kfrac × q_nrise` 를 추가해 **"얼마나 봤는지(k/F)"와 "본 상승의 질"이 결합**해 q 를 결정하게 한다
  (적게 본 날엔 관측-질 신호를 조건부로 더/덜 믿도록 모델이 학습).
- 소프트 라벨 가중 `w`(`--soft_k` 의 관측비율)와 **완전 호환**.
- `--q_uni` 보다 우선 적용된다. 관측-질 신호가 없으면(=`--lag1q` 미사용) 상호작용이 사라져 `--q_uni` 와 동일하게 동작하므로,
  **반드시 `--lag1feat --lag1q` 와 함께** 켜 시너지를 평가할 것.

검증(반드시 `--lag1feat --lag1q` 동반):
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint
```
채택: 9일 평균 PR > 0.5735, 큰 하락 없음. (EM 을 더 섞어 보려면 `--q_em 1` 을 추가로 얹어 비교 가능하나, 과거 EM 단독 중립이었으니 통합 모형 자체를 먼저 평가.)

## 3차 조합 평가
각 3차 플래그를 `--lag1feat --lag1q` 위에 1:1 로 평가해 0.5735 를 넘긴 것만 조합한다. 조합도 `--validate` 재확인:
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --label_measurable --q_joint
```

## 3차 최종 제출 (채택된 3차 플래그만, 반드시 `--F 26`, 2-모델 순위 반반 블렌딩, `pct=True` 필수)
`a`(lb7627 계열, C=0.1) 와 `b`(latest 계열, C=0.015) 를 만든 뒤, **두 출력의 `rank(pct=True)` 평균(0~1 범위)**으로 블렌딩한다
(과거 `pct` 누락으로 제출이 거부된 적 있으니 **반드시 `pct=True`** 로 블렌딩할 것):
```
# [채택3차] = 1:1 검증을 통과한 3차 플래그들(예: --label_measurable --q_joint). 두 명령에 동일하게 붙인다.
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode lb7627 --q_x --soft_k --lag1feat --lag1q [채택3차] --out a.csv
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode latest             --lag1feat --lag1q [채택3차] --out b.csv
```
블렌딩(파이썬):
```python
import pandas as pd
a = pd.read_csv('a.csv'); b = pd.read_csv('b.csv')
m = a.merge(b, on='row_id', suffixes=('_a', '_b'))
# 반드시 pct=True (0~1 범위) 순위 평균 — pct 누락 시 제출 거부됨
m['prediction'] = 0.5 * m.prediction_a.rank(pct=True) + 0.5 * m.prediction_b.rank(pct=True)
m[['row_id', 'prediction']].to_csv('submission_blend.csv', index=False)
```

> 주의: HANDOFF 5절 "효과 없던 것"(LightGBM/LambdaRank/survival/`--q_ext`/`--inter`/`--lag_int`/`--sim`/
> 24·48h속도/구독자증가율 등)과 1차에서 효과 없던 `--ap_weight`/`--q_uni`/`--q_em`(단독)·2차 `--lag1feat2`/`--lag1damp`(하락)는 3차에서 쓰지 말 것.

---

# 4차 개선 플래그 (q-모형 + 관측신호 심화)

> 지금까지 통한 것: **`--lag1feat`(리더보드 0.7090)**, **`--lag1q`(0.7114)**, **`--q_joint`**.
> **현재 최고 조합 = `--lag1feat --lag1q --q_joint`**(18일 창 9일 평균 PR = **0.5743**, 리더보드 **0.7115**).
> 통한 광맥은 **"q-모형(소프트 라벨) + lag 창 관측-질 신호"** 한 줄기다. 4차는 이 광맥을 더 깊게 파되,
> 모델 구조(로지스틱 + 순위변환 + 소프트 라벨)는 그대로 유지하고 이미 실패한 모델교체/손실변경/라벨교정류는 반복하지 않는다.
>
> 모두 **독립 플래그·기본 off**다. **아무 4차 플래그도 켜지 않으면 3차까지의 파이프라인이 그대로 재현된다.**
> 성능용 4차 플래그(`--lag1q2`)는 **반드시 `--lag1feat --lag1q --q_joint` 와 함께** 평가한다.
>
> **비교 기준(baseline for 4차)**: `--lag1feat --lag1q --q_joint` 9일 평균 PR = **0.5743**(현재 최고 조합). 이걸 넘겨야 4차가 유효.
> **채택 조건**: 그 위에 얹었을 때 9일 평균 PR > **0.5743** + 대부분의 날 개선 + 큰 하락 없음. KFold shuffle 금지.
> (검증 +0.009 ≈ 리더보드 +0.002 로 축소 반영된다.)

## ⚠️ 실행/검증 안내 (4차에도 동일 — 작성 환경에서 실행 불가)
- 4차 코드도 작성 환경(pandas/numpy/scikit-learn/scipy 없음, 네트워크 차단으로 pip/uv 불가)에서 **실행·검증하지 못했다.** `py_compile` 문법검증만 했다.
- 모든 신규 신호는 **[d, end) 관측구간 또는 D 이전** 정보만 사용 → 누수 없음. **미래(end 이후)를 절대 참조하지 않는다.**
- **반드시 로컬(pandas 설치, numpy<2)에서 아래 명령으로 수치 확인하라.** 기존 라이브러리만 쓰며 새 의존성은 없다.

## 먼저: 현재 최고 조합의 기준 검증(비교 기준 0.5743 재확인)
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint
# → "평균 ... PR 0.5743" 근처가 나와야 함 (4차는 이 수치를 넘겨야 유효)
```

## 4차-1 — q-모형에 관측 상승의 '궤적/최신성' 신호 주입: `--lag1q2`
`--lag1q` 가 통한 방향(q-모형에 관측-질 신호 주입)을 **소수의 신호로** 더 판다. `qfeat()` 에 다음 2개를 추가하되,
**잘린(미래 창이 데이터 끝에서 잘린) 날짜에서도 유효**하도록 `[d, end)` 관측구간 또는 D 이전 정보만 쓴다(누수 없음):
- `q_traj` = **관측 상승의 궤적**. 관측 구간을 **앞 절반**만 본 순위 `rk_half` 대비, 관측 구간 **전체**를 본 순위 `rk` 의
  변화량(`rk − rk_half`). 올라오는 중이면 +, 식는 중이면 −. "지금 상승 궤적에 있는가"를 잘린 날에도 포착한다.
  (`--q_ext` 에 유사 로직이 있으나 `--q_ext` 전체는 과거 하락 기록이므로 켜지 말고 이 **단일 신호만** 쓴다.)
- `q_recent` = **최신 업로드 활동도**. X 에서 날짜 d 기준 재사용하는 `rk_last3_m`(최근 3편 모멘텀 순위, 없으면 중립 0.5)에
  `na_m7`(최근 7일 업로드 공백 지시; 공백이면 1)을 곱으로 반영 → 최근에 활발히 올리면서 모멘텀이 좋을수록 큼.

`--q_joint` 와 함께 켜면 통합 q-모형에 **`kfrac × q_traj` 상호작용**이 자동 추가되어, "얼마나 봤는지(k/F)"와 "상승 궤적"이
결합해 q 를 결정한다(적게 본 날엔 궤적을 조건부로 더/덜 믿도록 모델이 학습). `--lag1q` 를 켜지 않아도 동작하지만,
관측-질 신호와의 결합 시너지를 노린 설계이므로 **반드시 `--lag1q` 와 함께** 켠다.

검증(반드시 `--lag1feat --lag1q --q_joint` 동반):
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --lag1q2
```
채택: 9일 평균 PR > **0.5743**, 큰 하락 없음.

## 4차-2 — q_joint + EM 결합 검증: `--q_joint --q_em 1` (새 플래그 불필요)
`--q_joint`(통합 q-모형)와 `--q_em N`(완전 EM)은 **이미 코드상 공존**한다(별도 보정·새 플래그 불필요). 흐름:
1. `train_set` 이 `q_predict` → **`qmodel_joint`**(통합 q-모형)로 소프트 라벨 초기 q 를 만든다.
2. 그 위에서 `fit_predict_em` 이 **메인 로지스틱 예측으로 소프트 라벨 가중 `w` 를 N회 갱신**(EM)한다.
   즉 통합 q-모형이 만든 q 를 EM 이 메인 모델 신호로 다듬는다. `_sk`(관측비율)와도 호환.
> 과거 `--q_em` **단독**은 중립이었으나, 통합 q-모형(q_joint) 위에서의 결합은 **아직 미검증**이다. 아래로 확인한다
> (반복이 많을수록 느리고 과적합 위험 → **`--q_em 1` 부터**).
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --q_em 1
# lag1q2 까지 얹어서도 비교
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --lag1q2 --q_em 1
```
채택: 9일 평균 PR > **0.5743**, 큰 하락 없음. (중립/하락이면 `--q_em` 은 쓰지 않는다.)

## 4차 조합 평가
각 4차 후보를 `--lag1feat --lag1q --q_joint` 위에 1:1 로 평가해 **0.5743** 을 넘긴 것만 조합한다. 조합도 `--validate` 재확인:
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --lag1q2
# (q_em 결합이 통과했다면)
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --lag1q2 --q_em 1
```

## 4차-3 — 블렌딩 비율 탐색 (코드 변경 없음, 리더보드 제출로만 확인)
현재 최종은 두 모델 출력 `a`(lb7627 계열, C=0.1)·`b`(latest 계열, C=0.015)의 `rank(pct=True)` **반반(0.5:0.5)** 평균이다.
비율은 **시간검증으로 못 재므로**(검증은 단일 모델 1개만 돈다) **리더보드 제출로만** 확인한다. 하루 제출 횟수가 한정되니
**소수 비율만** 시험하라(예: `w = 0.3, 0.4, 0.6, 0.7`). `a.csv`/`b.csv` 를 **한 번만 생성**한 뒤, 아래 스니펫으로
여러 비율의 블렌딩 파일을 만들어 각각 제출해 비교한다(반드시 `pct=True`, 과거 `pct` 누락으로 제출 거부된 적 있음):
```python
import pandas as pd
a = pd.read_csv('a.csv'); b = pd.read_csv('b.csv')
m = a.merge(b, on='row_id', suffixes=('_a', '_b'))
ra = m.prediction_a.rank(pct=True)      # 반드시 pct=True (0~1 범위)
rb = m.prediction_b.rank(pct=True)
for w in (0.3, 0.4, 0.5, 0.6, 0.7):     # a 가중 w, b 가중 (1-w). 소수만 시험
    m['prediction'] = w * ra + (1 - w) * rb
    m[['row_id', 'prediction']].to_csv(f'submission_blend_a{int(w*100)}.csv', index=False)
    print('saved submission_blend_a%d.csv' % int(w * 100))
# 0.5:0.5(submission_blend_a50.csv)가 현재 최종. 양옆 비율을 제출해 리더보드가 오르는 쪽을 채택.
```
> 비율 탐색은 **채택된 4차 플래그로 생성한 `a.csv`/`b.csv`** 위에서 하라(아래 최종 제출 명령 참고).

## 4차-4 — 규제 C 재탐색 (코드 변경 없음, `--C` 인자 이용)
피처가 늘면(4차 신호 추가) 최적 규제 `C` 가 달라졌을 수 있다. `--C` 는 이미 인자로 존재한다
(기본: lb7627=0.1, latest=0.015). 각 모델별로 **기본값 주변**을 `--validate` 로 쓸어보고,
현재 최고 조합 대비 9일 평균 PR 이 오르는 값만 채택한다(두 모델을 **따로** 탐색).

latest 모델(C 기본 0.015) 주변 — 검증 명령:
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --C 0.008
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --C 0.015
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --C 0.025
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --C 0.04
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate --lag1feat --lag1q --q_joint --C 0.06
```
lb7627 모델(C 기본 0.1) 주변 — 검증 명령(이 계열은 `--q_x --soft_k` 를 쓴다):
```
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode lb7627 --q_x --soft_k --validate --lag1feat --lag1q --q_joint --C 0.05
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode lb7627 --q_x --soft_k --validate --lag1feat --lag1q --q_joint --C 0.1
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode lb7627 --q_x --soft_k --validate --lag1feat --lag1q --q_joint --C 0.2
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode lb7627 --q_x --soft_k --validate --lag1feat --lag1q --q_joint --C 0.3
```
> `--validate` 는 단일 모델 1개의 9일 평균 PR 만 낸다(블렌딩은 안 함). 각 모델에서 가장 높은 C 를 고른 뒤,
> 그 C 를 최종 제출 명령의 해당 모델에 `--C` 로 지정해 쓴다. 큰 하락이 없고 소폭이라도 오르는 값만 채택.

## 4차 최종 제출 (채택된 4차 플래그만, 반드시 `--F 26`, 2-모델 순위 반반 블렌딩, `pct=True` 필수)
`a`(lb7627 계열) 와 `b`(latest 계열) 를 만든 뒤, **두 출력의 `rank(pct=True)` 평균(0~1 범위)**으로 블렌딩한다
(과거 `pct` 누락으로 제출이 거부된 적 있으니 **반드시 `pct=True`**):
```
# [채택4차] = 1:1 검증(0.5743 초과)을 통과한 4차 플래그들(예: --lag1q2, 통과 시 --q_em 1). 두 명령에 동일하게 붙인다.
# C 재탐색(4차-4)에서 더 좋은 값을 찾았으면 각 모델에 --C 로 지정(아래는 기본값 예시).
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode lb7627 --q_x --soft_k --lag1feat --lag1q --q_joint [채택4차] --out a.csv
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode latest             --lag1feat --lag1q --q_joint [채택4차] --out b.csv
```
블렌딩(파이썬, 기본 0.5:0.5 — 비율 탐색은 4차-3 스니펫 사용):
```python
import pandas as pd
a = pd.read_csv('a.csv'); b = pd.read_csv('b.csv')
m = a.merge(b, on='row_id', suffixes=('_a', '_b'))
# 반드시 pct=True (0~1 범위) 순위 평균 — pct 누락 시 제출 거부됨
m['prediction'] = 0.5 * m.prediction_a.rank(pct=True) + 0.5 * m.prediction_b.rank(pct=True)
m[['row_id', 'prediction']].to_csv('submission_blend.csv', index=False)
```

> 주의: HANDOFF 5절 "효과 없던 것"(LightGBM/LambdaRank/survival/`--q_ext`(전체)/`--inter`/`--lag_int`/`--sim`/
> 24·48h속도/구독자증가율 등), 1차 `--ap_weight`/`--q_uni`/`--q_em`(단독), 2차 `--lag1feat2`/`--lag1damp`(하락),
> 그리고 라벨교정류(`--label_fix`/`--label_measurable` 가 3차에서 미채택이었다면)는 4차에서 반복하지 말 것.
> 4차는 **q-모형+관측신호(`--lag1q2`)와 q_joint+EM 결합·블렌딩비율·C 재탐색**에만 집중한다.
