# 최종 제출 (Final Submission)

EPOCH 5th Datathon 본선 — 게임 유튜브 조회수 급등 예측 (PR-AUC)

## 최종 결과

| 항목 | 값 |
|------|-----|
| 최종 제출 | `blend_90` |
| 리더보드 PR-AUC | **0.7412** (한때 1위) |
| 출발점 | 2위, 0.7075 |

## 구성

```
final/
├── README.md                 # 이 파일
├── final_blend90.ipynb       # 최종 제출 재현 노트북 (blend_90)
└── presentation/
    └── (발표자료 PDF 를 여기에 둡니다)
```

> 재현에 필요한 입력 CSV(mate.csv, submit_qjoint.csv, epoch_data/)는
> 보안상 저장소에 포함하지 않습니다. 로컬에서 노트북 실행 시 사용합니다.

## 접근 요약

두 개의 서로 다른 모델 출력을 **순위(rank) 기반 90:10 블렌딩**하여
각 단일 모델보다 높은 점수를 얻었습니다.

- **mate** — 팀원 구조 시뮬레이션 모델 (BEST_v2), 단독 LB 0.7395
- **q_joint** — 로지스틱 + 순위 변환 기반 모델, 단독 LB 0.7115

### blend_90 공식

```
score = 0.90 * rankpct(mate) + 0.10 * rankpct(qjoint)
rr    = rankpct(score)
pred  = 1 / (1 + exp(-8 * (rr - 0.8)))
```

PR-AUC/ROC-AUC 는 예측값의 순위에만 의존하므로, 순위 보존 변환이면
점수가 동일하게 유지됩니다. 694행 / prediction 0~1 범위.

### 개선 과정

1. 2위(0.7075)에서 출발
2. 입력 특징 보강: `lag1feat`(직전 시점 특징), `lag1q`(직전 시점 분위)
3. `q_joint`(결합 로지스틱) 모델로 단일 모델 성능 개선 → 0.7115
4. 구조적으로 다른 팀원 시뮬레이션 모델(mate, 0.7395)과 순위 블렌딩(90:10)
5. 최종 **0.7412** (한때 리더보드 1위)

## 재현 방법

필요한 입력 파일:
1. `./epoch_data/sample_submission.csv`
2. `./mate.csv` (LB 0.7395)
3. `./submit_qjoint.csv` (LB 0.7115)

로컬 Jupyter(pandas, numpy < 2)에서 `final_blend90.ipynb` 를 위에서
아래로 실행하면 `blend_90_final.csv` 가 생성됩니다. 그 파일이 리더보드
0.7412 제출과 동일한 산출물입니다.
