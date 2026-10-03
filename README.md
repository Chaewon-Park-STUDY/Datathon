# EPOCH 5th Datathon — Rising 예측 (본선, F=26)

- 상세 맥락/결과/실패 목록: **HANDOFF_for_AI.md** 먼저 읽기
- 현재 최종 후보: `FINAL8_F26_ens_half.csv` (LB 기준 0.7075 계열)
- 데이터: `epoch_data/` (필수 컬럼만 남긴 축소본, 원본과 예측 동일 확인)

## 재현
```
pip install pandas numpy scikit-learn scipy
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode lb7627 --q_x --soft_k --out a.csv
python run_final.py --data ./epoch_data --F 26 --prev_F 18 --mode latest --out b.csv
# 최종 = 0.5*rank(a) + 0.5*rank(b)
python run_final.py --data ./epoch_data --F 18 --prev_F 18 --mode latest --validate   # 시간 검증
```
`experiments/` = 시도했지만 채택 안 된 스크립트(참고용, 원본 대용량 파일 필요할 수 있음).
