# -*- coding: utf-8 -*-
"""
blend_all.py — sim 재현 확률과 각 모델 제출을 순위(rank) 기반으로 블렌딩

입력 (로컬에 있는 것만 사용, 없으면 건너뛴다):
  sim_p.csv        : run_sim.py 가 생성한 sim 직접재현 조건부 확률 (R.p)
  submit_qjoint.csv: 내 모델(q_joint) 출력 (리더보드 0.7115)
  mate.csv         : 팀원 BEST_v2 (sim .80 + T01 .20, 리더보드 0.7395)
  mate_v3.csv      : 팀원 BEST_v3 (sim 단독)

블렌딩 방식:
  각 입력을 sample_submission.csv 의 row_id 순서로 정렬한 뒤 rank(pct=True) 로 변환하고,
  두 소스를 가중 평균한다. 블렌딩 점수도 다시 rank(pct=True) 로 변환해 순위만 보존한다.
  (PR-AUC/ROC-AUC 는 순위에만 의존하므로 순위 보존 변환이면 충분하다.)

  blend_w(A, B, w) = rankpct( w * rankpct(A) + (1 - w) * rankpct(B) )
  여기서 A = 팀원/sim 쪽(주 모델), B = 내 q_joint. w 가 클수록 A 비중이 크다.

핵심 비교 대상:
  - mate × q_joint, w=0.90  → 지금까지 최고 0.7412(blend_90) 재현
  - sim_p × q_joint         → sim 직접재현으로 더 정교한 결합(가중 탐색)

출력:
  blend_<A>_x_qjoint_w<ww>.csv  형태로 가중 스윕(0.80~0.95) 파일 생성.
  각 파일은 제출 형식(694행, row_id/prediction, 0~1) 점검을 통과해야 저장된다.

실행:
  python blend_all.py
  python blend_all.py --data ./epoch_data --w-min 0.80 --w-max 0.95 --w-step 0.05
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="sim/팀원/내 모델 순위 기반 블렌딩")
    ap.add_argument("--data", default="./epoch_data", help="epoch_data 경로 (sample_submission.csv 위치)")
    ap.add_argument("--qjoint", default="submit_qjoint.csv", help="내 모델 출력 (기본 submit_qjoint.csv)")
    ap.add_argument("--sim", default="sim_p.csv", help="sim 재현 확률 (run_sim.py 산출, 기본 sim_p.csv)")
    ap.add_argument("--mate", default="mate.csv", help="팀원 BEST_v2 (기본 mate.csv)")
    ap.add_argument("--mate-v3", default="mate_v3.csv", help="팀원 BEST_v3 (기본 mate_v3.csv)")
    ap.add_argument("--outdir", default="blends", help="블렌딩 출력 폴더 (기본 blends/)")
    ap.add_argument("--w-min", type=float, default=0.80, help="A(주 모델) 가중 최소 (기본 0.80)")
    ap.add_argument("--w-max", type=float, default=0.95, help="A(주 모델) 가중 최대 (기본 0.95)")
    ap.add_argument("--w-step", type=float, default=0.05, help="가중 스텝 (기본 0.05)")
    args = ap.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        sys.exit("[blend] 필수 패키지(pandas/numpy)가 없습니다: %s\n"
                 "       로컬에서 `pip install 'numpy<2' pandas` 후 실행하세요." % e)

    data_dir = os.path.abspath(args.data)
    ss_path = os.path.join(data_dir, "sample_submission.csv")
    if not os.path.isfile(ss_path):
        sys.exit("[blend] sample_submission.csv 를 찾을 수 없습니다: %s" % ss_path)
    ss = pd.read_csv(ss_path)
    ss.columns = [c.strip().lstrip("\ufeff") for c in ss.columns]
    if "row_id" not in ss.columns:
        sys.exit("[blend] sample_submission.csv 에 row_id 컬럼이 없습니다: %s" % list(ss.columns))
    order = ss.row_id.astype(str).values
    N = len(order)

    def rankpct(series):
        """row_id 순서로 정렬한 값을 rank(pct=True) 로 변환해 numpy 배열로 반환."""
        return series.rank(pct=True).values

    def load(path, label):
        if not os.path.isfile(path):
            print("[blend] (건너뜀) %s 없음 — '%s' 소스는 블렌딩에서 제외" % (path, label))
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        if "row_id" not in df.columns or "prediction" not in df.columns:
            print("[blend] (건너뜀) %s 형식 이상(컬럼 %s)" % (path, list(df.columns)))
            return None
        s = df.set_index(df.row_id.astype(str))["prediction"]
        s = s.reindex(order)
        if s.isna().any():
            miss = int(s.isna().sum())
            print("[blend] 경고: %s 에서 %d개 row_id 가 sample_submission 과 불일치 → 최저값으로 채움" % (path, miss))
            s = s.fillna(s.min() if s.notna().any() else 0.0)
        print("[blend] 로드: %-18s (%s), rows=%d" % (label, path, len(s)))
        return rankpct(s)

    # 소스 로드
    sim_r = load(args.sim, "sim_p")
    q_r = load(args.qjoint, "q_joint")
    mate_r = load(args.mate, "mate(BEST_v2)")
    matev3_r = load(args.mate_v3, "mate_v3(sim단독)")

    if q_r is None:
        print("[blend] 주의: q_joint(%s) 가 없어 'x q_joint' 블렌딩을 만들 수 없습니다." % args.qjoint)

    os.makedirs(args.outdir, exist_ok=True)

    # 가중 스윕 목록
    ws = []
    w = args.w_min
    while w <= args.w_max + 1e-9:
        ws.append(round(w, 4))
        w += args.w_step

    def to_sub(score_rank):
        """순위 점수(0~1)를 make_v2 와 동일한 순위 보존 로지스틱 변환으로 prediction 생성."""
        rr = pd.Series(score_rank).rank(pct=True).values
        pred = 1.0 / (1.0 + np.exp(-8.0 * (rr - 0.8)))
        out = ss[["row_id"]].copy()
        out["prediction"] = np.round(pred, 8)
        return out

    def check(df, name):
        ok = (len(df) == N and list(df.columns) == ["row_id", "prediction"]
              and int(df.prediction.isna().sum()) == 0
              and bool(df.prediction.between(0, 1).all())
              and int(df.row_id.duplicated().sum()) == 0)
        status = "OK" if ok else "FAIL"
        print("   [%s] %-40s rows=%d min=%.5g max=%.5g" % (status, name, len(df), df.prediction.min(), df.prediction.max()))
        return ok

    created = []

    def sweep(A_rank, A_name, B_rank, B_name):
        """A(주 모델) 가중 w, B 가중 (1-w) 로 스윕 파일 생성."""
        if A_rank is None or B_rank is None:
            return
        print("[blend] === %s × %s 가중 스윕 ===" % (A_name, B_name))
        for wv in ws:
            score = wv * A_rank + (1.0 - wv) * B_rank
            sub = to_sub(score)
            fn = os.path.join(args.outdir, "blend_%s_x_%s_w%02d.csv"
                              % (A_name, B_name, int(round(wv * 100))))
            if check(sub, os.path.basename(fn)):
                sub.to_csv(fn, index=False, encoding="utf-8", lineterminator="\n")
                created.append((fn, A_name, B_name, wv))

    # 핵심: mate × q_joint (0.7412=blend_90 재현), sim_p × q_joint (직접재현)
    sweep(mate_r, "mate", q_r, "qjoint")
    sweep(sim_r, "sim", q_r, "qjoint")
    sweep(matev3_r, "matev3", q_r, "qjoint")
    # 보조: sim_p × mate (sim 재현과 팀원 제출 결합)
    sweep(sim_r, "sim", mate_r, "mate")

    # 소스 단독 제출 파일도 순위 보존 형태로 보관(비교용)
    for rk, nm in [(sim_r, "sim"), (mate_r, "mate"), (matev3_r, "matev3"), (q_r, "qjoint")]:
        if rk is None:
            continue
        sub = to_sub(rk)
        fn = os.path.join(args.outdir, "only_%s.csv" % nm)
        if check(sub, os.path.basename(fn)):
            sub.to_csv(fn, index=False, encoding="utf-8", lineterminator="\n")

    # --- 안내 ---
    print("\n================ 제출 안내 ================")
    if not created:
        print("블렌딩 파일을 만들지 못했습니다. 입력 파일(sim_p.csv/submit_qjoint.csv/mate.csv 등)을 확인하세요.")
    else:
        print("생성된 블렌딩 파일: %d개 (폴더: %s/)" % (len(created), args.outdir))
    print("\n[권장 제출 순서]")
    print("  1) blend_mate_x_qjoint_w90.csv  — 기존 최고(0.7412) 재현. 베이스라인 확인용으로 먼저 제출.")
    print("  2) blend_sim_x_qjoint_w90.csv   — sim 직접재현 × q_joint, w=0.90. 1)과 같은 비율로 재현 품질 비교.")
    print("  3) blend_sim_x_qjoint_w85.csv / _w95.csv — sim 비중을 ±5%p 흔들어 더 좋은 지점 탐색.")
    print("  4) 2~3) 중 가장 높은 w 를 찾은 뒤, 그 근처(예: w88/w92)를 추가 제출해 미세 조정.")
    print("\n블렌딩 공식: prediction = logistic(8*(rank-0.8)), rank = rankpct( w*rankpct(A) + (1-w)*rankpct(B) )")
    print("A = 주 모델(mate/sim/matev3), B = q_joint. w 가 클수록 주 모델 비중이 큽니다.")
    print("리더보드로만 최종 확인하므로, 위 순서대로 제출해 점수를 비교하세요.")


if __name__ == "__main__":
    main()
