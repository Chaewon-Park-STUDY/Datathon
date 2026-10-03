# -*- coding: utf-8 -*-
"""
blend_t01.py — mate(주축) + T01(ranklearn, 구조적으로 다름) + qjoint(극소량) "막판 1장" 후보 생성기

목표
  제출 기회가 1번만 남았다. 현재 확정 최고는
      blend_90 = mate 0.90 : qjoint 0.10 = 0.7412
  경쟁팀은 0.7449. 이 1장으로 0.7449 를 넘겨야 한다.

핵심 아이디어
  T01(blends/T01_ranklearn50_S01x50.csv) 은 ranklearn 모델로, sim 계열(mate)과 구조가 다르다.
  (사용자 로컬 측정: T01 vs mate 순위상관 0.8554, T01 vs qjoint 0.9355)
  구조적으로 다른 신호를 소량 섞으면 앙상블 다양성으로 상승 여지가 있으나,
  T01 단독 LB 가 0.7032 로 약해 많이 섞으면 하락 위험. 그래서 mate 를 주축으로 두고
  T01 을 소량(10~25%), qjoint 를 극소량(0~10%)만 섞는다.

사용 소스 (전부 로컬에 있어야 함)
  mate.csv                           : 팀원 BEST_v2 (리더보드 0.7395)  ← 가장 신뢰, 주축
  blends/T01_ranklearn50_S01x50.csv  : ranklearn 모델 (리더보드 0.7032, mate 와 순위상관 0.8554)
  submit_qjoint.csv                  : 내 모델 q_joint (리더보드 0.7115)

블렌딩 공식 (기존 blend_pro.py / make_v2.py 와 완전히 동일)
  각 소스를 sample_submission.csv 의 row_id 순서로 reindex 후 rank(pct=True) 로 변환,
  가중합한 뒤 다시 rank(pct=True), 그리고 순위 보존 로지스틱으로 prediction 생성:
      rank  = rankpct( sum_i w_i * rankpct(src_i) )
      pred  = 1 / (1 + exp(-8 * (rank - 0.8)))
  8자리 반올림, utf-8, LF 줄바꿈(lineterminator="\n"). 694행 / 0~1 보장.
  row_id 불일치 시 최저값으로 채움.

출력
  blends_t01/ 에 각 후보를 제출 형식 점검(694행 / NaN·Inf 없음 / 0~1 / 중복 없음) 후 저장.
  파일명에 비중을 명시(예: blend_mate80_t0120_qj00.csv).
  비교 기준선 blend_mate90_qj10_BASELINE.csv (= blend_90, 확정 LB 0.7412) 도 반드시 생성.
  각 후보가 기준선 대비 spearman 순위상관 / top20·top50 겹침을 출력(scipy 없으면 pandas 폴백).
  "기준선과 유의미하게 다르되 과이탈 아닌" 후보 1개를 추천.

실행
  python blend_t01.py
  python blend_t01.py --data ./epoch_data
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="mate(주축)+T01(ranklearn)+qjoint 순위 블렌딩 — 막판 1장")
    ap.add_argument("--data", default="./epoch_data", help="epoch_data 경로 (sample_submission.csv 위치)")
    ap.add_argument("--mate", default="mate.csv", help="팀원 BEST_v2 (기본 mate.csv, LB 0.7395)")
    ap.add_argument("--t01", default="blends/T01_ranklearn50_S01x50.csv",
                    help="ranklearn 모델 (기본 blends/T01_ranklearn50_S01x50.csv, LB 0.7032)")
    ap.add_argument("--qjoint", default="submit_qjoint.csv", help="내 모델 출력 (기본 submit_qjoint.csv, LB 0.7115)")
    ap.add_argument("--outdir", default="blends_t01", help="출력 폴더 (기본 blends_t01/)")
    args = ap.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        sys.exit("[blend_t01] 필수 패키지(pandas/numpy)가 없습니다: %s\n"
                 "          로컬에서 `pip install 'numpy<2' pandas` 후 실행하세요." % e)

    data_dir = os.path.abspath(args.data)
    ss_path = os.path.join(data_dir, "sample_submission.csv")
    if not os.path.isfile(ss_path):
        sys.exit("[blend_t01] sample_submission.csv 를 찾을 수 없습니다: %s" % ss_path)
    ss = pd.read_csv(ss_path)
    ss.columns = [c.strip().lstrip("\ufeff") for c in ss.columns]
    if "row_id" not in ss.columns:
        sys.exit("[blend_t01] sample_submission.csv 에 row_id 컬럼이 없습니다: %s" % list(ss.columns))
    order = ss.row_id.astype(str).values
    N = len(order)
    print("[blend_t01] sample_submission rows=%d" % N)

    def load(path, label, required=True):
        """row_id 순서로 정렬한 rank(pct=True) 배열 반환. 필수 소스가 없으면 종료."""
        if not os.path.isfile(path):
            msg = "[blend_t01] %s 없음 — '%s' 소스 누락" % (path, label)
            if required:
                sys.exit(msg + " (필수 소스입니다. 로컬 datathon 폴더에서 실행하세요.)")
            print(msg + " (건너뜀)")
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        if "row_id" not in df.columns or "prediction" not in df.columns:
            sys.exit("[blend_t01] %s 형식 이상(컬럼 %s)" % (path, list(df.columns)))
        s = df.set_index(df.row_id.astype(str))["prediction"]
        s = s.reindex(order)
        if s.isna().any():
            miss = int(s.isna().sum())
            print("[blend_t01] 경고: %s 에서 %d개 row_id 가 불일치 → 최저값으로 채움" % (path, miss))
            s = s.fillna(s.min() if s.notna().any() else 0.0)
        print("[blend_t01] 로드: %-28s (%s), rows=%d" % (label, path, len(s)))
        return s.rank(pct=True).values

    # --- 필수 소스 로드 ---
    mate_r = load(args.mate, "mate(BEST_v2 0.7395)", required=True)
    t01_r = load(args.t01, "T01(ranklearn 0.7032)", required=True)
    q_r = load(args.qjoint, "qjoint(0.7115)", required=True)

    os.makedirs(args.outdir, exist_ok=True)

    def to_pred(score_rank):
        """순위 점수를 순위 보존 로지스틱 변환으로 prediction 배열 생성(make_v2 식)."""
        rr = pd.Series(score_rank).rank(pct=True).values
        return np.round(1.0 / (1.0 + np.exp(-8.0 * (rr - 0.8))), 8)

    def check(pred, name):
        """제출 형식 점검: 694행 / NaN·Inf 없음 / 0~1 / 중복 없음."""
        ok = (len(pred) == N
              and int(pd.isna(pred).sum()) == 0
              and int(np.isinf(pred).sum()) == 0
              and bool(((pred >= 0) & (pred <= 1)).all())
              and int(pd.Series(order).duplicated().sum()) == 0)
        status = "OK" if ok else "FAIL"
        print("   [%s] %-36s rows=%d min=%.5g max=%.5g uniq=%d"
              % (status, name, len(pred), float(np.min(pred)), float(np.max(pred)), int(pd.Series(pred).nunique())))
        return ok

    created = []   # (filename, pred_array, label)

    def build(fname, label, score_rank):
        """가중합 순위 점수로부터 제출 파일 생성(형식 점검 통과 시 저장)."""
        pred = to_pred(score_rank)
        if not check(pred, fname):
            print("   -> 형식 점검 실패로 저장하지 않음: %s" % fname)
            return None
        out = ss[["row_id"]].copy()
        out["prediction"] = pred
        path = os.path.join(args.outdir, fname)
        out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        created.append((fname, pred, label))
        return pred

    # ============================================================
    # 1) 비교 기준선: blend_90 (mate 0.90 : qjoint 0.10, T01 미포함) 재현 — 확정 LB 0.7412
    # ============================================================
    print("\n[blend_t01] === 기준선(blend_90, LB 0.7412 재현): mate 90 : qjoint 10 ===")
    base_score = 0.90 * mate_r + 0.10 * q_r
    base_pred = build("blend_mate90_qj10_BASELINE.csv",
                      "기준선 mate90:qjoint10 (= blend_90, 확정 LB 0.7412)",
                      base_score)

    # ============================================================
    # 2) mate(주축) + T01(소량) + qjoint(극소량) 가중치 후보 (합=1.0)
    #    (mate, T01, qjoint)
    # ============================================================
    print("\n[blend_t01] === mate 주축 + T01 소량 + qjoint 극소량 후보 ===")
    candidates = [
        (0.90, 0.10, 0.00),
        (0.85, 0.15, 0.00),
        (0.80, 0.20, 0.00),
        (0.75, 0.25, 0.00),
        (0.82, 0.10, 0.08),
        (0.78, 0.15, 0.07),
        (0.72, 0.20, 0.08),
        (0.85, 0.10, 0.05),
        (0.80, 0.15, 0.05),
        (0.75, 0.20, 0.05),
    ]
    for wm, wt, wq in candidates:
        score = wm * mate_r + wt * t01_r + wq * q_r
        fname = "blend_mate%02d_t01%02d_qj%02d.csv" % (
            int(round(wm * 100)), int(round(wt * 100)), int(round(wq * 100)))
        label = "mate %.0f%% / T01 %.0f%% / qjoint %.0f%%" % (wm * 100, wt * 100, wq * 100)
        build(fname, label, score)

    # ============================================================
    # 안내 + 기준선 대비 순위 상관(= 얼마나 다른지) + 상위 겹침
    # ============================================================
    print("\n================ 결과 요약 ================")
    print("생성된 후보: %d개 (폴더: %s/)" % (len(created), args.outdir))

    if base_pred is None:
        print("경고: 기준선(blend_90)을 생성하지 못해 상관 비교를 건너뜁니다.")
    else:
        # scipy 가 있으면 spearman, 없으면 pandas rank + corrcoef 로 폴백
        try:
            from scipy.stats import spearmanr

            def rho_fn(a, b):
                return float(spearmanr(a, b)[0])
        except ImportError:
            def rho_fn(a, b):
                ra = pd.Series(a).rank().values
                rb = pd.Series(b).rank().values
                return float(np.corrcoef(ra, rb)[0, 1])

        def top_overlap(a, b, k):
            return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k]))

        print("\n기준선(blend_mate90_qj10_BASELINE, LB 0.7412) 대비 순위 상관 / 상위 겹침:")
        print("  %-36s %8s %7s %7s   %s" % ("후보", "spearman", "top20", "top50", "구성"))
        rows = []
        for fname, pred, label in created:
            if fname == "blend_mate90_qj10_BASELINE.csv":
                continue
            rho = rho_fn(pred, base_pred)
            t20 = top_overlap(pred, base_pred, 20)
            t50 = top_overlap(pred, base_pred, 50)
            rows.append((fname, rho, t20, t50, label))
            print("  %-36s %8.4f %7d %7d   %s" % (fname, rho, t20, t50, label))

        # 추천 로직: 기준선과 "의미 있게 다르되 과하게 벗어나지 않은" 후보.
        # 상관이 너무 1에 가까우면 기준선과 사실상 동일(개선 여지 적음),
        # 너무 낮으면 검증 안 된(약한 T01) 방향으로 과도하게 이동(리스크 큼).
        # 0.985~0.998 구간을 선호하며, 그 안에서 가장 유망한 후보를 1순위로 둔다.
        def score_candidate(rho):
            target = 0.993
            return -abs(rho - target)
        best = None
        for fname, rho, t20, t50, label in rows:
            sc = score_candidate(rho)
            if best is None or sc > best[0]:
                best = (sc, fname, rho, label)

        print("\n[추천 단일 제출 후보]")
        if best is not None:
            print("  >>> %s" % best[1])
            print("      %s" % best[3])
            print("      기준선 대비 spearman=%.4f (0.7412 와 유의미하게 다르되 과이탈 아님)" % best[2])
        print("\n  보수적 선택을 원하면 기준선 자체(blend_mate90_qj10_BASELINE.csv, 확정 0.7412)를 제출하세요.")

    print("\n[해석 가이드]")
    print("  - spearman 이 ~1.000 이면 기준선과 거의 동일 → 0.7412 에서 거의 안 움직임(개선 기대 작음).")
    print("  - spearman 이 0.98~0.99 대면 '검증된 mate 주축을 유지하면서' T01 다양성을 준 상태 → 리스크 중간.")
    print("  - spearman 이 0.95 미만이면 T01(단독 0.7032) 쪽으로 과도하게 이동 → 1장 올인에는 위험.")
    print("  - 1장만 제출하므로, 추천 후보와 기준선 중 하나를 고르세요. 안전=기준선, 상방 베팅=추천 후보.")
    print("\n[주의] 이 샌드박스에는 pandas/numpy 가 없어 실행 검증을 못 했습니다(문법 검증만 수행).")
    print("        로컬(pandas, numpy<2)에서 `python blend_t01.py` 로 실제 파일을 생성·점검하세요.")


if __name__ == "__main__":
    main()
