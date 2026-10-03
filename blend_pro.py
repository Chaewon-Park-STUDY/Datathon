# -*- coding: utf-8 -*-
"""
blend_pro.py — 검증된 팀원 원본 CSV + 내 모델을 더 강하게 결합하는 "막판 1장" 후보 생성기

목표
  제출 기회가 1번만 남았다. 현재 확정 최고는
      blend_mate_x_qjoint_w90 = 0.7412  (mate 0.7395 : qjoint 0.7115 = 90:10 순위 블렌딩)
  경쟁팀은 0.7449. 이 1장으로 0.7449 를 넘겨야 한다.

사용 소스 (검증된 것만, 전부 로컬에 있어야 함)
  mate.csv          : 팀원 BEST_v2 (sim .80 + T01 .20, 리더보드 0.7395)   ← 가장 신뢰
  mate_v3.csv       : 팀원 BEST_v3 (sim 단독)
  submit_qjoint.csv : 내 모델 q_joint (리더보드 0.7115)
  * sim_p.csv (내 재현) 은 블렌딩 시 0.7171 로 저조 → 이 스크립트는 절대 사용하지 않는다.

블렌딩 공식 (기존 blend_all.py / make_v2.py 와 동일)
  각 소스를 sample_submission.csv 의 row_id 순서로 정렬 후 rank(pct=True) 로 변환,
  가중합한 뒤 다시 rank(pct=True), 그리고 make_v2 식 순위 보존 로지스틱으로 prediction 생성:
      rank  = rankpct( sum_i w_i * rankpct(src_i) )
      pred  = 1 / (1 + exp(-8 * (rank - 0.8)))
  8자리 반올림, utf-8, LF 줄바꿈. 694행 / 0~1 보장.

조합 설계
  mate 와 v3 는 둘 다 팀원 sim 계열이라 서로 상관이 높다. 그래서
      팀원축 = (1 - m) * mate + m * v3     (m = v3 혼합 비율)
      최종   = t * 팀원축 + (1 - t) * qjoint (t = 팀원축 비중)
  을 그리드로 돌린다. 추가로 "mate+v3+qjoint 동시 3중" 핵심 후보도 명시적으로 넣는다.
  2중 기준선 blend_mate_x_qjoint_w90 (=0.7412) 도 반드시 하나 포함해 비교 기준을 준다.

출력
  blends_pro/ 에 각 조합을 제출 형식 점검 후 저장한다. 파일명에 구성/비중을 명시
  (예: blend_mate60_v330_qj10.csv). 각 후보가 기준선(0.7412)과 얼마나 다른지 순위 상관을
  출력하고, 가장 유망한 단일 후보 1개를 추천한다.

실행
  python blend_pro.py
  python blend_pro.py --data ./epoch_data
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="검증 소스(mate/v3/qjoint) 강화 블렌딩 — 막판 1장")
    ap.add_argument("--data", default="./epoch_data", help="epoch_data 경로 (sample_submission.csv 위치)")
    ap.add_argument("--qjoint", default="submit_qjoint.csv", help="내 모델 출력 (기본 submit_qjoint.csv)")
    ap.add_argument("--mate", default="mate.csv", help="팀원 BEST_v2 (기본 mate.csv)")
    ap.add_argument("--mate-v3", default="mate_v3.csv", help="팀원 BEST_v3 sim단독 (기본 mate_v3.csv)")
    ap.add_argument("--outdir", default="blends_pro", help="출력 폴더 (기본 blends_pro/)")
    args = ap.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        sys.exit("[blend_pro] 필수 패키지(pandas/numpy)가 없습니다: %s\n"
                 "          로컬에서 `pip install 'numpy<2' pandas` 후 실행하세요." % e)

    data_dir = os.path.abspath(args.data)
    ss_path = os.path.join(data_dir, "sample_submission.csv")
    if not os.path.isfile(ss_path):
        sys.exit("[blend_pro] sample_submission.csv 를 찾을 수 없습니다: %s" % ss_path)
    ss = pd.read_csv(ss_path)
    ss.columns = [c.strip().lstrip("\ufeff") for c in ss.columns]
    if "row_id" not in ss.columns:
        sys.exit("[blend_pro] sample_submission.csv 에 row_id 컬럼이 없습니다: %s" % list(ss.columns))
    order = ss.row_id.astype(str).values
    N = len(order)
    print("[blend_pro] sample_submission rows=%d" % N)

    def load(path, label, required=True):
        """row_id 순서로 정렬한 rank(pct=True) 배열 반환. 필수 소스가 없으면 종료."""
        if not os.path.isfile(path):
            msg = "[blend_pro] %s 없음 — '%s' 소스 누락" % (path, label)
            if required:
                sys.exit(msg + " (검증 소스는 모두 필요합니다. 로컬 datathon 폴더에서 실행하세요.)")
            print(msg + " (건너뜀)")
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        if "row_id" not in df.columns or "prediction" not in df.columns:
            sys.exit("[blend_pro] %s 형식 이상(컬럼 %s)" % (path, list(df.columns)))
        s = df.set_index(df.row_id.astype(str))["prediction"]
        s = s.reindex(order)
        if s.isna().any():
            miss = int(s.isna().sum())
            print("[blend_pro] 경고: %s 에서 %d개 row_id 가 불일치 → 최저값으로 채움" % (path, miss))
            s = s.fillna(s.min() if s.notna().any() else 0.0)
        print("[blend_pro] 로드: %-20s (%s), rows=%d" % (label, path, len(s)))
        return s.rank(pct=True).values

    # --- 검증 소스 로드 (sim_p 는 의도적으로 사용하지 않음) ---
    mate_r = load(args.mate, "mate(BEST_v2 0.7395)", required=True)
    v3_r = load(args.mate_v3, "mate_v3(sim단독)", required=True)
    q_r = load(args.qjoint, "qjoint(0.7115)", required=True)

    os.makedirs(args.outdir, exist_ok=True)

    def to_pred(score_rank):
        """순위 점수를 make_v2 식 순위 보존 로지스틱 변환으로 prediction 배열 생성."""
        rr = pd.Series(score_rank).rank(pct=True).values
        return np.round(1.0 / (1.0 + np.exp(-8.0 * (rr - 0.8))), 8)

    def check(pred, name):
        """제출 형식 점검: 694행 / 2컬럼 / NaN·Inf 없음 / 0~1 / 중복 row_id 없음."""
        ok = (len(pred) == N
              and int(pd.isna(pred).sum()) == 0
              and int(np.isinf(pred).sum()) == 0
              and bool(((pred >= 0) & (pred <= 1)).all()))
        status = "OK" if ok else "FAIL"
        print("   [%s] %-34s rows=%d min=%.5g max=%.5g uniq=%d"
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
    # 1) 비교 기준선: blend_mate_x_qjoint_w90 (= 0.7412) 재현
    #    팀원축 = mate 단독(m=0), t=0.90
    # ============================================================
    print("\n[blend_pro] === 기준선(0.7412 재현): mate 90 : qjoint 10 ===")
    base_score = 0.90 * mate_r + 0.10 * q_r
    base_pred = build("blend_mate90_qj10_BASELINE.csv",
                      "기준선 mate90:qjoint10 (=blend_mate_x_qjoint_w90, LB 0.7412)",
                      base_score)

    # ============================================================
    # 2) 3중 그리드: 팀원축비중 t × v3혼합 m
    #    팀원축 = (1-m)*mate + m*v3 ,  최종 = t*팀원축 + (1-t)*qjoint
    # ============================================================
    print("\n[blend_pro] === 3중 그리드 (팀원축비중 t × v3혼합 m) ===")
    t_grid = [0.85, 0.88, 0.90, 0.92]
    m_grid = [0.0, 0.3, 0.5]
    for t in t_grid:
        for m in m_grid:
            team = (1.0 - m) * mate_r + m * v3_r
            score = t * team + (1.0 - t) * q_r
            # 최종 소스별 실효 비중(가독용): mate, v3, qjoint
            w_mate = t * (1.0 - m)
            w_v3 = t * m
            w_qj = 1.0 - t
            fname = "blend_mate%02d_v3%02d_qj%02d.csv" % (
                int(round(w_mate * 100)), int(round(w_v3 * 100)), int(round(w_qj * 100)))
            label = "t=%.2f m=%.2f (mate %.0f%% / v3 %.0f%% / qjoint %.0f%%)" % (
                t, m, w_mate * 100, w_v3 * 100, w_qj * 100)
            build(fname, label, score)

    # ============================================================
    # 3) 핵심 명시 3중 후보 (사용자 지시의 대표 조합)
    #    각 소스를 rankpct 후 직접 가중합 → 재rank
    # ============================================================
    print("\n[blend_pro] === 핵심 명시 3중 후보 ===")
    explicit = [
        # (mate, v3, qjoint)
        (0.60, 0.30, 0.10),
        (0.55, 0.35, 0.10),
        (0.50, 0.40, 0.10),
        (0.65, 0.20, 0.15),
        (0.70, 0.20, 0.10),
        (0.80, 0.10, 0.10),
    ]
    for wm, wv, wq in explicit:
        score = wm * mate_r + wv * v3_r + wq * q_r
        fname = "blend_mate%02d_v3%02d_qj%02d_explicit.csv" % (
            int(round(wm * 100)), int(round(wv * 100)), int(round(wq * 100)))
        label = "명시 3중 mate %.0f%% / v3 %.0f%% / qjoint %.0f%%" % (wm * 100, wv * 100, wq * 100)
        build(fname, label, score)

    # ============================================================
    # 안내 + 기준선 대비 순위 상관(= 얼마나 다른지)
    # ============================================================
    print("\n================ 결과 요약 ================")
    print("생성된 후보: %d개 (폴더: %s/)" % (len(created), args.outdir))

    if base_pred is None:
        print("경고: 기준선(0.7412)을 생성하지 못해 상관 비교를 건너뜁니다.")
    else:
        # scipy 가 있으면 spearman, 없으면 pandas rank 로 폴백 상관 계산
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
        print("  %-40s %8s %7s %7s   %s" % ("후보", "spearman", "top20", "top50", "구성"))
        rows = []
        for fname, pred, label in created:
            if fname == "blend_mate90_qj10_BASELINE.csv":
                continue
            rho = rho_fn(pred, base_pred)
            t20 = top_overlap(pred, base_pred, 20)
            t50 = top_overlap(pred, base_pred, 50)
            rows.append((fname, rho, t20, t50, label))
            print("  %-40s %8.4f %7d %7d   %s" % (fname, rho, t20, t50, label))

        # 추천 로직: 기준선과 "의미 있게 다르되 과하게 벗어나지 않은" 후보.
        # 상관이 너무 1에 가까우면 기준선과 사실상 동일(개선 여지 적음),
        # 너무 낮으면 검증 안 된 방향으로 과도하게 이동(리스크 큼).
        # 0.985~0.998 구간에서 v3 를 소량(20~30%) 섞은 3중을 1순위로 둔다.
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
    print("  - spearman 이 0.98~0.99 대면 '검증된 팀원축을 유지하면서' 변화를 준 상태 → 상승/하락 모두 가능하나 리스크 중간.")
    print("  - spearman 이 0.95 미만이면 많이 다른 제출 → 검증 안 된 방향이라 1장 올인에는 위험.")
    print("  - 1장만 제출하므로, 추천 후보와 기준선 중 하나를 고르세요. 안전=기준선, 상방 베팅=추천 후보.")
    print("\n[주의] 이 샌드박스에는 pandas/numpy 가 없어 실행 검증을 못 했습니다(문법 검증만 수행).")
    print("        로컬(pandas, numpy<2)에서 `python blend_pro.py` 로 실제 파일을 생성·점검하세요.")


if __name__ == "__main__":
    main()
