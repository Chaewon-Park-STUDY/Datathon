# -*- coding: utf-8 -*-
"""
blend_aggr.py — mate(주축) + T01(ranklearn) + qjoint "공격적인 막판 1장" 후보 생성기

목표
  제출 기회가 1번만 남았다. 현재 확정 최고는
      blend_90 = mate 0.90 : qjoint 0.10 = 0.7412
  직전 blend_t01.py 후보들은 mate 주축이라 기준선과 ρ 0.994~0.999 로 거의 안 벌어졌다.
  그래서 이번엔 "최대한 공격적인" 버전이 필요하다. 사용자 목표는 0.75(매우 공격적),
  최고점 반영 방식으로 간주.

핵심 아이디어 (blend_t01.py 보다 더 공격적)
  PR-AUC 는 상위권 순서가 점수를 거의 다 결정한다. 그래서
    (1) T01 비중을 blend_t01 보다 더 높인 선형 블렌딩,
    (2) base rank 상위 k 개만 선택적으로 T01 로 흔드는 "상위권 재배치",
    (3) 기하평균/최댓값/곱보정 같은 비선형 결합
  으로 기준선과 상관이 더 많이 벌어진 후보를 다양하게 만든다.
  T01(blends/T01_ranklearn50_S01x50.csv) 은 ranklearn 모델로 sim 계열(mate)과 구조가 다르다.
  (사용자 로컬 측정: T01 vs mate 순위상관 0.8554, T01 vs qjoint 0.9355)

사용 소스 (전부 로컬에 있어야 함)
  mate.csv                           : 팀원 BEST_v2 (리더보드 0.7395)  ← 가장 신뢰, 주축
  blends/T01_ranklearn50_S01x50.csv  : ranklearn 모델 (리더보드 0.7032, mate 와 순위상관 0.8554)
  submit_qjoint.csv                  : 내 모델 q_joint (리더보드 0.7115)

블렌딩 공식 (기존 blend_t01.py / blend_pro.py / make_v2.py 와 완전히 동일)
  각 소스를 sample_submission.csv 의 row_id 순서로 reindex 후 rank(pct=True) 로 변환,
  최종 점수를 다시 rank(pct=True), 그리고 순위 보존 로지스틱으로 prediction 생성:
      rank  = rankpct( score )
      pred  = 1 / (1 + exp(-8 * (rank - 0.8)))
  8자리 반올림, utf-8, LF 줄바꿈(lineterminator="\n"). 694행 / 0~1 보장.
  row_id 불일치 시 최저값으로 채움.

출력
  blends_aggr/ 에 각 후보를 제출 형식 점검(694행 / NaN·Inf 없음 / 0~1 / 중복 없음) 후 저장.
  파일명에 구성을 명시.
  비교 기준선 blend_mate90_qj10_BASELINE.csv (= blend_90, 확정 LB 0.7412) 도 반드시 생성.
  각 후보가 기준선 대비 spearman 순위상관 / top20·top50·top100 겹침을 출력(scipy 없으면 pandas 폴백).
  상관 오름차순(= 공격적일수록 위)으로 정렬해 표로 출력.
  "가장 공격적이되 상위권 구조가 완전히 무너지진 않은"(spearman 0.90~0.96, top50 겹침 30 이상)
  후보 1~2개를 추천.

실행
  python blend_aggr.py
  python blend_aggr.py --data ./epoch_data
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="mate(주축)+T01(ranklearn)+qjoint 공격적 순위 블렌딩 — 막판 1장")
    ap.add_argument("--data", default="./epoch_data", help="epoch_data 경로 (sample_submission.csv 위치)")
    ap.add_argument("--mate", default="mate.csv", help="팀원 BEST_v2 (기본 mate.csv, LB 0.7395)")
    ap.add_argument("--t01", default="blends/T01_ranklearn50_S01x50.csv",
                    help="ranklearn 모델 (기본 blends/T01_ranklearn50_S01x50.csv, LB 0.7032)")
    ap.add_argument("--qjoint", default="submit_qjoint.csv", help="내 모델 출력 (기본 submit_qjoint.csv, LB 0.7115)")
    ap.add_argument("--outdir", default="blends_aggr", help="출력 폴더 (기본 blends_aggr/)")
    args = ap.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        sys.exit("[blend_aggr] 필수 패키지(pandas/numpy)가 없습니다: %s\n"
                 "           로컬에서 `pip install 'numpy<2' pandas` 후 실행하세요." % e)

    data_dir = os.path.abspath(args.data)
    ss_path = os.path.join(data_dir, "sample_submission.csv")
    if not os.path.isfile(ss_path):
        sys.exit("[blend_aggr] sample_submission.csv 를 찾을 수 없습니다: %s" % ss_path)
    ss = pd.read_csv(ss_path)
    ss.columns = [c.strip().lstrip("\ufeff") for c in ss.columns]
    if "row_id" not in ss.columns:
        sys.exit("[blend_aggr] sample_submission.csv 에 row_id 컬럼이 없습니다: %s" % list(ss.columns))
    order = ss.row_id.astype(str).values
    N = len(order)
    print("[blend_aggr] sample_submission rows=%d" % N)

    def load(path, label, required=True):
        """row_id 순서로 정렬한 rank(pct=True) 배열 반환. 필수 소스가 없으면 종료."""
        if not os.path.isfile(path):
            msg = "[blend_aggr] %s 없음 — '%s' 소스 누락" % (path, label)
            if required:
                sys.exit(msg + " (필수 소스입니다. 로컬 datathon 폴더에서 실행하세요.)")
            print(msg + " (건너뜀)")
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        if "row_id" not in df.columns or "prediction" not in df.columns:
            sys.exit("[blend_aggr] %s 형식 이상(컬럼 %s)" % (path, list(df.columns)))
        s = df.set_index(df.row_id.astype(str))["prediction"]
        s = s.reindex(order)
        if s.isna().any():
            miss = int(s.isna().sum())
            print("[blend_aggr] 경고: %s 에서 %d개 row_id 가 불일치 → 최저값으로 채움" % (path, miss))
            s = s.fillna(s.min() if s.notna().any() else 0.0)
        print("[blend_aggr] 로드: %-28s (%s), rows=%d" % (label, path, len(s)))
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
        print("   [%s] %-44s rows=%d min=%.5g max=%.5g uniq=%d"
              % (status, name, len(pred), float(np.min(pred)), float(np.max(pred)), int(pd.Series(pred).nunique())))
        return ok

    created = []   # (filename, pred_array, label)

    def build(fname, label, score_rank):
        """점수 배열로부터 제출 파일 생성(형식 점검 통과 시 저장)."""
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
    # 4) 비교 기준선: blend_90 (mate 0.90 : qjoint 0.10, T01 미포함) 재현 — 확정 LB 0.7412
    #    (먼저 만들어 두어야 이후 상관 비교의 기준이 된다)
    # ============================================================
    print("\n[blend_aggr] === 기준선(blend_90, LB 0.7412 재현): mate 90 : qjoint 10 ===")
    base_score = 0.90 * mate_r + 0.10 * q_r
    base_pred = build("blend_mate90_qj10_BASELINE.csv",
                      "기준선 mate90:qjoint10 (= blend_90, 확정 LB 0.7412)",
                      base_score)

    # ============================================================
    # 1) T01 고비중 선형 블렌딩 — mate 와 T01 을 선형 가중합, qjoint 는 0 또는 소량(8%).
    #    qjoint 8% 버전은 그만큼 mate 에서 차감한다.
    #    (mate, T01) 비중 = (0.70,0.30), (0.65,0.35), (0.60,0.40), (0.55,0.45), (0.50,0.50)
    # ============================================================
    print("\n[blend_aggr] === (1) T01 고비중 선형 블렌딩 (mate + T01, qjoint 0% / 8%) ===")
    linear_pairs = [
        (0.70, 0.30),
        (0.65, 0.35),
        (0.60, 0.40),
        (0.55, 0.45),
        (0.50, 0.50),
    ]
    for wm, wt in linear_pairs:
        # qjoint 0% 버전
        score0 = wm * mate_r + wt * t01_r
        fname0 = "blend_mate%02d_t01%02d_qj00.csv" % (int(round(wm * 100)), int(round(wt * 100)))
        label0 = "mate %.0f%% / T01 %.0f%% / qjoint 0%%" % (wm * 100, wt * 100)
        build(fname0, label0, score0)

        # qjoint 8% 버전 — mate 에서 8%p 차감
        wmq = wm - 0.08
        score8 = wmq * mate_r + wt * t01_r + 0.08 * q_r
        fname8 = "blend_mate%02d_t01%02d_qj08.csv" % (int(round(wmq * 100)), int(round(wt * 100)))
        label8 = "mate %.0f%% / T01 %.0f%% / qjoint 8%%" % (wmq * 100, wt * 100)
        build(fname8, label8, score8)

    # ============================================================
    # 2) 상위권 선택적 재배치
    #    먼저 mate 주축(mate 0.85 + qjoint 0.15)으로 base rank 를 만든다.
    #    base 기준 상위 k 개(값이 큰 k개) 행에 대해서만 T01 의 rankpct 를 alpha 로 섞어
    #    상위권만 더 흔든다. 나머지 행은 base 유지. 섞은 뒤 전체 재rank → 로지스틱.
    #    k = 50, 100 / alpha = 0.4, 0.6 조합 전부 생성.
    # ============================================================
    print("\n[blend_aggr] === (2) 상위권 선택적 재배치 (base=mate85:qj15, 상위 k 행에만 T01 혼합) ===")
    base2_score = 0.85 * mate_r + 0.15 * q_r
    base2_rank = pd.Series(base2_score).rank(pct=True).values  # 0~1 순위, 값이 클수록 상위
    for k in (50, 100):
        # base rank 기준 상위(= 값이 큰) k개 인덱스
        top_idx = np.argsort(base2_rank)[-k:]
        for alpha in (0.4, 0.6):
            score = base2_rank.copy()
            # 상위 k 행에만 T01 rankpct 를 alpha 가중으로 혼합
            score[top_idx] = (1.0 - alpha) * base2_rank[top_idx] + alpha * t01_r[top_idx]
            # 섞은 뒤 전체 재rank (to_pred 내부에서 다시 rank 하지만, 재배치 효과를 명시적으로 둔다)
            score = pd.Series(score).rank(pct=True).values
            fname = "blend_topreassign_k%03d_alpha%02d.csv" % (k, int(round(alpha * 100)))
            label = "상위권재배치 base(mate85:qj15), 상위 %d행에 T01 alpha=%.1f 혼합" % (k, alpha)
            build(fname, label, score)

    # ============================================================
    # 3) 비선형 결합
    #    - 기하평균:   score = sqrt( rankpct(mate) * rankpct(T01) )
    #    - 최댓값:     score = max( rankpct(mate), rankpct(T01) )  (상위권 강조)
    #      그리고 혼합: score = 0.5*rankpct(mate) + 0.5*max(...)
    #    - 곱 보정:    score = rankpct(mate) * (0.5 + 0.5*rankpct(T01))
    # ============================================================
    print("\n[blend_aggr] === (3) 비선형 결합 (기하평균 / 최댓값 / 곱 보정) ===")
    geo = np.sqrt(mate_r * t01_r)
    build("blend_geomean_mate_t01.csv", "기하평균 sqrt(mate*T01)", geo)

    mx = np.maximum(mate_r, t01_r)
    build("blend_max_mate_t01.csv", "최댓값 max(mate,T01) (상위권 강조)", mx)

    mix_max = 0.5 * mate_r + 0.5 * mx
    build("blend_half_mate_half_max.csv", "0.5*mate + 0.5*max(mate,T01)", mix_max)

    prod = mate_r * (0.5 + 0.5 * t01_r)
    build("blend_mate_prodcorr_t01.csv", "곱 보정 mate*(0.5+0.5*T01)", prod)

    # ============================================================
    # 결과 요약: 기준선 대비 순위 상관(= 얼마나 다른지) + 상위 겹침, 상관 오름차순 정렬
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

        rows = []
        for fname, pred, label in created:
            if fname == "blend_mate90_qj10_BASELINE.csv":
                continue
            rho = rho_fn(pred, base_pred)
            t20 = top_overlap(pred, base_pred, 20)
            t50 = top_overlap(pred, base_pred, 50)
            t100 = top_overlap(pred, base_pred, 100)
            rows.append((fname, rho, t20, t50, t100, label))

        # 상관 오름차순(= 가장 공격적인 후보가 위로)
        rows.sort(key=lambda r: r[1])

        print("\n기준선(blend_mate90_qj10_BASELINE, LB 0.7412) 대비 순위 상관 / 상위 겹침")
        print("(상관 오름차순 — 위일수록 더 공격적):")
        print("  %-40s %8s %6s %6s %7s   %s" % ("후보", "spearman", "top20", "top50", "top100", "구성"))
        for fname, rho, t20, t50, t100, label in rows:
            print("  %-40s %8.4f %6d %6d %7d   %s" % (fname, rho, t20, t50, t100, label))

        # 추천 로직: "가장 공격적이되 상위권 구조가 완전히 무너지진 않은" 후보 1~2개.
        #   조건: spearman 0.90~0.96 구간, top50 겹침 30 이상 유지.
        #   그 안에서 상관이 낮을수록(= 더 공격적) 우선.
        picks = [r for r in rows if 0.90 <= r[1] <= 0.96 and r[3] >= 30]
        picks.sort(key=lambda r: r[1])  # 상관 낮은 순

        print("\n[추천 단일 제출 후보] (spearman 0.90~0.96, top50 겹침 ≥ 30)")
        if picks:
            for fname, rho, t20, t50, t100, label in picks[:2]:
                print("  >>> %s" % fname)
                print("      %s" % label)
                print("      기준선 대비 spearman=%.4f, top50 겹침=%d, top100 겹침=%d (공격적이되 상위 구조 유지)"
                      % (rho, t50, t100))
        else:
            print("  해당 구간(spearman 0.90~0.96 & top50≥30)에 드는 후보가 없습니다.")
            print("  → 가장 가까운 후보들을 참고하거나, 보수적으로 기준선을 제출하세요.")

        print("\n  보수적 선택을 원하면 기준선 자체(blend_mate90_qj10_BASELINE.csv, 확정 0.7412)를 제출하세요.")

    print("\n[해석 가이드]")
    print("  - PR-AUC 는 상위권 순서가 점수를 거의 결정하므로, top50/top100 겹침이 핵심 안전지표.")
    print("  - spearman ~1.000 이면 기준선과 거의 동일 → 0.7412 에서 거의 안 움직임(공격성 부족).")
    print("  - spearman 0.90~0.96 & top50 겹침 ≥ 30 이면 '상위 구조는 유지하되 공격적으로 흔든' 상태.")
    print("  - spearman 0.90 미만 또는 top50 겹침 급락이면 T01(단독 0.7032) 쪽으로 과도 이동 → 1장 올인 위험.")
    print("  - 1장만 제출하므로, 추천 후보와 기준선 중 하나를 고르세요. 안전=기준선, 상방 베팅=추천 후보.")
    print("\n[주의] 이 샌드박스에는 pandas/numpy 가 없어 실행 검증을 못 했습니다(문법 검증만 수행).")
    print("        로컬(pandas, numpy<2)에서 `python blend_aggr.py` 로 실제 파일을 생성·점검하세요.")


if __name__ == "__main__":
    main()
