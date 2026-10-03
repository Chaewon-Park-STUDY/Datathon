# -*- coding: utf-8 -*-
"""
run_sim.py — 팀원 구조 시뮬레이션(sim.py) 실행 래퍼

하는 일:
  1) sim.py 를 import 하고, 하드코딩된 데이터 경로 sim.P 를 ./epoch_data/ 로 덮어쓴다.
     (sim.py 원본은 수정하지 않는다. 런타임에 모듈 변수만 교체한다.)
  2) L = sim.load_long() 로 롱폼 데이터를 만들고,
     prev = train_labels.csv 의 2026-08-25 공식 라벨(Series: channel_id -> 0/1) 을 만든 뒤,
     R = sim.simulate(L, "2026-09-01", W=26, prev=prev, W_prev=18, S=8000, seed=0) 을 호출한다.
  3) R.p(조건부 확률)와 R.p0(prev 미사용)을 channel_id 별로 뽑아,
     sample_submission.csv 의 row_id 순서에 맞춰 sim_p.csv (컬럼 row_id, prediction=R.p) 로 저장한다.
     sim 이 eligibility 미달로 제외한 채널(누락 row)은 최저값으로 채워 694 행 전체를 보장한다.
  4) 참고용으로 R.p0 도 sim_p0.csv 로 함께 저장한다.

주의:
  - sim.simulate 의 기본 인자 vb_mult=0.0 이므로 fit_boost(use_vb=False) 가 되어
    chan_vb()(channel_snapshots 의 total_view_count 컬럼 필요)는 호출되지 않는다.
    현재 배포된 channel_snapshots.csv 에는 total_view_count 가 없으므로 vb 경로는 건드리지 않는다.
  - 미래 정보 미사용(누수 없음) 설계는 sim.py 그대로다.

실행:
  python run_sim.py                      # 기본 S=8000
  python run_sim.py --S 3000             # 느리면 S 축소
  python run_sim.py --data ./epoch_data --out sim_p.csv --S 8000 --seed 0
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="sim.py 실행 래퍼: 2026-09-01 기준 제출 확률 생성")
    ap.add_argument("--data", default="./epoch_data", help="epoch_data 디렉터리 경로 (기본 ./epoch_data)")
    ap.add_argument("--out", default="sim_p.csv", help="조건부 확률(R.p) 출력 파일 (기본 sim_p.csv)")
    ap.add_argument("--out-p0", default="sim_p0.csv", help="참고용 R.p0 출력 파일 (기본 sim_p0.csv)")
    ap.add_argument("--D", default="2026-09-01", help="기준일 D (기본 2026-09-01)")
    ap.add_argument("--W", type=int, default=26, help="미래 창 길이(일), 본선=26 (기본 26)")
    ap.add_argument("--W-prev", type=int, default=18, help="prev 구간 길이(일) (기본 18)")
    ap.add_argument("--prev-date", default="2026-08-25", help="prev 라벨 기준일(train_labels) (기본 2026-08-25)")
    ap.add_argument("--S", type=int, default=8000, help="몬테카를로 표본 수 (기본 8000; 느리면 3000)")
    ap.add_argument("--seed", type=int, default=0, help="난수 시드 (기본 0)")
    args = ap.parse_args()

    # --- 의존성/경로 방어 ---
    try:
        import numpy as np  # noqa: F401
        import pandas as pd
    except ImportError as e:
        sys.exit("[run_sim] 필수 패키지(pandas/numpy/scipy)가 없습니다: %s\n"
                 "        로컬에서 `pip install 'numpy<2' pandas scipy` 후 실행하세요." % e)

    data_dir = os.path.abspath(args.data)
    if not os.path.isdir(data_dir):
        sys.exit("[run_sim] 데이터 폴더를 찾을 수 없습니다: %s" % data_dir)

    # sim.py 는 P + "파일명" 으로 접근하므로 끝에 구분자가 필요하다.
    P = data_dir + os.sep
    for fn in ("video_5d_views.csv", "channel_snapshots.csv", "sample_submission.csv"):
        if not os.path.isfile(os.path.join(data_dir, fn)):
            sys.exit("[run_sim] 필수 파일 누락: %s" % os.path.join(data_dir, fn))

    labels_path = os.path.join(data_dir, "train_labels.csv")
    if not os.path.isfile(labels_path):
        sys.exit("[run_sim] train_labels.csv 가 없어 prev 라벨을 만들 수 없습니다: %s" % labels_path)

    # sim.py 를 이 스크립트와 같은 폴더에서 import 할 수 있게 보장
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import sim
    except ImportError as e:
        sys.exit("[run_sim] sim.py 를 import 할 수 없습니다: %s" % e)

    # --- 하드코딩 경로 교체 (원본 수정 없이 런타임 변수만) ---
    sim.P = P
    sim._CS = None  # 혹시 이전 호출 캐시가 있으면 초기화 (chan_vb 캐시)
    print("[run_sim] sim.P 를 %r 로 설정" % sim.P)

    # --- prev 라벨 Series (channel_id -> 0/1), 기준일 2026-08-25 공식 라벨 ---
    lab = pd.read_csv(labels_path)
    # row_id 형식: "<channel_id>_<YYYY-MM-DD>"
    suffix = "_" + args.prev_date
    sel = lab[lab.row_id.astype(str).str.endswith(suffix)].copy()
    if sel.empty:
        sys.exit("[run_sim] train_labels.csv 에 %s 기준 라벨이 없습니다." % args.prev_date)
    sel["channel_id"] = sel.row_id.astype(str).str.slice(0, -len(suffix))
    prev = sel.set_index("channel_id")["target"].astype(float)
    prev = prev[~prev.index.duplicated(keep="last")]
    print("[run_sim] prev 라벨(%s): 채널 %d개, 양성 %d개" % (args.prev_date, len(prev), int((prev == 1).sum())))

    # --- 데이터 로드 및 시뮬레이션 ---
    print("[run_sim] load_long() ...")
    L = sim.load_long()
    print("[run_sim] simulate(D=%s, W=%d, W_prev=%d, S=%d, seed=%d) ..."
          % (args.D, args.W, args.W_prev, args.S, args.seed))
    R = sim.simulate(L, args.D, W=args.W, prev=prev, W_prev=args.W_prev, S=args.S, seed=args.seed)
    # R 는 channel_id 색인 DataFrame. p=조건부확률, p0=prev미사용.
    print("[run_sim] simulate 완료: %d개 채널에 대해 p 산출" % len(R))

    # --- sample_submission row_id 순서에 맞춰 저장 ---
    ss = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    ss.columns = [c.strip().lstrip("\ufeff") for c in ss.columns]  # BOM 방어
    if "row_id" not in ss.columns:
        sys.exit("[run_sim] sample_submission.csv 에 row_id 컬럼이 없습니다: %s" % list(ss.columns))

    # row_id -> channel_id (D suffix 제거)
    d_suffix = "_" + args.D
    def rid_to_chan(rid):
        rid = str(rid)
        return rid[:-len(d_suffix)] if rid.endswith(d_suffix) else rid.rsplit("_", 1)[0]
    ss["channel_id"] = ss.row_id.map(rid_to_chan)

    def build(col, fill_desc):
        series = R[col]
        vals = ss.channel_id.map(series)
        n_missing = int(vals.isna().sum())
        if n_missing:
            # eligibility 미달 등으로 sim 이 제외한 채널 → 최저값으로 채워 694행 보장
            fill = float(series.min()) if len(series) else 0.0
            vals = vals.fillna(fill)
            print("[run_sim] %s: 누락 채널 %d개를 최저값 %.6g 로 채움 (%s)"
                  % (col, n_missing, fill, fill_desc))
        return vals.clip(0.0, 1.0)

    out = ss[["row_id"]].copy()
    out["prediction"] = build("p", "조건부 확률")
    out_p0 = ss[["row_id"]].copy()
    out_p0["prediction"] = build("p0", "prev 미사용 확률")

    # --- 제출 형식 점검 ---
    def check(df, name):
        ok_rows = len(df) == len(ss)
        ok_cols = list(df.columns) == ["row_id", "prediction"]
        ok_nan = int(df.prediction.isna().sum()) == 0
        ok_range = bool(df.prediction.between(0, 1).all())
        ok_dup = int(df.row_id.duplicated().sum()) == 0
        print("[check:%s] rows=%d(%s) cols_ok=%s nan_ok=%s in[0,1]=%s no_dup=%s min=%.6g max=%.6g"
              % (name, len(df), ok_rows, ok_cols, ok_nan, ok_range, ok_dup,
                 float(df.prediction.min()), float(df.prediction.max())))
        if not (ok_rows and ok_cols and ok_nan and ok_range and ok_dup):
            sys.exit("[run_sim] 제출 형식 점검 실패: %s" % name)

    out.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    out_p0.to_csv(args.out_p0, index=False, encoding="utf-8", lineterminator="\n")
    check(out, args.out)
    check(out_p0, args.out_p0)
    print("[run_sim] 저장 완료: %s (R.p), %s (R.p0)" % (args.out, args.out_p0))
    print("[run_sim] 다음 단계: python blend_all.py")


if __name__ == "__main__":
    main()
