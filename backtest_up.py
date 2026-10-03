# -*- coding: utf-8 -*-
"""
backtest_up.py — (1) 그룹 A 변형 그리드 백테스트 + (2) FINAL20 상향 제출 생성

로컬 datathon 폴더(epoch_data/ 와 blend_90.csv 가 있는 곳)에서 실행한다.
  python backtest_up.py
  python backtest_up.py --data ./epoch_data --base blend_90.csv

배경
  직전 backtest_drop.py 백테스트에서 그룹 A("직전 라벨 target==1 이고 관측구간 롱폼 업로드
  0편인 채널 → 다음 라벨 양성률이 높다")가 다음 양성률 50%(전체 20.12%의 2.5배)로 유일하게
  신호를 보였지만 표본이 2개뿐이라 신뢰할 수 없었다. 이번엔 OBS/MAXUP 그리드로 표본을 15+ 로
  키워 신호가 유지되는지 검증하고, 신호가 유지되면 blend_90 에서 해당 테스트 채널들을 위로
  올리는 제출 후보(FINAL20_up.csv)를 만든다.

  라벨 날짜 D(=2026-08-11, 601행)를 "직전 라벨", D_next(=2026-08-25, 666행)를 "다음 라벨"로
  둔다. 전체 D_next 양성률은 약 0.2012.

[1] 그룹 A 변형 그리드 백테스트 (표본 확대가 핵심)
  OBS(관측창 일) × MAXUP(관측구간 롱폼 업로드 상한) 그리드로 조합별 다음 양성률/리프트를 표로
  출력한다. 직전 라벨 조건은 target==1 고정.
    - OBS   : 7, 14, 21, 28  → 관측구간 [D-OBS, D)
    - MAXUP : 0, 1, 2         → "관측구간 롱폼 업로드 수 <= MAXUP"
  각 (OBS, MAXUP) 조합에 대해 조건을 만족하는 채널 집합을 D에서 뽑고, 그 채널들의 D_next 양성률을
  전체 양성률과 비교한다. 표에 n(표본수)/양성률/리프트(그룹÷전체)를 모두 보여주고,
  "n>=15 이면서 양성률>=0.35" 를 만족하는 조합을 '신뢰 가능 신호'로 별도 표시한다.
  대조군으로 "라벨1인데 관측구간 롱폼 많이(>=5편) 올린 채널"의 다음 양성률도 참고 출력.

[2] FINAL20 생성 (신호가 있을 때만 의미, 없어도 파일은 만들되 경고)
  가장 신뢰도 높은 조합(신뢰 가능 신호 중 리프트 최대, 없으면 n이 가장 큰 조합)을 선택한다.
  그 조합과 동일한 조건을 테스트셋(2026-09-01)에 적용한다. 테스트 적용 시 "직전 라벨"은
  train_labels 의 가장 최근 날짜(08-25) target==1 을 쓰고, 관측구간은 데이터 최신일 기준
  [max_date-OBS, max_date) 롱폼 업로드 수 <= MAXUP.
  blend_90.csv 를 읽어 prediction 을 "그대로" 복사하고(재랭킹/로지스틱 재적용 절대 금지),
  선택된 테스트 채널들의 row_id(_2026-09-01) 에 대해서만 prediction 을 상위권(0.9+)으로 상향한다.
  상향식: new = max(cur, 0.5*cur + 0.5*0.95) 에 미세 차등을 더해 서로 구분. 나머지 채널은 blend_90
  과 비트단위 동일. 결과를 FINAL20_up.csv 로 저장(utf-8, LF).

보안: 데이터/제출 CSV 는 저장소에 올리지 않는다. 이 스크립트(.py)만 커밋 대상이며,
      FINAL20_up.csv 는 사용자가 로컬 실행 시 생성한다.
"""
import argparse
import os
import sys


# --- 백테스트 상수 (바꾸기 쉽게 상단에 둠) ---------------------------------
LABEL_DATE_PREV = "2026-08-11"   # 직전 라벨 날짜 D (train_labels 에 실제 존재하는 날짜를 코드로 재확인)
LABEL_DATE_NEXT = "2026-08-25"   # 다음 라벨 날짜 D_next
TEST_DATE = "2026-09-01"         # 테스트/제출 날짜 (blend_90 의 row_id 날짜)

OBS_GRID = [7, 14, 21, 28]       # 관측창 길이(일) 그리드
MAXUP_GRID = [0, 1, 2]           # 관측구간 롱폼 업로드 수 상한 그리드

# '신뢰 가능 신호' 판정 기준
MIN_N_TRUST = 15                 # 표본 하한
MIN_RATE_TRUST = 0.35            # 양성률 하한

CONTROL_MIN_UP = 5               # 대조군: 라벨1인데 관측구간 롱폼 >=5편

# --- [2] 상향 방식 상수 ---------------------------------------------------
UP_ANCHOR = 0.95                 # 상향 앵커값 (new = max(cur, 0.5*cur + 0.5*0.95))
UP_STEP = 1e-6                   # 서로 구분하기 위한 미세 차등 간격


def channel_of(row_id):
    """row_id("{channel_id}_{날짜}")에서 channel_id 추출.
    channel_id 자체엔 밑줄이 없고(UC 로 시작 24자) 날짜 앞에만 '_'가 있으므로
    rsplit('_', 1) 로 마지막 '_...' 만 떼어낸다."""
    return str(row_id).rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser(
        description="그룹 A 변형 그리드 백테스트 + FINAL20 상향 제출 생성")
    ap.add_argument("--data", default="./epoch_data", help="데이터 폴더 (기본 ./epoch_data)")
    ap.add_argument("--base", default="blend_90.csv", help="기준 제출 파일 (기본 blend_90.csv)")
    args = ap.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        sys.exit("[backtest_up] 필수 패키지(pandas/numpy)가 없습니다: %s\n"
                 "            로컬에서 `pip install 'numpy<2' pandas` 후 실행하세요." % e)

    data_dir = os.path.abspath(args.data)
    labels_path = os.path.join(data_dir, "train_labels.csv")
    views_path = os.path.join(data_dir, "video_5d_views.csv")
    base_path = os.path.abspath(args.base)

    # 필수 파일 점검 (명확한 에러로 종료)
    for p, lbl in [(labels_path, "train_labels.csv"),
                   (views_path, "video_5d_views.csv"),
                   (base_path, "기준 제출 파일(--base)")]:
        if not os.path.isfile(p):
            sys.exit("[backtest_up] 필수 파일을 찾을 수 없습니다: %s (%s)" % (p, lbl))

    def read_csv_clean(path):
        """CSV 로드 후 컬럼명 strip/BOM 제거 (기존 blend_* 스크립트 공통 스타일)."""
        df = pd.read_csv(path)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        return df

    # ------------------------------------------------------------
    # 공통: 라벨 로드
    # ------------------------------------------------------------
    labels = read_csv_clean(labels_path)
    if "row_id" not in labels.columns or "target" not in labels.columns:
        sys.exit("[backtest_up] train_labels.csv 형식 이상(컬럼 %s)" % list(labels.columns))
    labels["channel_id"] = labels["row_id"].map(channel_of)
    labels["label_date"] = labels["row_id"].map(lambda r: str(r).rsplit("_", 1)[-1])
    labels["target"] = pd.to_numeric(labels["target"], errors="coerce")

    # train_labels 에 실제 존재하는 날짜 분포 확인
    date_counts = labels["label_date"].value_counts().sort_index()
    print("=" * 64)
    print("[0] train_labels 날짜 분포")
    print("=" * 64)
    for d, c in date_counts.items():
        print("   %s : %d행" % (d, int(c)))

    d_prev, d_next = LABEL_DATE_PREV, LABEL_DATE_NEXT
    if d_prev not in date_counts.index or d_next not in date_counts.index:
        # 상수가 데이터와 안 맞으면 실제 존재하는 두 날짜(가장 이른 것=D, 그 다음=D_next)로 대체
        avail = list(date_counts.index)
        if len(avail) < 2:
            sys.exit("[backtest_up] 라벨 날짜가 2개 미만이라 D→D_next 백테스트 불가: %s" % avail)
        d_prev, d_next = avail[0], avail[1]
        print("\n[주의] 상수 날짜가 데이터에 없어 실제 날짜로 대체: D=%s, D_next=%s" % (d_prev, d_next))
    print("\n직전 라벨 D=%s, 다음 라벨 D_next=%s" % (d_prev, d_next))

    prev = labels[labels["label_date"] == d_prev].copy()
    nxt = labels[labels["label_date"] == d_next].copy()
    next_target = nxt.set_index("channel_id")["target"]  # D_next target 을 channel_id 로 조회
    overall_next_rate = float(nxt["target"].mean()) if len(nxt) else float("nan")
    print("전체 D_next 평균 양성률: %.4f (%d채널)" % (overall_next_rate, len(nxt)))

    # ------------------------------------------------------------
    # 공통: 영상 데이터 — 롱폼(is_shorts==0)만, 날짜 파싱
    # ------------------------------------------------------------
    views = read_csv_clean(views_path)
    need_cols = {"channel_id", "published_at", "view_5d", "is_shorts"}
    if not need_cols.issubset(set(views.columns)):
        sys.exit("[backtest_up] video_5d_views.csv 형식 이상(컬럼 %s)" % list(views.columns))
    views["published_at"] = pd.to_datetime(views["published_at"], errors="coerce")
    views["view_5d"] = pd.to_numeric(views["view_5d"], errors="coerce")
    # is_shorts 는 0/1(혹은 True/False) 모두 안전 처리 → 롱폼 = 거짓/0
    is_short = views["is_shorts"].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes"])
    longform = views[~is_short].copy()

    def obs_longform_count(anchor_ts, obs_window):
        """관측구간 [anchor_ts-obs_window, anchor_ts) 의 채널별 롱폼 업로드 편수(Series)."""
        start = anchor_ts - pd.Timedelta(days=int(obs_window))
        in_obs = longform[(longform["published_at"] >= start)
                          & (longform["published_at"] < anchor_ts)]
        return in_obs.groupby("channel_id").size()

    # ============================================================
    # [1] 그룹 A 변형 그리드 백테스트
    # ============================================================
    print("\n" + "=" * 64)
    print("[1] 그룹 A 변형 그리드 백테스트 (직전 라벨 target==1 고정)")
    print("=" * 64)
    print("조건: 관측구간 [D-OBS, D) 의 롱폼 업로드 수 <= MAXUP")
    print("전체 D_next 양성률 기준선: %.4f\n" % overall_next_rate)

    d_prev_ts = pd.to_datetime(d_prev)
    prev_pos = prev[prev["target"] == 1]["channel_id"].tolist()  # 직전 라벨 target==1 채널

    # OBS 별 관측구간 롱폼 편수 캐시 (OBS 가 바뀌면 구간도 바뀜)
    cnt_cache = {obs: obs_longform_count(d_prev_ts, obs) for obs in OBS_GRID}

    def eval_combo(obs, maxup):
        """(OBS, MAXUP) 조합: 조건 만족 채널의 D_next 양성률/리프트를 계산.
        반환 dict: n(표본수, D_next 라벨 존재 기준), rate, lift, channels(조건 만족 전체)."""
        cnt = cnt_cache[obs]
        # 직전 라벨 1 & 관측구간 롱폼 편수 <= MAXUP (업로드 0편이면 cnt 에 없음 → get 0)
        chans = [c for c in prev_pos if int(cnt.get(c, 0)) <= maxup]
        with_next = [c for c in chans if c in next_target.index]
        if len(with_next) == 0:
            return {"obs": obs, "maxup": maxup, "n_all": len(chans), "n": 0,
                    "rate": float("nan"), "lift": float("nan"), "channels": chans}
        vals = next_target.reindex(with_next)
        rate = float(vals.mean())
        lift = (rate / overall_next_rate) if overall_next_rate else float("nan")
        return {"obs": obs, "maxup": maxup, "n_all": len(chans), "n": len(with_next),
                "rate": rate, "lift": lift, "channels": chans}

    results = []
    for obs in OBS_GRID:
        for maxup in MAXUP_GRID:
            results.append(eval_combo(obs, maxup))

    # --- 그리드 표 출력 ---
    print("  OBS  MAXUP      n   양성률    리프트   신뢰?")
    print("  ---  -----  -----  -------  -------  -----")
    for r in results:
        trust = (r["n"] >= MIN_N_TRUST and not _isnan(r["rate"]) and r["rate"] >= MIN_RATE_TRUST)
        rate_s = ("%.4f" % r["rate"]) if not _isnan(r["rate"]) else "  n/a "
        lift_s = ("%.2fx" % r["lift"]) if not _isnan(r["lift"]) else " n/a "
        print("  %3d  %5d  %5d  %7s  %7s  %s"
              % (r["obs"], r["maxup"], r["n"], rate_s, lift_s,
                 "<== 신뢰" if trust else ""))

    # '신뢰 가능 신호' 목록 (n>=15 & rate>=0.35)
    trusted = [r for r in results
               if r["n"] >= MIN_N_TRUST and not _isnan(r["rate"]) and r["rate"] >= MIN_RATE_TRUST]
    print("\n'신뢰 가능 신호' 기준: n>=%d 이면서 양성률>=%.2f" % (MIN_N_TRUST, MIN_RATE_TRUST))
    if trusted:
        print("  신뢰 가능 조합 %d개:" % len(trusted))
        for r in trusted:
            print("    OBS=%d MAXUP=%d : n=%d, 양성률=%.4f, 리프트=%.2fx"
                  % (r["obs"], r["maxup"], r["n"], r["rate"], r["lift"]))
    else:
        print("  [경고] 신뢰 가능 신호 없음 — 표본 15+ & 양성률 0.35+ 조합이 하나도 없습니다.")

    # --- 대조군: 라벨1인데 관측구간 롱폼 많이(>=5편) 올린 채널 (OBS 는 가독성 위해 14일 기준) ---
    print("\n--- 대조군 (라벨1 & 관측구간 롱폼 >=%d편, OBS=14 기준) ---" % CONTROL_MIN_UP)
    ctrl_cnt = cnt_cache.get(14, obs_longform_count(d_prev_ts, 14))
    ctrl_chans = [c for c in prev_pos if int(ctrl_cnt.get(c, 0)) >= CONTROL_MIN_UP]
    ctrl_with_next = [c for c in ctrl_chans if c in next_target.index]
    if len(ctrl_with_next) == 0:
        print("  [표본부족] 조건을 만족하는 채널이 0개 → 대조 양성률 계산 불가.")
    else:
        cvals = next_target.reindex(ctrl_with_next)
        crate = float(cvals.mean())
        clift = (crate / overall_next_rate) if overall_next_rate else float("nan")
        print("  대조군 n=%d, D_next 양성률=%.4f, 리프트=%.2fx (전체 %.4f)"
              % (len(ctrl_with_next), crate, clift, overall_next_rate))

    # --- 사용할 조합 선택: 신뢰 가능 신호 중 리프트 최대, 없으면 n 최대 ---
    if trusted:
        chosen = max(trusted, key=lambda r: (r["lift"], r["n"]))
        chosen_reason = "신뢰 가능 신호 중 리프트 최대"
        signal_ok = True
    else:
        # n 이 가장 큰 조합 (동률이면 양성률 높은 쪽)
        valid = [r for r in results if r["n"] > 0]
        if valid:
            chosen = max(valid, key=lambda r: (r["n"], 0.0 if _isnan(r["rate"]) else r["rate"]))
        else:
            chosen = results[0]
        chosen_reason = "신뢰 가능 신호 없음 → n 이 가장 큰 조합"
        signal_ok = False
    print("\n선택된 조합: OBS=%d, MAXUP=%d  (%s)" % (chosen["obs"], chosen["maxup"], chosen_reason))

    # ============================================================
    # [2] FINAL20 생성 → FINAL20_up.csv
    # ============================================================
    print("\n" + "=" * 64)
    print("[2] FINAL20 상향 제출 생성 (FINAL20_up.csv)")
    print("=" * 64)
    if not signal_ok:
        print("[경고] 신뢰 가능한 신호가 없습니다. FINAL20_up.csv 는 생성하되 제출을 권장하지 않습니다.")

    # --- 테스트셋 적용: 직전 라벨은 D_next(최신 라벨) target==1, 관측구간은 데이터 최신일 기준 ---
    test_prev = nxt[nxt["target"] == 1]["channel_id"].tolist()  # 가장 최근 라벨(08-25) 양성 채널
    max_date = longform["published_at"].max()
    if pd.isna(max_date):
        sys.exit("[backtest_up] video_5d_views.csv 의 published_at 을 날짜로 해석할 수 없습니다.")
    print("데이터 최신일(max_date): %s" % pd.to_datetime(max_date).strftime("%Y-%m-%d"))
    test_cnt = obs_longform_count(pd.to_datetime(max_date), chosen["obs"])
    test_chans = [c for c in test_prev if int(test_cnt.get(c, 0)) <= chosen["maxup"]]
    print("테스트 조건 만족 채널 수(상향 후보): %d" % len(test_chans))

    base = read_csv_clean(base_path)
    if "row_id" not in base.columns or "prediction" not in base.columns:
        sys.exit("[backtest_up] %s 형식 이상(컬럼 %s)" % (base_path, list(base.columns)))
    base["row_id"] = base["row_id"].astype(str)
    base["prediction"] = pd.to_numeric(base["prediction"], errors="coerce")
    n_rows = len(base)
    base_ids = set(base["row_id"].tolist())
    print("기준 파일: %s  (행=%d)" % (base_path, n_rows))

    # 테스트 채널 → row_id("_2026-09-01") 로 변환, 기준 파일에 실제 존재하는 것만 상향 대상
    up_row_ids = []
    for c in test_chans:
        rid = "%s_%s" % (c, TEST_DATE)
        if rid in base_ids:
            up_row_ids.append(rid)
    missing = [c for c in test_chans if ("%s_%s" % (c, TEST_DATE)) not in base_ids]
    if missing:
        print("[주의] 테스트 채널 중 기준 파일에 row_id 가 없는 채널 %d개(상향 제외): 예시 %s"
              % (len(missing), missing[:5]))

    # blend_90 순위(prediction 내림차순, 1위=최고) 사전 계산 — 전/후 비교용
    base_sorted = base.sort_values("prediction", ascending=False).reset_index(drop=True)
    base_rank = {rid: i + 1 for i, rid in enumerate(base_sorted["row_id"].tolist())}

    # 원본 prediction 을 그대로 복사 (재랭킹/로지스틱 재적용 금지)
    out = base.copy()
    orig_pred = base.set_index("row_id")["prediction"]

    # 상향 값 계산: new = max(cur, 0.5*cur + 0.5*0.95), 서로 미세 차등(1e-6 간격).
    # 현재값이 큰 채널일수록 더 위에 오도록, cur 내림차순으로 미세 차등을 더한다.
    up_sorted = sorted(up_row_ids, key=lambda r: float(orig_pred.get(r, 0.0)), reverse=True)
    up_values = {}
    for i, rid in enumerate(up_sorted):
        cur = float(orig_pred.get(rid, 0.0))
        base_up = max(cur, 0.5 * cur + 0.5 * UP_ANCHOR)
        # 미세 차등: 상위부터 조금씩 깎아 서로 구분 (최대 1.0 안 넘도록 클립)
        val = base_up - i * UP_STEP
        if val > 1.0:
            val = 1.0
        if val < 0.0:
            val = 0.0
        up_values[rid] = val

    if up_values:
        mask = out["row_id"].isin(up_values.keys())
        out.loc[mask, "prediction"] = out.loc[mask, "row_id"].map(up_values)

    out_path = os.path.join(os.path.dirname(base_path) or ".", "FINAL20_up.csv")
    out[["row_id", "prediction"]].to_csv(
        out_path, index=False, encoding="utf-8", lineterminator="\n")
    print("저장: %s" % out_path)

    # ------------------------------------------------------------
    # 자체 검증 출력
    # ------------------------------------------------------------
    print("\n--- 자체 검증 ---")
    chk = read_csv_clean(out_path)
    chk["row_id"] = chk["row_id"].astype(str)
    chk["prediction"] = pd.to_numeric(chk["prediction"], errors="coerce")
    new_pred = chk.set_index("row_id")["prediction"]

    # 1) 행 수 (694 기대)
    rows_ok = (len(chk) == n_rows)
    print("행 수: %d  (기준 %d) → %s" % (len(chk), n_rows, "OK" if rows_ok else "FAIL"))

    # 2) 값이 바뀐 행 수 == 올린 채널 수, 나머지는 비트동일
    common = orig_pred.index.intersection(new_pred.index)
    changed = [rid for rid in common if new_pred[rid] != orig_pred[rid]]
    identical = [rid for rid in common if new_pred[rid] == orig_pred[rid]]
    n_up = len(up_values)
    changed_ok = (len(changed) == n_up)
    print("값이 바뀐 행 수: %d  (올린 채널 수 %d) → %s"
          % (len(changed), n_up, "OK" if changed_ok else "FAIL"))
    print("비트단위 동일한 행 수: %d  (기대 %d) → %s"
          % (len(identical), n_rows - n_up,
             "OK" if len(identical) == n_rows - n_up else "FAIL"))
    # 바뀐 행이 정확히 상향 대상 집합인지
    changed_set_ok = (set(changed) == set(up_values.keys()))
    print("바뀐 행이 정확히 상향 대상 집합인지 → %s" % ("OK" if changed_set_ok else "FAIL"))

    # 3) prediction 범위 0~1
    pmin, pmax = float(chk["prediction"].min()), float(chk["prediction"].max())
    range_ok = (pmin >= 0.0) and (pmax <= 1.0)
    print("prediction 범위: min=%.3g max=%.3g → %s" % (pmin, pmax, "OK" if range_ok else "FAIL"))

    # 4) 중복/NaN 없음
    dup = int(chk["row_id"].duplicated().sum())
    nan = int(chk["prediction"].isna().sum())
    clean_ok = (dup == 0 and nan == 0)
    print("row_id 중복=%d, prediction NaN=%d → %s" % (dup, nan, "OK" if clean_ok else "FAIL"))

    # 5) 올린 채널 목록 + before/after + 순위 변화
    print("\n--- 올린 채널 %d개: before/after prediction, blend_90 순위 변화 ---" % n_up)
    if n_up == 0:
        print("  (상향 대상 채널 없음)")
    else:
        # 상향 후 순위 재계산 (출력 비교용일 뿐, 저장 파일은 재랭킹하지 않음)
        new_sorted = chk.sort_values("prediction", ascending=False).reset_index(drop=True)
        new_rank = {rid: i + 1 for i, rid in enumerate(new_sorted["row_id"].tolist())}
        for rid in up_sorted:
            before = float(orig_pred.get(rid, float("nan")))
            after = float(new_pred.get(rid, float("nan")))
            rb = base_rank.get(rid, -1)
            ra = new_rank.get(rid, -1)
            print("  %s : %.6f → %.6f  (순위 %d위 → %d위)" % (rid, before, after, rb, ra))

    all_ok = all([rows_ok, changed_ok, changed_set_ok, range_ok, clean_ok])
    print("\n[종합] %s" % (
        ("모든 검증 통과 — FINAL20_up.csv %s."
         % ("사용 가능(신호 있음)" if signal_ok else "생성됨(신호 약함, 제출 비권장)"))
        if all_ok else "검증 실패 항목이 있습니다. 위 FAIL 을 확인하세요."))
    print("\n[주의] 이 샌드박스에는 pandas/numpy 가 없어 실행 검증을 못 했습니다(문법 검증만 수행).")
    print("        로컬(anaconda, pandas, numpy<2)에서 `python backtest_up.py` 로 실제 실행하세요.")


def _isnan(x):
    """numpy 없이도 안전한 NaN 체크 (x != x 는 NaN 일 때만 참)."""
    try:
        return x != x
    except Exception:
        return False


if __name__ == "__main__":
    main()
