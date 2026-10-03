# -*- coding: utf-8 -*-
"""
backtest_drop.py — (1) 제출 규칙 백테스트 + (2) 올바른 FINAL19 재생성

로컬 datathon 폴더(epoch_data/ 와 blend_90.csv 가 있는 곳)에서 실행한다.
  python backtest_drop.py
  python backtest_drop.py --data ./epoch_data --base blend_90.csv --obs-window 14

[1] 규칙 백테스트 (가장 중요)
  "1장 남은 제출"의 근거가 된 두 가설이 과거 라벨 데이터에서 실제로 성립하는지 수치로 검증한다.
  라벨 날짜 D(=2026-08-11)를 "직전 라벨", 그 다음 라벨 날짜 D_next(=2026-08-25)를 "다음 라벨"로 둔다.
    (A) "라벨1+업로드0" 그룹: D의 target==1 이고, D 직전 관측구간(OBS_WINDOW 일)에 롱폼 업로드 0편.
        → 가설: 다음 라벨에서 양성이 많다.
    (B) "초반강세+라벨0" 그룹: D의 target==0 이고, D 직전 관측구간에 롱폼 2편 이상,
        그 영상들의 view_5d 로그값이 그 채널 과거 평균 롱폼 로그뷰보다 1.5 이상 큼.
        → 가설: 다음 라벨에서 양성이 거의 없다.
  각 그룹의 D_next 양성률을 전체 평균 양성률과 비교해 출력한다. 그룹 분모가 0이면 "표본부족"으로 안전 출력.

[2] 올바른 FINAL19 재생성 (FINAL19_clean.csv)
  blend_90.csv 의 prediction 을 "그대로" 복사하고, 지정된 9개 row_id 의 prediction 만
  그 파일 최솟값보다 더 작은 서로 다른 미세값(1e-6 간격)으로 덮어쓴다. 재랭킹/로지스틱 재적용은
  절대 하지 않는다(그게 569개 순위 변동의 원인). → 9개를 제외한 685개는 blend_90 과 비트단위 동일.

보안: 데이터/제출 CSV 는 저장소에 올리지 않는다. 이 스크립트(.py)만 커밋 대상이며,
      FINAL19_clean.csv 는 사용자가 로컬 실행 시 생성한다.
"""
import argparse
import os
import sys


# --- 백테스트 상수 (바꾸기 쉽게 상단에 둠) ---------------------------------
LABEL_DATE_PREV = "2026-08-11"   # 직전 라벨 날짜 D (train_labels 에 실제 존재하는 날짜를 코드로 재확인)
LABEL_DATE_NEXT = "2026-08-25"   # 다음 라벨 날짜 D_next
STRONG_LOGVIEW_DELTA = 1.5        # (B) 초반강세 기준: 로그뷰가 과거 평균보다 이만큼 이상 크면 "강세"
MIN_LONGFORM_FOR_B = 2            # (B) 관측구간 롱폼 최소 편수

# --- [2] 바닥으로 내릴 9개 row_id (제출/blend_90 의 날짜는 2026-09-01) ---------
DROP_ROW_IDS = [
    "UCJA6MZLzTeljmGNFaugdoWQ_2026-09-01",
    "UCgWOZQvFZhGuqBCtjmee0XQ_2026-09-01",
    "UCcvKf8NxAw9i09hZw1MQPBQ_2026-09-01",
    "UCj5Y6sl23ln9PL6OiDTL7Gg_2026-09-01",
    "UCMF3M48BakI-ptofqoUqdAA_2026-09-01",
    "UCHoo90P0hvh2rrbGvX53G0A_2026-09-01",
    "UCaN3tSrrzIrXSDonEe7LK9A_2026-09-01",
    "UCi5yzN6jpdz2akGTm-mJSKQ_2026-09-01",
    "UCnhGcri8x6xhehanhAe9ipw_2026-09-01",
]


def channel_of(row_id):
    """row_id("{channel_id}_{날짜}")에서 channel_id 추출.
    channel_id 자체엔 밑줄이 없고(UC 로 시작 24자) 날짜 앞에만 '_'가 있으므로
    rsplit('_', 1) 로 마지막 '_...' 만 떼어낸다."""
    return str(row_id).rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser(description="제출 규칙 백테스트 + 올바른 FINAL19 재생성")
    ap.add_argument("--data", default="./epoch_data", help="데이터 폴더 (기본 ./epoch_data)")
    ap.add_argument("--base", default="blend_90.csv", help="기준 제출 파일 (기본 blend_90.csv)")
    ap.add_argument("--obs-window", type=int, default=14, help="관측구간 길이(일) (기본 14)")
    args = ap.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        sys.exit("[backtest_drop] 필수 패키지(pandas/numpy)가 없습니다: %s\n"
                 "              로컬에서 `pip install 'numpy<2' pandas` 후 실행하세요." % e)

    data_dir = os.path.abspath(args.data)
    labels_path = os.path.join(data_dir, "train_labels.csv")
    views_path = os.path.join(data_dir, "video_5d_views.csv")
    snap_path = os.path.join(data_dir, "channel_snapshots.csv")
    base_path = os.path.abspath(args.base)

    # 필수 파일 점검 (명확한 에러로 종료)
    for p, lbl in [(labels_path, "train_labels.csv"),
                   (views_path, "video_5d_views.csv"),
                   (base_path, "기준 제출 파일(--base)")]:
        if not os.path.isfile(p):
            sys.exit("[backtest_drop] 필수 파일을 찾을 수 없습니다: %s (%s)" % (p, lbl))

    def read_csv_clean(path):
        """CSV 로드 후 컬럼명 strip/BOM 제거 (기존 blend_* 스크립트 공통 스타일)."""
        df = pd.read_csv(path)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        return df

    obs_window = int(args.obs_window)

    # ============================================================
    # [1] 규칙 백테스트
    # ============================================================
    print("=" * 64)
    print("[1] 규칙 백테스트 (관측구간 OBS_WINDOW=%d일)" % obs_window)
    print("=" * 64)

    labels = read_csv_clean(labels_path)
    if "row_id" not in labels.columns or "target" not in labels.columns:
        sys.exit("[backtest_drop] train_labels.csv 형식 이상(컬럼 %s)" % list(labels.columns))
    labels["channel_id"] = labels["row_id"].map(channel_of)
    labels["label_date"] = labels["row_id"].map(lambda r: str(r).rsplit("_", 1)[-1])
    labels["target"] = pd.to_numeric(labels["target"], errors="coerce")

    # train_labels 에 실제 존재하는 날짜 분포 확인
    date_counts = labels["label_date"].value_counts().sort_index()
    print("\ntrain_labels 날짜 분포:")
    for d, c in date_counts.items():
        print("   %s : %d행" % (d, int(c)))

    d_prev, d_next = LABEL_DATE_PREV, LABEL_DATE_NEXT
    if d_prev not in date_counts.index or d_next not in date_counts.index:
        # 상수가 데이터와 안 맞으면 실제 존재하는 두 날짜(가장 이른 것=D, 그 다음=D_next)로 대체
        avail = list(date_counts.index)
        if len(avail) < 2:
            sys.exit("[backtest_drop] 라벨 날짜가 2개 미만이라 D→D_next 백테스트 불가: %s" % avail)
        d_prev, d_next = avail[0], avail[1]
        print("\n[주의] 상수 날짜가 데이터에 없어 실제 날짜로 대체: D=%s, D_next=%s" % (d_prev, d_next))
    print("\n직전 라벨 D=%s, 다음 라벨 D_next=%s" % (d_prev, d_next))

    prev = labels[labels["label_date"] == d_prev].copy()
    nxt = labels[labels["label_date"] == d_next].copy()
    # D_next target 을 channel_id 로 조회할 수 있게 매핑
    next_target = nxt.set_index("channel_id")["target"]

    overall_next_rate = float(nxt["target"].mean()) if len(nxt) else float("nan")
    print("전체 D_next 평균 양성률: %.4f (%d채널)" % (overall_next_rate, len(nxt)))

    # --- 영상 데이터: 롱폼(is_shorts==0)만, 날짜 파싱 ---
    views = read_csv_clean(views_path)
    need_cols = {"channel_id", "published_at", "view_5d", "is_shorts"}
    if not need_cols.issubset(set(views.columns)):
        sys.exit("[backtest_drop] video_5d_views.csv 형식 이상(컬럼 %s)" % list(views.columns))
    views["published_at"] = pd.to_datetime(views["published_at"], errors="coerce")
    views["view_5d"] = pd.to_numeric(views["view_5d"], errors="coerce")
    # is_shorts 는 0/1 (혹은 True/False) 모두 안전 처리 → 롱폼 = 거짓/0
    is_short = views["is_shorts"].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes"])
    longform = views[~is_short].copy()
    longform["log_view"] = np.log1p(longform["view_5d"].clip(lower=0))

    d_prev_ts = pd.to_datetime(d_prev)
    obs_start = d_prev_ts - pd.Timedelta(days=obs_window)
    # D 직전 관측구간 [D-obs_window, D) 의 롱폼
    in_obs = longform[(longform["published_at"] >= obs_start)
                      & (longform["published_at"] < d_prev_ts)].copy()
    # 각 채널의 "과거"(관측구간 시작 이전) 평균 롱폼 로그뷰 — (B)의 비교 기준
    past = longform[longform["published_at"] < obs_start].copy()
    past_mean_logview = past.groupby("channel_id")["log_view"].mean()

    # 채널별 관측구간 롱폼 편수 / 평균 로그뷰
    obs_longform_count = in_obs.groupby("channel_id").size()
    obs_mean_logview = in_obs.groupby("channel_id")["log_view"].mean()

    def group_report(name, hypothesis, channel_ids):
        """그룹 채널들의 D_next 양성률 vs 전체 양성률 비교 출력. 분모 0이면 표본부족."""
        channel_ids = list(channel_ids)
        # D_next 라벨이 존재하는 채널만 양성률 분모에 포함
        with_next = [c for c in channel_ids if c in next_target.index]
        print("\n--- 그룹 (%s) ---" % name)
        print("  가설: %s" % hypothesis)
        print("  그룹 채널 수(D 기준): %d, 그중 D_next 라벨 존재: %d" % (len(channel_ids), len(with_next)))
        if len(with_next) == 0:
            print("  [표본부족] D_next 라벨이 있는 그룹 채널이 0개 → 양성률 계산 불가.")
        else:
            vals = next_target.reindex(with_next)
            grp_rate = float(vals.mean())
            n_pos = int(vals.sum())
            print("  그룹 D_next 양성률: %.4f (%d/%d 양성)" % (grp_rate, n_pos, len(with_next)))
            print("  전체 D_next 양성률: %.4f" % overall_next_rate)
            diff = grp_rate - overall_next_rate
            print("  차이(그룹-전체): %+.4f  →  %s" % (
                diff, "가설 방향과 일치" if _hyp_match(name, diff) else "가설 방향과 불일치"))
        sample = channel_ids[:10]
        print("  그룹 채널 예시(최대 10개): %s" % (sample if sample else "(없음)"))

    def _hyp_match(name, diff):
        # (A)는 전체보다 높아야 가설 일치, (B)는 전체보다 낮아야 가설 일치
        if name.startswith("A"):
            return diff > 0
        return diff < 0

    # (A) 라벨1 + 관측구간 롱폼 업로드 0편
    prev_pos = prev[prev["target"] == 1]["channel_id"].tolist()
    group_a = [c for c in prev_pos if int(obs_longform_count.get(c, 0)) == 0]
    group_report(
        "A: 라벨1+업로드0",
        "직전 라벨 1인데 관측구간 롱폼 업로드 0편 → 다음 라벨 양성 많다(높아야 일치)",
        group_a,
    )

    # (B) 라벨0 + 관측구간 롱폼 2편 이상 + 초반강세(관측 평균 로그뷰 - 과거 평균 로그뷰 >= 1.5)
    prev_neg = prev[prev["target"] == 0]["channel_id"].tolist()
    group_b = []
    for c in prev_neg:
        cnt = int(obs_longform_count.get(c, 0))
        if cnt < MIN_LONGFORM_FOR_B:
            continue
        if c not in obs_mean_logview.index or c not in past_mean_logview.index:
            continue  # 과거 비교 기준이 없으면 "강세" 판정 불가 → 제외
        delta = float(obs_mean_logview.loc[c]) - float(past_mean_logview.loc[c])
        if delta >= STRONG_LOGVIEW_DELTA:
            group_b.append(c)
    group_report(
        "B: 초반강세+라벨0",
        "롱폼 2편↑ & 로그뷰가 과거평균보다 %.1f↑ & 직전 라벨 0 → 다음 라벨 양성 거의 없다(낮아야 일치)"
        % STRONG_LOGVIEW_DELTA,
        group_b,
    )

    if os.path.isfile(snap_path):
        print("\n(참고) channel_snapshots.csv 존재 — 구독자수 등 참고용 데이터 사용 가능.")

    # ============================================================
    # [2] 올바른 FINAL19 재생성 → FINAL19_clean.csv
    # ============================================================
    print("\n" + "=" * 64)
    print("[2] 올바른 FINAL19 재생성 (FINAL19_clean.csv)")
    print("=" * 64)

    base = read_csv_clean(base_path)
    if "row_id" not in base.columns or "prediction" not in base.columns:
        sys.exit("[backtest_drop] %s 형식 이상(컬럼 %s)" % (base_path, list(base.columns)))
    base["row_id"] = base["row_id"].astype(str)
    base["prediction"] = pd.to_numeric(base["prediction"], errors="coerce")

    n_rows = len(base)
    base_min = float(base["prediction"].min())
    print("기준 파일: %s  (행=%d, 최솟값=%.8g)" % (base_path, n_rows, base_min))

    # 지정 9개 row_id 가 기준 파일에 전부 있는지 확인
    base_ids = set(base["row_id"].tolist())
    missing = [r for r in DROP_ROW_IDS if r not in base_ids]
    if missing:
        sys.exit("[backtest_drop] 바닥으로 내릴 row_id 중 기준 파일에 없는 것이 있습니다:\n   %s"
                 % "\n   ".join(missing))

    # 원본 prediction 을 그대로 복사 (재랭킹/로지스틱 재적용 금지)
    out = base.copy()
    orig_pred = base.set_index("row_id")["prediction"]

    # 9개만 최솟값보다 더 작은 서로 다른 미세값으로 덮어쓴다.
    # base_min(~0.000865)보다 확실히 작은 1e-7 ~ 9e-7 범위, 1e-6 미만 간격으로 차등.
    # (서로 순위가 구분되도록 1e-7 간격; 모두 base_min 보다 작아 반드시 최하위 9개가 됨)
    drop_values = {}
    for i, rid in enumerate(DROP_ROW_IDS):
        drop_values[rid] = round((i + 1) * 1e-7, 10)  # 1e-7, 2e-7, ..., 9e-7
    # 안전장치: 혹시 base_min 이 이미 매우 작아도 9개가 더 작도록 보장
    max_drop = max(drop_values.values())
    if max_drop >= base_min:
        # base_min 바로 아래로 스케일 재조정 (여전히 서로 차등)
        scale = (base_min * 0.1) / max_drop
        drop_values = {r: v * scale for r, v in drop_values.items()}
        print("[주의] 기준 최솟값이 매우 작아 drop 값을 재조정했습니다(최대 %.3g)." % max(drop_values.values()))

    mask = out["row_id"].isin(DROP_ROW_IDS)
    out.loc[mask, "prediction"] = out.loc[mask, "row_id"].map(drop_values)

    out_path = os.path.join(os.path.dirname(base_path) or ".", "FINAL19_clean.csv")
    out[["row_id", "prediction"]].to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    print("저장: %s" % out_path)

    # ------------------------------------------------------------
    # 자체 검증 출력
    # ------------------------------------------------------------
    print("\n--- 자체 검증 ---")
    chk = read_csv_clean(out_path)
    chk["row_id"] = chk["row_id"].astype(str)
    chk["prediction"] = pd.to_numeric(chk["prediction"], errors="coerce")
    new_pred = chk.set_index("row_id")["prediction"]

    # 1) 행 수
    rows_ok = (len(chk) == n_rows)
    print("행 수: %d  (기준 %d) → %s" % (len(chk), n_rows, "OK" if rows_ok else "FAIL"))

    # 2) 값이 바뀐 행 수 == 9, 나머지 685개는 비트단위 동일
    common = orig_pred.index.intersection(new_pred.index)
    changed = [rid for rid in common if new_pred[rid] != orig_pred[rid]]
    identical = [rid for rid in common if new_pred[rid] == orig_pred[rid]]
    changed_ok = (len(changed) == len(DROP_ROW_IDS))
    print("값이 바뀐 행 수: %d  (기대 %d) → %s"
          % (len(changed), len(DROP_ROW_IDS), "OK" if changed_ok else "FAIL"))
    print("비트단위 동일한 행 수: %d  (기대 %d) → %s"
          % (len(identical), n_rows - len(DROP_ROW_IDS),
             "OK" if len(identical) == n_rows - len(DROP_ROW_IDS) else "FAIL"))
    # 바뀐 행이 정확히 지정한 9개인지
    changed_set_ok = (set(changed) == set(DROP_ROW_IDS))
    print("바뀐 행이 정확히 지정 9개인지 → %s" % ("OK" if changed_set_ok else "FAIL"))

    # 3) 9개가 전부 최하위 9개인지
    bottom9 = set(chk.sort_values("prediction").head(len(DROP_ROW_IDS))["row_id"].tolist())
    bottom_ok = (bottom9 == set(DROP_ROW_IDS))
    print("9개가 전부 최하위 9개인지 → %s" % ("OK" if bottom_ok else "FAIL"))

    # 4) prediction 범위 0~1
    pmin, pmax = float(chk["prediction"].min()), float(chk["prediction"].max())
    range_ok = (pmin >= 0.0) and (pmax <= 1.0)
    print("prediction 범위: min=%.3g max=%.3g → %s" % (pmin, pmax, "OK" if range_ok else "FAIL"))

    # 5) 중복/NaN 없음
    dup = int(chk["row_id"].duplicated().sum())
    nan = int(chk["prediction"].isna().sum())
    clean_ok = (dup == 0 and nan == 0)
    print("row_id 중복=%d, prediction NaN=%d → %s" % (dup, nan, "OK" if clean_ok else "FAIL"))

    all_ok = all([rows_ok, changed_ok, changed_set_ok, bottom_ok, range_ok, clean_ok])
    print("\n[종합] %s" % ("모든 검증 통과 — FINAL19_clean.csv 사용 가능." if all_ok
                           else "검증 실패 항목이 있습니다. 위 FAIL 을 확인하세요."))
    print("\n[주의] 이 샌드박스에는 pandas/numpy 가 없어 실행 검증을 못 했습니다(문법 검증만 수행).")
    print("        로컬(anaconda, pandas, numpy<2)에서 `python backtest_drop.py` 로 실제 실행하세요.")


if __name__ == "__main__":
    main()
