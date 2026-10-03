# -*- coding: utf-8 -*-
"""
구조 시뮬레이션 모델 (structural simulation)
  채널별로 '현재 수준(level)'을 추정한 뒤 미래 업로드를 모의 생성해, 공식 정답 산식 그대로
  growth_score 가 상위 20% 에 들 확률을 계산한다.
  D-7 기준일의 정답(prev label)은 '모의 생성한 미래가 그 정답과 일치하는 정도'로 가중(soft conditioning)한다.

사용 정보: published_at < D 인 영상의 배포 view_5d (코드북 §2.5 허용 범위), D-7 기준일 정답(train_labels 또는 자체 계산)
미래 정보: 사용하지 않음 (미래 영상은 전부 모의 생성)
"""
import numpy as np, pandas as pd
from scipy.stats import norm

P = "data/5th-Datathon/data/epoch_data/"
T = pd.Timestamp
SHIFT0, SHIFT1, SHIFT2 = T("2026-08-25"), T("2026-08-26"), T("2026-08-27")   # 조회수 급등 구간 (실측)
END = T("2026-09-01")


def load_long():
    f5 = pd.read_csv(P + "video_5d_views.csv", parse_dates=["published_at"])
    L = f5[(f5.is_shorts == 0) & f5.view_5d.notna()].copy()
    L["x"] = np.log1p(L.view_5d)
    return L.sort_values("published_at").reset_index(drop=True)


_CS = None


def chan_vb(D):
    """채널 전체 조회수 증가 속도의 급등 전후 log 비 (channel_snapshots, collected_at < D 만 사용). 중앙값을 뺀 값."""
    global _CS
    if _CS is None:
        _CS = pd.read_csv(P + "channel_snapshots.csv", parse_dates=["collected_at"]).sort_values(["channel_id", "collected_at"])
    cs = _CS[_CS.collected_at < T(D)]
    at = lambda t: cs[cs.collected_at <= T(t)].groupby("channel_id").tail(1).set_index("channel_id")
    s0, s1, s2, s3 = at("2026-08-19"), at("2026-08-26"), at("2026-08-27 12:00"), at(D)

    def vel(a, b):
        dt = (b.collected_at - a.collected_at.reindex(b.index)).dt.total_seconds() / 86400
        return ((b.total_view_count - a.total_view_count.reindex(b.index)) / dt).where(dt >= 0.4)
    vb = np.log((vel(s2, s3).clip(lower=0) + 10) / (vel(s0, s1).clip(lower=0) + 10)).dropna().clip(-2, 4)
    return vb - vb.median()


def fit_boost(L, D, use_vb=True):
    """급등 폭 b(게시일 구간, 채널 규모, 채널 조회 속도 변화). D 이전 정보만 사용. D <= 08-26 이면 추정 불가 → None."""
    D = T(D)
    if D <= SHIFT0 + pd.Timedelta(days=1):
        return None
    pre = L[(L.published_at < SHIFT0)].groupby("channel_id").x.agg(["median", "count"])
    pre = pre[pre["count"] >= 3]["median"].rename("mu_pre")
    post = L[(L.published_at >= SHIFT0) & (L.published_at < D)].merge(pre, on="channel_id")
    post["r"] = (post.x - post.mu_pre).clip(-3, 4)
    post["seg"] = np.where(post.published_at < SHIFT1, 0, np.where(post.published_at < SHIFT2, 1, 2))
    seg = {}
    for s, g in post.groupby("seg"):
        if len(g) < 30:
            continue
        z = g.mu_pre - 9.0
        q = pd.qcut(z, 5, labels=False, duplicates="drop")              # 강건 직선: 규모 5분위 중앙값에 직선 적합
        mx = z.groupby(q).median(); my = g.r.groupby(q).median()
        beta, alpha = np.polyfit(mx.values, my.values, 1)
        seg[s] = (alpha, beta)
    gamma = 0.0; vb = pd.Series(dtype=float)
    if use_vb and 2 in seg and D >= SHIFT2 + pd.Timedelta(days=1):
        vb = chan_vb(D)
        g = post[post.seg == 2].copy(); g["vb"] = g.channel_id.map(vb)
        g = g[g.vb.notna()]
        if len(g) >= 100:
            res = g.r - (seg[2][0] + seg[2][1] * (g.mu_pre - 9.0))
            q = pd.qcut(g.vb, 5, labels=False, duplicates="drop")
            gamma = float(np.polyfit(g.vb.groupby(q).median().values, res.groupby(q).median().values, 1)[0])
    return dict(seg=seg, gamma=gamma, vb=vb)


GSCALE = {0: 0.15, 1: 0.5, 2: 1.0}


def boost_of(t, size, B, vbc=0.0):
    """게시 시각 t(배열), 채널 규모 size(급등 전 log 중앙값), vbc(채널 조회 속도 변화) → 급등 폭."""
    if B is None:
        return np.zeros(len(t))
    seg = np.where(t < SHIFT0, -1, np.where(t < SHIFT1, 0, np.where(t < SHIFT2, 1, 2)))
    b = np.zeros(len(t))
    for s, (a, be) in B["seg"].items():
        b[seg == s] = a + be * (size - 9.0) + GSCALE[s] * B["gamma"] * vbc
    return b


def kalman(x, days, sig2, q, tau2_times=None, clip=2.5):
    """강건 local-level 필터. x: 관측(시간순), days: 관측 시점(일), 반환 (level, var, 마지막 시점)."""
    lvl = np.median(x[:3]); var = 1.0; last = days[0]
    for xi, di in zip(x, days):
        var += q * max(di - last, 0.0)
        if tau2_times is not None and last < tau2_times[0] <= di:
            var += tau2_times[1]
        s = var + sig2
        e = np.clip(xi - lvl, -clip * np.sqrt(s), clip * np.sqrt(s))
        k = var / s
        lvl += k * e; var *= (1 - k); last = di
    return lvl, var, last


def simulate(L, D, W, prev=None, W_prev=18, future_boost=True, S=3000, seed=0,
             q=0.0015, n0=6.0, nu=5.0, kappa=0.25, tau_delta=0.30, rate_days=28, od=4.0, lvl_extra=0.0, sig_mult=1.0, bf_mult=1.0, bf_slope_mult=1.0, vb_mult=0.0, bfC_mult=1.0, phi=1.0, resid="t", boot_n0=5.0,
             channels=None, return_draws=False, prev_D=None, use_boost=True):
    """
    D 기준일, W 미래 구간(일). prev: Series(channel_id → 0/1) = D-7 기준일의 정답(구간 W_prev 일).
    반환 DataFrame(channel_id 색인): p(조건부 확률), p0(prev 미사용), g_mean, past_med, lam, level, sig ...
    """
    rng = np.random.default_rng(seed)
    D = T(D); Dp = T(prev_D) if prev_D is not None else D - pd.Timedelta(days=7)
    B = fit_boost(L, D, use_vb=vb_mult > 0) if use_boost else None
    day = pd.Timedelta(days=1); prev_end = Dp + W_prev * day; tgt_end = D + W * day
    past = L[L.published_at < D]
    t0 = T("2026-07-01")
    grp = {c: g for c, g in past.groupby("channel_id")}
    # 채널 규모(급등 전 중앙값; 없으면 전체 중앙값)
    rows = []; draws = {}
    # 전역 잔차 표준편차(규모 무관 기본값)
    for c, g in grp.items():
        if channels is not None and c not in channels:
            continue
        xv = g.view_5d.values
        if len(xv) < 3 or np.median(xv) < 100:
            continue
        t = g.published_at.values.astype("datetime64[s]"); tt = pd.to_datetime(t)
        days = (tt - t0) / pd.Timedelta(days=1); days = np.asarray(days, float)
        pre = g.x.values[tt < SHIFT0]
        size = np.median(pre) if len(pre) >= 3 else np.median(g.x.values) - (0.5 if D > SHIFT2 else 0.0)
        vbc = float(B["vb"].get(c, 0.0)) * vb_mult if B is not None else 0.0
        xb = g.x.values - boost_of(tt, size, B, vbc)                       # 급등분을 뺀 값(같은 척도)
        # 잔차 표준편차: 채널 중앙값 기준 MAD, 전역값으로 수축
        mad = np.median(np.abs(xb - np.median(xb))) * 1.4826
        sig_g = 0.95 - 0.045 * (np.clip(size, 4, 13) - 4)               # 규모별 기본 잔차(실측: 소형 0.8, 대형 0.56 수준)
        n = len(xb)
        sig = np.sqrt((n * mad ** 2 + n0 * sig_g ** 2) / (n + n0)) * sig_mult
        shift_day = (SHIFT2 - t0) / pd.Timedelta(days=1)
        lvl, var, last = kalman(xb, days, sig ** 2, q, tau2_times=(shift_day, tau_delta ** 2) if (B is not None) else None)
        dnow = (D - t0) / pd.Timedelta(days=1)
        var += q * max(dnow - last, 0)
        if B is None and D > SHIFT2: pass
        # 업로드율(유효 롱폼/일): 최근 rate_days 일, 관측 가능한 기간으로 나눔
        first = days[0]; span = min(rate_days, max(dnow - first, 3.0))
        n_rec = (days >= dnow - span).sum()
        lam = (n_rec + 0.3) / (span + 2.0)
        past_med = np.median(xv)
        # 미래 급등 폭 (채널 규모별, 지속 가정). 검증 기준일(D<=08-25)은 0
        bf = (bf_mult * B["seg"][2][0] + bf_slope_mult * B["seg"][2][1] * (size - 9.0) + B["gamma"] * vbc) if (B is not None and future_boost and 2 in B["seg"]) else 0.0
        # ---- 모의 생성 ----
        dB = max((min(prev_end, tgt_end) - D) / day, 0.0); dC = abs((prev_end - tgt_end) / day)     # dB: 두 구간이 겹치는 부분, dC: 한쪽에만 속하는 부분
        prev_longer = prev_end > tgt_end
        lam_s = lam * rng.gamma(od, 1.0 / od, S)
        nB = rng.poisson(lam_s * dB); nC = rng.poisson(lam_s * dC)
        nmax = int(max((nB + nC).max(), 1))
        lvl_f = lvl - (1.0 - phi) * (lvl - np.median(xb))                 # phi<1: 현재 수준이 장기 중앙값 쪽으로 부분 회귀한다는 가정 (기본 1.0 = 현재 수준 유지)
        level = lvl_f + np.sqrt(var + q * W / 2 + lvl_extra ** 2) * rng.standard_normal(S)
        eps = rng.standard_t(nu, (S, nmax)) * sig / np.sqrt(nu / (nu - 2))
        if resid == "boot":                                                # 잔차를 t 분포 대신 채널 자신의 과거 잔차에서 복원추출(쌍봉·치우침 반영). 영상이 적으면 t 분포와 섞음
            e = xb - np.median(xb)
            eb = e[rng.integers(0, len(e), (S, nmax))]
            eps = np.where(rng.random((S, nmax)) < len(e) / (len(e) + boot_n0), eb, eps)
        idx = np.arange(nmax)[None, :]
        mB = idx < nB[:, None]; mBC = idx < (nB + nC)[:, None]
        bfm = bf if bfC_mult == 1.0 else bf * np.where(mB, 1.0, bfC_mult)   # bfC_mult<1: 지난주 정답 구간 이후(뒤쪽 구간)에는 급등 폭이 줄어든다는 가정
        X = np.expm1(np.clip(level[:, None] + bfm + eps, 0, 20))
        XBC = np.where(mB if prev_longer else mBC, X, np.nan)
        with np.errstate(all="ignore"):
            fut = np.nanmedian(XBC, axis=1)
        fut = np.where((nB if prev_longer else (nB + nC)) == 0, 0.0, fut)
        g_test = np.log1p(fut) - np.log1p(past_med)
        # prev 구간: A(실측, D-7~D) ∪ B(모의)
        gp = None; has_prev = prev is not None and c in prev.index and not pd.isna(prev.get(c))
        pp = g[g.published_at < Dp]
        if len(pp) >= 1:
            A = g[g.published_at >= Dp].view_5d.values
            XA = np.broadcast_to(A[None, :], (S, len(A))) if len(A) else np.empty((S, 0))
            XB = np.where(mBC if prev_longer else mB, X, np.nan)
            both = np.concatenate([XA, XB], axis=1)
            with np.errstate(all="ignore"):
                fp = np.nanmedian(both, axis=1) if both.shape[1] else np.zeros(S)
            fp = np.where(np.isnan(fp), 0.0, fp)
            gp = np.log1p(fp) - np.log1p(np.median(pp.view_5d.values))
        rows.append(dict(channel_id=c, level=lvl, lvar=var, sig=sig, lam=lam, size=size, bf=bf, past_med=np.log1p(past_med), n_past=n,
                         has_prev=bool(has_prev), prev=(float(prev.get(c)) if has_prev else np.nan)))
        draws[c] = (g_test, gp)
    R = pd.DataFrame(rows).set_index("channel_id")
    # prev 임계값: prev 정답이 있는 채널들의 모의 성장 점수를 모아 상위 20% 선
    pool = np.concatenate([draws[c][1] for c in R.index if R.at[c, "has_prev"] and draws[c][1] is not None]) if R.has_prev.any() else None
    thr_prev = np.quantile(pool, 0.8) if pool is not None else np.nan
    Wt = {}
    for c in R.index:
        g_test, gp = draws[c]
        if R.at[c, "has_prev"] and gp is not None:
            pr = norm.cdf((gp - thr_prev) / kappa)
            w = pr if R.at[c, "prev"] == 1 else 1 - pr
            w = w + 1e-6
        else:
            w = np.ones(len(g_test))
        Wt[c] = w / w.sum()
    allg = np.concatenate([draws[c][0] for c in R.index]); allw = np.concatenate([Wt[c] for c in R.index])
    o = np.argsort(allg); cw = np.cumsum(allw[o]) / allw.sum()
    thr = allg[o][np.searchsorted(cw, 0.8)]
    thr0 = np.quantile(allg, 0.8)
    R["thr"] = thr; R["thr_prev"] = thr_prev
    R["p"] = [float((Wt[c] * (draws[c][0] >= thr)).sum()) for c in R.index]
    R["p0"] = [float((draws[c][0] >= thr0).mean()) for c in R.index]
    R["g_mean"] = [float((Wt[c] * np.clip(draws[c][0], -3, 5)).sum()) for c in R.index]
    R["g_med0"] = [float(np.median(draws[c][0])) for c in R.index]
    R["p_prev_sim"] = [float((draws[c][1] >= thr_prev).mean()) if draws[c][1] is not None and not np.isnan(thr_prev) else np.nan for c in R.index]
    if return_draws:
        return R, draws, Wt
    return R


def self_label(L, D, W, end=END):
    D = T(D); hi = min(D + pd.Timedelta(days=W), end)
    p = L[L.published_at < D].groupby("channel_id").view_5d.agg(["median", "count"])
    f = L[(L.published_at >= D) & (L.published_at < hi)].groupby("channel_id").view_5d.median()
    a = p.join(f.rename("fut"), how="left"); a = a[(a["count"] >= 3) & (a["median"] >= 100)].copy()
    a["g"] = np.log1p(a.fut.fillna(0)) - np.log1p(a["median"])
    a["y"] = (a.g >= a.g.quantile(0.8)).astype(int)
    return a
