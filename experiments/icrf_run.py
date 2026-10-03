"""
==========================================================================================
 ICRF (Interval-Censored Recursive Forest) — 구간중도절단 생존 랜덤 포레스트로 Rising 확률 예측
==========================================================================================
[아이디어]  (Cho, Jewell & Kosorok 의 icrf R 패키지 사용)
  점수 s = log(1+미래 중앙값) - log(1+과거 중앙값) 를 '생존시간'처럼 보고 T = s + 15 (양수로 이동).
  관측 형태는 생존분석의 중도절단과 똑같다:
    - 미래 창이 다 보임         → T 정확 관측                (L ≈ R)
    - 미래 영상 0편             → T ≤ 14 (좌측중도절단)       (L = 0, R = 14)
    - 미래 창이 잘림(데이터 끝) → T ∈ [L, R] (구간중도절단)   (순서통계량 구간, R 이 ∞ 면 우측중도절단)
  ICRF 는 비모수(분포 가정 없음) 생존 포레스트라 변수 간 상호작용·비선형을 자동 반영.
  예측: 채널별 생존함수 S(t | x) = P(T > t).  Rising 확률 = S(τ),  τ 는 '평균 = 0.2' 가 되도록.

[실행]  run_final.py 와 같은 폴더. R 과 icrf 패키지 필요 (R CMD INSTALL icrf)
  python icrf_run.py --data ./epoch_data --F 18 --validate      # 검증 (PR/ROC, 로지스틱과 비교)
  python icrf_run.py --data ./epoch_data --F 28 --out icrf_submission.csv
==========================================================================================
"""
import sys, runpy, argparse, subprocess, os, tempfile, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='./epoch_data')
ap.add_argument('--F', type=int, default=18)
ap.add_argument('--validate', action='store_true')
ap.add_argument('--days', type=int, default=9, help='검증일 수')
ap.add_argument('--ntree', type=int, default=150)
ap.add_argument('--nmax', type=int, default=2500, help='학습 행 상한 (최근일 가중 확률로 표본추출)')
ap.add_argument('--out', default='icrf_submission.csv')
ap.add_argument('--blend_out', default='icrf_blend_submission.csv')
a = ap.parse_args()

sys.argv = ['run_final.py', '--data', a.data, '--F', str(a.F), '--mode', 'latest', '--build_only']
R = runpy.run_path('run_final.py')
X, design, intervals, partial_s, label = R['X'], R['design'], R['intervals'], R['partial_s'], R['label']
F, DAY, START, TEST_D, ss = R['F'], R['DAY'], R['START'], R['TEST_D'], R['ss']
fit_logit, train_set = R['fit_predict'], R['train_set']
from sklearn.metrics import roc_auc_score, average_precision_score
SHIFT = 15.0

def build_ic(C):
    """기준일마다 T 의 구간 [L, R] (dgp_ic.py 와 동일한 규칙)"""
    rows = []
    for D in pd.date_range(START, C - DAY):
        r = X[X.D == D].copy()
        if D + F * DAY <= C:
            o = partial_s(D, D + F * DAY); ex = o.fm.notna()
            I = pd.DataFrame({'L': np.where(ex, o.s, -np.inf), 'R': np.where(ex, o.s, -1.0)}, index=o.index)
        else:
            if (C - D).days < 3: continue
            I = intervals(D, C)
        r = r[r.channel_id.isin(I.index)]
        r['L'] = r.channel_id.map(I.L).values; r['R'] = r.channel_id.map(I.R).values
        rows.append(r)
    tr = pd.concat(rows, ignore_index=True)
    tr['w'] = 0.5 ** ((tr.D.max() - tr.D).dt.days.values / 7)
    return tr

R_SCRIPT = r'''
suppressMessages(library(icrf))
a <- commandArgs(TRUE); d <- a[1]; ntree <- as.integer(a[2])
tr <- read.csv(file.path(d, "tr.csv")); te <- read.csv(file.path(d, "te.csv"))
L <- tr$L; R <- tr$R; R[is.na(R)] <- Inf
x <- tr[, !(names(tr) %in% c("L", "R"))]
set.seed(1)
fit <- icrf(x = x, L = L, R = R, ntree = ntree, nodesize = 20L, nfold = 1L, returnBest = FALSE,
            split.rule = "Wilcoxon", tau = 40)
p <- predict(fit, newdata = te[, names(x)])
tm <- attr(p, "time")
write.csv(data.frame(t = tm), file.path(d, "time.csv"), row.names = FALSE)
write.csv(as.data.frame(p), file.path(d, "surv.csv"), row.names = FALSE)
'''

def run_icrf(C, te):
    tr = build_ic(C)
    if len(tr) > a.nmax:                                       # 최근일 가중 표본추출
        rng = np.random.default_rng(0)
        tr = tr.iloc[rng.choice(len(tr), a.nmax, replace=False, p=tr.w / tr.w.sum())]
    A_, B_ = design(tr, te)
    Lb = tr.L.values + SHIFT; Rb = tr.R.values + SHIFT
    Lb = np.where(np.isneginf(Lb), 0.0, Lb)
    ex = np.isclose(Lb, Rb)                                    # 정확 관측은 아주 좁은 구간으로
    Lb = np.where(ex, Lb - 0.01, Lb); Rb = np.where(ex, Rb + 0.01, Rb)
    Lb = np.clip(Lb, 0, None)
    # 시간 격자를 0.25 단위로 거칠게 (고유 시점 수가 많으면 NPMLE 메모리가 폭증)
    Lb = np.floor(Lb / 0.25) * 0.25
    Rb = np.where(np.isinf(Rb), Rb, np.ceil(Rb / 0.25) * 0.25)
    Rb = np.where(Rb <= Lb, Lb + 0.25, Rb)
    d = tempfile.mkdtemp(dir=os.environ.get('TMPDIR', None))
    A_.assign(L=Lb, R=np.where(np.isinf(Rb), np.nan, Rb)).to_csv(f'{d}/tr.csv', index=False)
    B_.to_csv(f'{d}/te.csv', index=False)
    open(f'{d}/run.R', 'w').write(R_SCRIPT)
    subprocess.run(['Rscript', f'{d}/run.R', d, str(a.ntree)], check=True, capture_output=True)
    S = pd.read_csv(f'{d}/surv.csv').values                   # n × 시간 : 생존확률 P(T > t)
    t = pd.read_csv(f'{d}/time.csv').t.values
    mean_S = S.mean(0)
    j = np.argmin(np.abs(mean_S - 0.2))                         # 평균 생존확률 = 0.2 인 시점 τ
    return S[:, j]

rk = lambda x: pd.Series(x).rank(pct=True).values
if a.validate:
    res = []
    last_full = TEST_D - F * DAY
    for vd in pd.date_range(last_full - (a.days - 1) * DAY, last_full):
        va = X[X.D == vd].copy(); va['y'] = va.channel_id.map(label(vd, F)); va = va[va.y.notna()]
        t0 = time.time()
        p_ic = run_icrf(vd, va)
        p_lg, _ = fit_logit(train_set(vd), va)
        bl = 0.5 * rk(p_ic) + 0.5 * rk(p_lg)
        r = [average_precision_score(va.y, p_ic), average_precision_score(va.y, p_lg), average_precision_score(va.y, bl),
             roc_auc_score(va.y, p_ic), roc_auc_score(va.y, p_lg), roc_auc_score(va.y, bl)]
        res.append(r)
        print(vd.date(), 'PR  ICRF %.4f | 로지스틱 %.4f | 순위평균 %.4f   ROC %.4f | %.4f | %.4f' % tuple(r), f'({time.time()-t0:.0f}s)', flush=True)
    print('평균 PR  ICRF %.4f | 로지스틱 %.4f | 순위평균 %.4f   ROC %.4f | %.4f | %.4f' % tuple(np.mean(res, 0)))
else:
    te = X[X.D == TEST_D]
    p_ic = run_icrf(TEST_D, te)
    p_lg, _ = fit_logit(train_set(TEST_D), te)
    key = te.channel_id + '_' + TEST_D.strftime('%Y-%m-%d')
    for out, p in [(a.out, p_ic), (a.blend_out, 0.5 * rk(p_ic) + 0.5 * rk(p_lg))]:
        s2 = ss.copy(); s2['prediction'] = s2.row_id.map(dict(zip(key, p)))
        assert s2.prediction.notna().all(); s2.to_csv(out, index=False); print('저장:', out)
