"""
==========================================================================================
 Random Survival Forest (RSF) + 로지스틱(latest) 결합 — 28일 창용
==========================================================================================
[생존분석 정식화]
  '생존시간' T = s + 15  (s = log(1+미래 중앙값) - log(1+과거 중앙값), 양수가 되도록 이동)
  - 미래 창이 다 보임          → 사건 관측(event=1), T = s+15
  - 미래 영상 0편              → 아주 낮은 값으로 사건 관측(T = 13)
  - 미래 창이 잘림, R = ∞      → 우측중도절단(event=0), T = L+15   ← RSF 가 원래 다루는 형태
  - 미래 창이 잘림, R 유한     → 구간 중앙값으로 사건 관측(근사)
  RSF: 배깅된 생존나무, log-rank 분할 → 비선형·상호작용 자동 반영, 분포 가정 없음
  Rising 확률 = S(τ | x) = P(T > τ),  τ 는 '평균 = 0.2' 가 되도록
  최종 = RSF 와 로지스틱(latest)의 순위 평균  (PR-AUC 는 순위만 보므로 순위로 결합)

[실행]  python rsf_run.py --data ./epoch_data --F 18 --validate --days 3
        python rsf_run.py --data ./epoch_data --F 28
==========================================================================================
"""
import sys, runpy, argparse, time
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('--data', default='./epoch_data'); ap.add_argument('--F', type=int, default=28)
ap.add_argument('--validate', action='store_true'); ap.add_argument('--days', type=int, default=3)
ap.add_argument('--nmax', type=int, default=8000); ap.add_argument('--w_rsf', type=float, default=0.5)
a = ap.parse_args()
sys.argv = ['run_final.py', '--data', a.data, '--F', str(a.F), '--mode', 'latest', '--build_only']
R = runpy.run_path('run_final.py')
X, design, intervals, partial_s, label = R['X'], R['design'], R['intervals'], R['partial_s'], R['label']
F, DAY, START, TEST_D, ss = R['F'], R['DAY'], R['START'], R['TEST_D'], R['ss']
fit_logit, train_set = R['fit_predict'], R['train_set']
from sksurv.ensemble import RandomSurvivalForest
from sklearn.metrics import roc_auc_score, average_precision_score
SH = 15.0

def build(C):
    rows = []
    for D in pd.date_range(START, C - DAY):
        r = X[X.D == D].copy()
        if D + F * DAY <= C:
            o = partial_s(D, D + F * DAY)
            t = np.where(o.fm.notna(), o.s + SH, SH - 2); e = np.ones(len(o), bool)
            I = pd.DataFrame({'t': t, 'e': e}, index=o.index)
        else:
            if (C - D).days < 3: continue
            iv = intervals(D, C)
            fin = np.isfinite(iv.R)
            I = pd.DataFrame({'t': np.where(fin, (iv.L + iv.R.where(fin, 0)) / 2, iv.L) + SH, 'e': fin.values}, index=iv.index)
        r = r[r.channel_id.isin(I.index)]
        r['t'] = r.channel_id.map(I.t).values; r['e'] = r.channel_id.map(I.e).values.astype(bool)
        rows.append(r)
    tr = pd.concat(rows, ignore_index=True)
    tr['w'] = 0.5 ** ((tr.D.max() - tr.D).dt.days.values / 7)
    return tr

def run_rsf(C, te):
    tr = build(C)
    if len(tr) > a.nmax:                                     # 최근일 가중 표본추출 (RSF 는 가중치 미지원)
        tr = tr.iloc[np.random.default_rng(0).choice(len(tr), a.nmax, replace=False, p=tr.w / tr.w.sum())]
    A_, B_ = design(tr, te)
    y = np.array(list(zip(tr.e.values, np.clip(tr.t.values, 0.01, None))), dtype=[('e', bool), ('t', float)])
    m = RandomSurvivalForest(n_estimators=300, min_samples_leaf=40, max_features='sqrt', n_jobs=2, random_state=0, low_memory=False)
    m.fit(A_.values, y)
    sf = m.predict_survival_function(B_.values, return_array=True)   # n × 고유시점
    j = np.argmin(np.abs(sf.mean(0) - 0.2))
    return sf[:, j]

rk = lambda x: pd.Series(x).rank(pct=True).values
if a.validate:
    res = []
    last_full = TEST_D - F * DAY
    for vd in pd.date_range(last_full - 8 * DAY, last_full, freq=f'{max(9 // a.days, 1)}D')[:a.days]:
        va = X[X.D == vd].copy(); va['y'] = va.channel_id.map(label(vd, F)); va = va[va.y.notna()]
        t0 = time.time(); pr = run_rsf(vd, va); pl, _ = fit_logit(train_set(vd), va)
        out = []
        for w in (0, 0.3, 0.5, 1.0):
            p = w * rk(pr) + (1 - w) * rk(pl); out.append(average_precision_score(va.y, p))
        res.append(out); print(vd.date(), 'AP  로지스틱 %.4f | RSF0.3 %.4f | RSF0.5 %.4f | RSF단독 %.4f' % tuple(out), f'({time.time()-t0:.0f}s)', flush=True)
    print('평균 AP  로지스틱 %.4f | RSF0.3 %.4f | RSF0.5 %.4f | RSF단독 %.4f' % tuple(np.mean(res, 0)))
else:
    te = X[X.D == TEST_D]
    pr = run_rsf(TEST_D, te); pl, _ = fit_logit(train_set(TEST_D), te)
    p = a.w_rsf * rk(pr) + (1 - a.w_rsf) * rk(pl)
    key = te.channel_id + '_' + TEST_D.strftime('%Y-%m-%d')
    s2 = ss.copy(); s2['prediction'] = s2.row_id.map(dict(zip(key, p))); assert s2.prediction.notna().all()
    s2.to_csv(f'submission_F{F}_latest_rsf{a.w_rsf}.csv', index=False); print('저장 완료')
