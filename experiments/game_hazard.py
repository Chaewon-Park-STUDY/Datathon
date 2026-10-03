"""
==========================================================================================
 게임 시계열 × 생존분석(이산시간 위험) 결합 피처  →  기존 최고 설정(FINAL2)에 추가
==========================================================================================
[아이디어]
  사건 = '히트 영상' : 채널의 그때까지 평소 조회수(로그 중앙값)보다 δ 이상 높은 롱폼 영상
  1) 게임 시계열: 게임마다 '최근 14일 동안 그 게임 영상들이 각 채널 평소 대비 얼마나 잘 됐나'
                   = 유튜브 안에서의 게임 열기(heat). 날짜별로 계산 → 시간의존 공변량
  2) 채널 열기:    채널이 최근 28일 다룬 게임들의 열기 평균 (게임 매핑 없는 채널은 결측 + 표시)
  3) 이산시간 위험모형(= 영상 단위 로지스틱): P(히트 | 업로드 시점의 채널 열기, 모멘텀, 변동성, 업로드 간격)
     → 영상 하나가 히트일 확률 p_c  (기준일 D 이전 영상으로만 학습 → 누수 없음)
  4) 미래 창 F일 동안 n편 ~ Poisson(업로드율 × F), 그중 과반이 히트일 확률
     hazard_p = Σ_n Pois(n) · P(Binom(n, p_c) ≥ n//2+1)   (중앙값 상승 ≈ 과반이 평소보다 잘 됨)
  → rk_hazard_p, rk_ch_heat 를 기존 로지스틱(FINAL2 설정)에 추가

[실행]  python game_hazard.py --data ./epoch_data --validate      (18일 창 검증, 기존과 비교)
        python game_hazard.py --data ./epoch_data --F 28          (제출 파일)
==========================================================================================
"""
import sys, runpy, argparse, time
import numpy as np, pandas as pd
from scipy.stats import poisson, binom
ap = argparse.ArgumentParser()
ap.add_argument('--data', default='./epoch_data'); ap.add_argument('--F', type=int, default=18)
ap.add_argument('--validate', action='store_true'); ap.add_argument('--delta', type=float, default=0.3)
ap.add_argument('--out', default='submission_F28_game_hazard.csv')
g = ap.parse_args()
sys.argv = ['run_final.py', '--data', g.data, '--F', str(g.F), '--mode', 'lb7627', '--q_x', '--soft_k', '--build_only']
R = runpy.run_path('run_final.py')
X, L, label, ts, DAY, F, TEST_D, ss = R['X'], R['L'], R['label'], R['train_set'], R['DAY'], R['F'], R['TEST_D'], R['ss']
MAIN, IND, EXTRA, A = R['MAIN'], R['IND'], R['EXTRA'], R['A']
DATA = g.data.rstrip('/') + '/'
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import average_precision_score as AP

# ── 영상 단위 표: 업로드 시점 기준 채널 상태 (그 영상 이전 정보만) ──────────────────────────
V = L[['video_id', 'channel_id', 'published_at', 'lv']].sort_values(['channel_id', 'published_at']).copy()
gv = V.groupby('channel_id').lv
V['pmed'] = gv.transform(lambda s: s.expanding().median().shift(1))               # 그때까지 평소 수준
V['mom3'] = gv.transform(lambda s: s.rolling(3).mean().shift(1)) - V.pmed          # 직전 3편 모멘텀
V['cv10'] = gv.transform(lambda s: s.rolling(10, min_periods=3).std().shift(1))    # 직전 변동성
V['gap'] = V.groupby('channel_id').published_at.diff().dt.total_seconds() / 86400  # 직전 업로드와 간격(일)
V['resid'] = V.lv - V.pmed                                                         # 평소 대비 성과
V['hit'] = (V.resid >= g.delta).astype(float)
V['day'] = V.published_at.dt.normalize()
gm = pd.read_csv(DATA + 'video_igdb_games.csv').drop_duplicates('video_id').set_index('video_id').igdb_id
V['game'] = V.video_id.map(gm)

# ── 게임 열기 시계열 (날짜 × 게임): 최근 14일 그 게임 영상들의 평소 대비 성과 평균 ─────────────
days = pd.date_range(V.day.min(), TEST_D)
Vg = V.dropna(subset=['game', 'resid'])
heat = {}
for d in days:
    w = Vg[(Vg.published_at < d) & (Vg.published_at >= d - 14 * DAY)]
    heat[d] = w.groupby('game').resid.mean()
def ch_heat(d):
    """채널 열기 = 채널이 최근 28일 다룬 게임들의 그날 열기 평균"""
    w = V[(V.published_at < d) & (V.published_at >= d - 28 * DAY)].dropna(subset=['game'])
    w = w.assign(h=w.game.map(heat[d]))
    return w.groupby('channel_id').h.mean()
CH = {d: ch_heat(d) for d in days}
V['ch_heat'] = [CH[d].get(c, np.nan) for d, c in zip(V.day, V.channel_id)]

# ── 이산시간 위험모형: 영상 하나가 '히트'일 확률 ─────────────────────────────────────────────
FEATS = ['ch_heat', 'mom3', 'cv10', 'gap']
def hazard_p(D):
    tr = V[(V.published_at < D) & V.pmed.notna()]
    Xv = tr[FEATS].copy(); Xv['na_heat'] = Xv.ch_heat.isna().astype(float)
    med = Xv.median(); Xv = Xv.fillna(med)
    sc = StandardScaler().fit(Xv); m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xv), tr.hit)
    # 기준일 D 시점의 채널 상태
    pl = V[V.published_at < D].groupby('channel_id')
    st = pd.DataFrame({'ch_heat': CH[D], 'mom3': pl.lv.apply(lambda s: s.tail(3).mean()) - pl.lv.median(),
                       'cv10': pl.lv.apply(lambda s: s.tail(10).std()),
                       'gap': (D - pl.published_at.max()).dt.total_seconds() / 86400})
    st = st.reindex(pl.size().index)
    Xs = st[FEATS].copy(); Xs['na_heat'] = Xs.ch_heat.isna().astype(float); Xs = Xs.fillna(med)
    p = m.predict_proba(sc.transform(Xs))[:, 1]
    lam = X[X.D == D].set_index('channel_id').upl_rate.reindex(st.index).fillna(0.2).values * F
    NS = np.arange(40); pn = poisson.pmf(NS[None, :], lam[:, None]); pn[:, -1] += 1 - pn.sum(1)
    tail = binom.sf((NS // 2 + 1)[None, :] - 1, NS[None, :], p[:, None]); tail[:, 0] = 0
    return pd.Series((pn * tail).sum(1), index=st.index), st.ch_heat

t0 = time.time()
X['hazard_p'] = np.nan; X['ch_heat'] = np.nan
for D, idx in X.groupby('D').groups.items():
    hp, chh = hazard_p(D)
    X.loc[idx, 'hazard_p'] = X.loc[idx, 'channel_id'].map(hp).values
    X.loc[idx, 'ch_heat'] = X.loc[idx, 'channel_id'].map(chh).values
for c in ['hazard_p', 'ch_heat']:
    X['rk_' + c] = X.groupby('D')[c].rank(pct=True)
X['na_heat'] = X.ch_heat.isna().astype(float)
print(f'게임열기·위험 피처 생성 {time.time()-t0:.0f}s, 게임 열기 있는 채널 비율 {X.ch_heat.notna().mean():.2f}', flush=True)

def fit(tr, te, extra):
    a, b = tr[MAIN].copy(), te[MAIN].copy(); emp = a.columns[a.isna().all()]; a[emp] = .5; b[emp] = b[emp].fillna(.5)
    ii = IterativeImputer(max_iter=10, random_state=0).fit(a.to_numpy(copy=True))
    a = pd.DataFrame(ii.transform(a.to_numpy(copy=True)), columns=MAIN, index=a.index)
    b = pd.DataFrame(ii.transform(b.to_numpy(copy=True)), columns=MAIN, index=b.index)
    a['lag'], b['lag'] = a.lag.clip(0, 1), b.lag.clip(0, 1)
    for c in IND: a['na_' + c], b['na_' + c] = tr['na_' + c].values, te['na_' + c].values
    for c in extra:
        med = tr[c].median(); med = 0.5 if pd.isna(med) else med
        a[c] = tr[c].fillna(med).values; b[c] = te[c].fillna(med).values
    sw = tr.w.values * 0.5 ** ((tr.D.max() - tr.D).dt.days.values / 7); sc = StandardScaler().fit(a)
    m = LogisticRegression(C=A.C, max_iter=3000).fit(sc.transform(a), tr.y, sample_weight=sw)
    return m.predict_proba(sc.transform(b))[:, 1], pd.Series(m.coef_[0], a.columns)

CF = {'기존(FINAL2)': EXTRA, '+위험확률': EXTRA + ['rk_hazard_p'],
      '+채널게임열기': EXTRA + ['rk_ch_heat', 'na_heat'], '+둘다': EXTRA + ['rk_hazard_p', 'rk_ch_heat', 'na_heat']}
if g.validate:
    res = {k: [] for k in CF}; co = []
    for vd in pd.date_range(TEST_D - (F + 8) * DAY, TEST_D - F * DAY):
        va = X[X.D == vd].copy(); va['y'] = va.channel_id.map(label(vd, F)); va = va[va.y.notna()]; tr = ts(vd)
        for k, ex in CF.items():
            p, c = fit(tr, va, ex); res[k].append(AP(va.y, p))
            if k == '+둘다': co.append(c)
        print(vd.date(), ' '.join(f'{res[k][-1]:.4f}' for k in CF), flush=True)
    r = pd.DataFrame(res)
    for k in CF:
        d = r[k] - r['기존(FINAL2)']
        print(f'{k:14s} 평균AP {r[k].mean():.4f} 대비 {d.mean():+.4f} 개선 {int((d>0).sum())}/9 최대하락 {d.min():+.4f}')
    c = pd.concat(co, axis=1).mean(1); print('계수:', {x: round(c[x], 3) for x in ['lag', 'rk_hazard_p', 'rk_ch_heat', 'na_heat']})
    print('단독 AP(위험확률):', round(np.mean([AP(X[X.D == vd].assign(y=lambda d: d.channel_id.map(label(vd, F))).dropna(subset=['y']).y,
          X[X.D == vd].assign(y=lambda d: d.channel_id.map(label(vd, F))).dropna(subset=['y']).hazard_p.fillna(0))
          for vd in pd.date_range(TEST_D - (F + 8) * DAY, TEST_D - F * DAY)]), 4))
else:
    te = X[X.D == TEST_D]; key = te.channel_id + '_' + TEST_D.strftime('%Y-%m-%d'); tr = ts(TEST_D)
    for k in ['+위험확률', '+채널게임열기', '+둘다']:
        p, c = fit(tr, te, CF[k])
        s2 = ss.copy(); s2['prediction'] = s2.row_id.map(dict(zip(key, p))); assert s2.prediction.notna().all()
        name = g.out.replace('.csv', '_' + {'+위험확률': 'hazard', '+채널게임열기': 'heat', '+둘다': 'both'}[k] + '.csv')
        s2.to_csv(name, index=False); print('저장:', name)
