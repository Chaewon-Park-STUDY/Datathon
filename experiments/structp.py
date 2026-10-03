"""구조 모형(생성 모형) 피처: 라벨이 '미래 n편 중앙값 ≥ 문턱'이라는 사실을 그대로 확률로 계산.
 n ~ Poisson(λ·18),  각 영상 log조회수 ~ N(μ_c, σ_c)  (μ: 최근 가중평균, σ: 최근 잔차 표준편차)
 P(중앙값 ≥ v) = Σ_n Pois(n) · P(Binom(n, p_c) ≥ n//2+1),  p_c = P(영상 ≥ v),  n=0이면 0
 v = τ + log1p(과거중앙값), τ는 날짜마다 평균 확률이 0.2가 되도록 풀어서 결정(라벨 불필요)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import norm, poisson, binom
from scipy.optimize import brentq
from cens import X2, L, day
LL = L.assign(l=np.log1p(L.view_5d)).sort_values('published_at')
NS = np.arange(0, 40)
def struct_D(D, F=18, hl=5):
    h = LL[LL.published_at < D]
    rows = []
    for c, g in h.groupby('channel_id'):
        x = g.l.values
        if len(x) < 3: continue
        w = 0.5 ** (np.arange(len(x))[::-1] / hl); mu = (w * x).sum() / w.sum()
        sd = max(np.sqrt((w * (x - mu) ** 2).sum() / w.sum()), 0.25)
        rows.append((c, mu, sd, np.log1p(np.median(g.view_5d.values))))
    S = pd.DataFrame(rows, columns=['channel_id', 'mu', 'sd', 'lp']).set_index('channel_id')
    lam = X2[X2.Dt == D].set_index('channel_id').upl_rate.reindex(S.index).fillna(0.2).values * F
    pn = poisson.pmf(NS[None, :], lam[:, None]); pn[:, -1] += 1 - pn.sum(1)
    kreq = NS // 2 + 1
    def P(t):
        p = 1 - norm.cdf((t + S.lp.values - S.mu.values) / S.sd.values)
        tail = binom.sf(kreq[None, :] - 1, NS[None, :], p[:, None]); tail[:, 0] = 0
        return (pn * tail).sum(1)
    t = brentq(lambda t: P(t).mean() - 0.2, -5, 5)
    S['struct_p'] = P(t); S['struct_pup'] = 1 - pn[:, 0]
    return S[['struct_p', 'struct_pup']].reset_index().assign(D=D.strftime('%Y-%m-%d'))
ST = pd.concat([struct_D(D) for D in pd.to_datetime(sorted(X2.D.unique()))])
for c in ['struct_p', 'struct_pup']: X2[c] = X2.merge(ST, on=['channel_id', 'D'], how='left')[c].values
if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score as AUC
    V = X2[(X2.D >= '2026-08-01') & (X2.D <= '2026-08-14') & X2.y18.notna()]
    for c in ['struct_p', 'struct_pup', 'lag7']:
        print(c, round(V.groupby('D').apply(lambda g: AUC(g.y18, g[c].fillna(g[c].median()))).mean(), 4))
