"""
==========================================================================================
 Deep Gaussian Process + 구간중도절단(Interval-Censored) AFT — 실험용 (본선 제출용 아님)
==========================================================================================

[아이디어]
  정답 점수 s = log(1+미래 중앙값) - log(1+과거 중앙값) 를 '분류'하지 않고 직접 모형화한다.
  학습 데이터의 s 는 세 종류로 관측된다 (생존분석의 중도절단과 같은 구조):
    - 정확 관측   : 미래 창이 다 보이는 날짜 → s 를 정확히 안다
    - 좌측중도절단 : 미래 창에 영상이 0편 → s 는 매우 작다 (s <= -1 로 처리)
    - 구간중도절단 : 미래 창이 데이터 끝에서 잘린 날짜 → s ∈ [L, R] (run_final.py 의 intervals)
  모형:  s | x ~ Normal( μ(x), σ(x)² ),  [μ(x), log σ(x)] = 2층 Deep GP(x)
    → σ(x) 도 입력에 따라 달라지는 '이분산' 모형 (변동 큰 채널 = σ 큼 = 문턱을 넘기 쉬움)
  우도:  정확 → 정규 밀도,  구간 → Φ((R-μ)/σ) - Φ((L-μ)/σ),  좌측 → Φ((R-μ)/σ)
  학습:  Doubly Stochastic Variational Inference (Salimbeni & Deisenroth, 2017), 유도점 M개
  예측:  P(s >= t | x) 를 샘플로 평균, 문턱 t 는 '그날 평균 확률 = 0.2'가 되도록 결정

[실행]  run_final.py 와 같은 폴더에서
  python dgp_ic.py --data ./epoch_data --F 18 --validate     # 시간 기준 검증 (로지스틱과 비교 출력)
  python dgp_ic.py --data ./epoch_data --F 18                # 제출 파일 dgp_submission.csv
  필요 패키지: pip install jax optax

[연습 데이터 검증 결과]  (F=18, 검증일 08-06~08-14 9일, 데이터가 검증일에 끝나는 상황 재현)
  Deep GP 구간중도절단        평균 ROC 0.744   (하루 약 4분)
  로지스틱 + 소프트 라벨(0.7627) 평균 ROC 0.782   ← 9일 모두 로지스틱이 같거나 높음
  둘의 순위 평균              평균 ROC 0.779
  참고: 선형 이분산 구간중도절단 회귀 0.771
  → 신호가 약하고 행이 반복되는 데이터라 유연한 모형이 이득을 못 봄. 본선 제출에는 쓰지 않는다.
==========================================================================================
"""
import sys, runpy, argparse, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='./epoch_data')
ap.add_argument('--F', type=int, default=18)
ap.add_argument('--mode', default='lb7627', choices=['lb7627', 'latest'])
ap.add_argument('--validate', action='store_true')
ap.add_argument('--steps', type=int, default=2500, help='최적화 반복 수')
ap.add_argument('--M', type=int, default=64, help='층마다 유도점 수')
ap.add_argument('--H', type=int, default=4, help='숨은 층 차원')
ap.add_argument('--out', default='dgp_submission.csv')
a = ap.parse_args()

# ── run_final.py 를 '피처만 만들기' 모드로 실행해서 X, design, intervals 등을 가져온다 ─────────────
sys.argv = ['run_final.py', '--data', a.data, '--F', str(a.F), '--mode', a.mode, '--build_only']
R = runpy.run_path('run_final.py')
X, design, intervals, partial_s, label = R['X'], R['design'], R['intervals'], R['partial_s'], R['label']
F, DAY, START, TEST_D, ss = R['F'], R['DAY'], R['START'], R['TEST_D'], R['ss']
fit_logit = R['fit_predict'];  train_set = R['train_set']

import jax, jax.numpy as jnp, optax
from jax.scipy.stats import norm as jnorm
from jax.scipy.linalg import solve_triangular
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, average_precision_score
jax.config.update('jax_enable_x64', True)


# ═════════════════════════════════════════════════════════════════════════════════════
# 1. 구간중도절단 학습 데이터:  각 행에 [L, R] (정확 관측이면 L = R)
# ═════════════════════════════════════════════════════════════════════════════════════
def build_ic(C):
    rows = []
    for D in pd.date_range(START, C - 1 * DAY):
        r = X[X.D == D].copy()
        if D + F * DAY <= C:                          # 미래 창 전체가 보임
            o = partial_s(D, D + F * DAY)
            exact = o.fm.notna()
            Lb = np.where(exact, o.s, -np.inf)        # 0편: 좌측중도절단 (-inf, -1]
            Rb = np.where(exact, o.s, -1.0)
            I = pd.DataFrame({'L': Lb, 'R': Rb}, index=o.index)
        else:                                         # 미래 창이 C 에서 잘림 → 순서통계량 구간
            if (C - D).days < 3:
                continue
            I = intervals(D, C)
        r = r[r.channel_id.isin(I.index)]
        r['L'] = r.channel_id.map(I.L).values
        r['R'] = r.channel_id.map(I.R).values
        rows.append(r)
    tr = pd.concat(rows, ignore_index=True)
    tr['w'] = 0.5 ** ((tr.D.max() - tr.D).dt.days.values / 7)   # 최근일 가중 (run_final 과 동일)
    return tr


# ═════════════════════════════════════════════════════════════════════════════════════
# 2. 2층 Deep GP (변분 추론, whitened inducing points)
# ═════════════════════════════════════════════════════════════════════════════════════
def rbf(A_, B_, log_ls, log_var):
    d = (A_[:, None, :] - B_[None, :, :]) / jnp.exp(log_ls)
    return jnp.exp(log_var) * jnp.exp(-0.5 * jnp.sum(d ** 2, -1))

def layer_marginals(p, Xin, mean_out):
    """한 층의 GP 를 입력 Xin (n×Din) 에서 평가: 출력별 평균 (n×Dout), 분산 (n×Dout)"""
    M = p['Z'].shape[0]
    Kzz = rbf(p['Z'], p['Z'], p['log_ls'], p['log_var']) + 1e-5 * jnp.eye(M)
    Lz = jnp.linalg.cholesky(Kzz)
    Kzx = rbf(p['Z'], Xin, p['log_ls'], p['log_var'])
    A_ = solve_triangular(Lz, Kzx, lower=True)                       # M×n
    mean = A_.T @ p['m'] + mean_out                                  # n×Dout
    S = jnp.tril(p['Lraw'])                                          # Dout×M×M
    SA = jnp.einsum('dji,jn->din', S, A_)                            # Sᵀ A
    var = jnp.exp(p['log_var']) - jnp.sum(A_ ** 2, 0)[None, :] + jnp.sum(SA ** 2, 1)   # Dout×n
    return mean, jnp.clip(var.T, 1e-8)

def kl(p):
    S = jnp.tril(p['Lraw']); M = p['m'].shape[0]; Dout = p['m'].shape[1]
    return 0.5 * (jnp.sum(S ** 2) + jnp.sum(p['m'] ** 2) - M * Dout) - jnp.sum(jnp.log(jnp.abs(jnp.diagonal(S, axis1=1, axis2=2)) + 1e-12))

def forward(P, Xb, key, nsamp):
    """입력 Xb 를 두 층에 통과시켜 (μ, log σ) 샘플 nsamp 개를 만든다"""
    k1, k2 = jax.random.split(key)
    m1, v1 = layer_marginals(P['l1'], Xb, Xb @ P['W'])               # 1층: 평균함수 = 선형 투영 W
    h = m1[None] + jnp.sqrt(v1)[None] * jax.random.normal(k1, (nsamp,) + m1.shape)
    def l2(hs, k):
        m2, v2 = layer_marginals(P['l2'], hs, 0.)
        return m2 + jnp.sqrt(v2) * jax.random.normal(k, m2.shape)
    out = jax.vmap(l2)(h, jax.random.split(k2, nsamp))                # nsamp×n×2
    return out[..., 0] + P['b'][0], out[..., 1] + P['b'][1]

def loglik(mu, ls, Lb, Rb, typ):
    sig = jnp.exp(jnp.clip(ls, -3, 3))
    zL = (jnp.where(jnp.isfinite(Lb), Lb, 0.) - mu) / sig
    zR = (jnp.where(jnp.isfinite(Rb), Rb, 0.) - mu) / sig
    FL = jnp.where(jnp.isfinite(Lb), jnorm.cdf(zL), 0.)
    FR = jnp.where(jnp.isfinite(Rb), jnorm.cdf(zR), 1.)
    exact = jnorm.logpdf(zL) - jnp.log(sig)
    cens = jnp.log(jnp.clip(FR - FL, 1e-12))
    return jnp.where(typ == 0, exact, cens)

def fit_dgp(Xtr, Lb, Rb, w, steps, M, H, seed=0):
    n, d = Xtr.shape
    rng = np.random.default_rng(seed)
    U, Sv, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
    W = Vt[:H].T                                                     # 1층 평균함수: 주성분 H 개 (고정)
    Z1 = Xtr[rng.choice(n, M, replace=False)]
    typ = (Lb != Rb).astype(float)
    s_exact = Lb[typ == 0]
    P = dict(W=jnp.asarray(W),
             b=jnp.array([np.median(s_exact), np.log(np.std(s_exact) * 0.8)]),
             l1=dict(Z=jnp.asarray(Z1), log_ls=jnp.log(jnp.ones(d) * np.sqrt(d)), log_var=jnp.log(1.0),
                     m=jnp.zeros((M, H)), Lraw=jnp.tile(jnp.eye(M) * 1e-3, (H, 1, 1))),
             l2=dict(Z=jnp.asarray(Z1 @ W), log_ls=jnp.zeros(H), log_var=jnp.log(0.3),
                     m=jnp.zeros((M, 2)), Lraw=jnp.tile(jnp.eye(M), (2, 1, 1))))
    trainable = {k: v for k, v in P.items() if k != 'W'}
    wn = w / w.mean()
    Xj, Lj, Rj, Tj, Wj = map(jnp.asarray, (Xtr, Lb, Rb, typ, wn))
    B = min(512, n)

    def loss(tp, idx, key):
        Pf = {**tp, 'W': P['W']}
        mu, ls = forward(Pf, Xj[idx], key, 5)
        ll = loglik(mu, ls, Lj[idx][None], Rj[idx][None], Tj[idx][None]).mean(0)
        return -(n / B) * jnp.sum(Wj[idx] * ll) + kl(tp['l1']) + kl(tp['l2'])

    opt = optax.adam(0.01)
    st = opt.init(trainable)
    @jax.jit
    def step(tp, st, idx, key):
        l, g = jax.value_and_grad(loss)(tp, idx, key)
        up, st = opt.update(g, st, tp)
        return optax.apply_updates(tp, up), st, l
    key = jax.random.PRNGKey(seed)
    for it in range(steps):
        key, k1 = jax.random.split(key)
        idx = jnp.asarray(rng.choice(n, B, replace=False))
        trainable, st, l = step(trainable, st, idx, k1)
    return {**trainable, 'W': P['W']}

def predict_rising(P, Xte, nsamp=64):
    mu, ls = forward(P, jnp.asarray(Xte), jax.random.PRNGKey(1), nsamp)
    mu, sig = np.asarray(mu), np.exp(np.clip(np.asarray(ls), -3, 3))
    surv = lambda t: (1 - jnorm.cdf((t - mu) / sig)).mean(0)          # P(s >= t), 샘플 평균
    t = brentq(lambda t: float(np.asarray(surv(t)).mean()) - 0.2, -20, 20)
    return np.asarray(surv(t)), mu.mean(0), sig.mean(0)


# ═════════════════════════════════════════════════════════════════════════════════════
# 3. 검증 / 제출
# ═════════════════════════════════════════════════════════════════════════════════════
def run(C, te):
    tr = build_ic(C)
    A_, B_ = design(tr, te)                                           # run_final 과 같은 결측 처리
    mu_, sd_ = A_.mean(0), A_.std(0).replace(0, 1)
    Xa, Xb = ((A_ - mu_) / sd_).values, ((B_ - mu_) / sd_).values
    P = fit_dgp(Xa, tr.L.values.astype(float), tr.R.values.astype(float), tr.w.values, a.steps, a.M, a.H)
    return predict_rising(P, Xb)

if a.validate:
    res = []
    last_full = TEST_D - F * DAY
    for vd in pd.date_range(last_full - 8 * DAY, last_full):
        va = X[X.D == vd].copy(); va['y'] = va.channel_id.map(label(vd, F)); va = va[va.y.notna()]
        t0 = time.time()
        p_dgp, _, _ = run(vd, va)
        p_log, _ = fit_logit(train_set(vd), va)
        bl = 0.5 * pd.Series(p_dgp).rank(pct=True).values + 0.5 * pd.Series(p_log).rank(pct=True).values
        r = [roc_auc_score(va.y, p_dgp), roc_auc_score(va.y, p_log), roc_auc_score(va.y, bl)]
        res.append(r); print(vd.date(), 'ROC  DGP %.4f | 로지스틱 %.4f | 순위평균 %.4f' % tuple(r), f'({time.time()-t0:.0f}s)', flush=True)
    print('평균 ROC  DGP %.4f | 로지스틱 %.4f | 순위평균 %.4f' % tuple(np.mean(res, 0)))
else:
    te = X[X.D == TEST_D]
    p, mu, sig = run(TEST_D, te)
    ss['prediction'] = ss.row_id.map(dict(zip(te.channel_id + '_' + TEST_D.strftime('%Y-%m-%d'), p)))
    assert ss.prediction.notna().all()
    ss.to_csv(a.out, index=False); print('저장:', a.out)
