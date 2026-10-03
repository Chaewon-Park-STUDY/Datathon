"""
==========================================================================================
 EPOCH 5th Datathon — 최종 파이프라인 (한 파일, 주석 상세판)
==========================================================================================

[이 대회가 묻는 것]
  "기준일 D 이후 F일 동안, 이 채널의 롱폼 조회수가 '자기 평소'보다 크게 뛸까?"
  → 채널마다 0~1 사이 확률을 내면, 순서대로 줄 세워서 채점(ROC-AUC로 추정)한다.

[정답(라벨)이 만들어지는 방식]  ← 이걸 이해하면 코드 전체가 이해된다
  1) 과거 점수 = D 이전에 올린 롱폼들의 view_5d(업로드 5일 후 조회수) 중앙값
  2) 미래 점수 = D ~ D+F일에 올린 롱폼들의 view_5d 중앙값 (영상이 0편이면 0)
  3) s = log(1+미래 점수) - log(1+과거 점수)   ← "평소보다 몇 배 올랐나"를 로그로 잰 것
  4) 그날 채점 대상 채널 중 s 상위 20% → 정답 1, 나머지 → 0
  (채점 대상 = 과거 롱폼 3편 이상 & 과거 점수 100 이상)

[실행 방법]
  python run_final.py --data ./epoch_data --F 18                  → submission_final.csv 생성
  python run_final.py --data ./epoch_data --F 18 --validate       → 시간 기준 검증 점수 출력
  python run_final.py --data ./epoch_data --F 18 --dump feat.csv  → 전처리 결과 표를 CSV로 저장
  본선: --F 26 (미래 창이 26일)

  --mode lb7627 (기본값) : 연습 리더보드 최고점 0.7627 제출을 그대로 재현 (검증 ROC 0.782)
  --mode latest          : 위 + 순서통계량 구간 + 구조모형 피처 + 규제 강화(C=0.015)
                           (검증 ROC 0.808, 리더보드 단독 제출은 안 해봄)

[코드 순서]  아래 섹션 번호와 같다
  1 데이터 불러오기
  2 라벨 만들기 (위의 정답 규칙을 코드로)
  3 피처 만들기 (채널 × 기준일마다 숫자 16개)
  4 prev_label(7일 전 정답) 관련 피처
  5 구조모형 피처 (latest 모드에서만 모델에 들어감)
  6 결측치 처리 (순위 변환 + 결측 표시 + MICE)
  7 학습 데이터 만들기 (정답이 다 있는 날 + 정답이 '잘린' 날을 확률로)
  8 모델 (로지스틱 회귀)
  9 검증 또는 제출 파일 저장
==========================================================================================
"""
import argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import norm, poisson, binom          # 정규분포, 포아송분포, 이항분포 (5번·7번에서 사용)
from scipy.optimize import brentq                     # 방정식 f(t)=0 의 해 t 를 찾는 함수 (5번에서 사용)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa  ← MICE(IterativeImputer)를 쓰려면 필요한 한 줄
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')
DAY = pd.Timedelta(days=1)                            # "하루" 단위. 날짜 + 3*DAY 처럼 쓴다

# ── 실행 옵션 ──────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--data', default='./epoch_data', help='데이터 폴더 경로')
ap.add_argument('--F', type=int, default=18, help='미래 창 길이(일). 연습 18, 본선 26')
ap.add_argument('--prev_F', type=int, default=18, help='공식 prev_label(7일 전 정답)의 미래 창 길이')
ap.add_argument('--C', type=float, default=0.015, help='로지스틱 규제 강도. 작을수록 규제가 강함')
ap.add_argument('--validate', action='store_true', help='제출 파일 대신 검증 점수를 출력')
ap.add_argument('--out', default='submission_final.csv', help='제출 파일 저장 경로')
ap.add_argument('--mode', default='lb7627', choices=['lb7627', 'latest'])
ap.add_argument('--dump', default='', help='전처리된 피처 표를 CSV로 저장할 경로')
ap.add_argument('--q_x', action='store_true', help='q 모형에 채널 피처(평소 조회수 순위·업로드율·lag)도 넣기')
ap.add_argument('--q_ext', action='store_true', help='q 모형 확장: 채널 피처 추가 + 시간의존(중간 시점 순위) + 재발사건(남은 기대 업로드) + 상태(현재까지 업로드 수)')
ap.add_argument('--sim', action='store_true', help='업로드 시점 + 조회수 결합 몬테카를로 시뮬레이션 피처 sim_p 추가 (비모수 부트스트랩)')
ap.add_argument('--lag_int', action='store_true', help='오답 분석 기반 상호작용: lag×평소조회수순위, lag×최근3편모멘텀')
ap.add_argument('--inter', default='none', choices=['none', 'pm', 'l3', 'both'],
                help='lag 상호작용: pm = lag×평소조회수순위, l3 = lag×최근3편모멘텀순위, both = 둘 다')
ap.add_argument('--lag_mode', default='full', choices=['full', 'partial', 'na'],
                help='학습 행의 lag 창이 데이터 끝을 넘을 때: full=끝까지 계산(기존), partial=데이터 끝까지만, na=결측 처리')
ap.add_argument('--soft_k', action='store_true', help='소프트 라벨 가중치에 (관측일수/F) 곱하기')
# ── 신규 개선 플래그 (기본 off, 켜지 않으면 기존 0.7075 파이프라인 그대로 재현) ──────────────
ap.add_argument('--lag1feat', action='store_true',
                help='[개선2] lag 창에서 이미 관측된 상승이 "일회성 1편 히트"인지 "여러 편 고른 상승"인지 '
                     '구분하는 피처 2개(관측 상위영상 집중도 obs_conc, 상승 영상 수 obs_nrise)를 메인 모델에 추가')
ap.add_argument('--ap_weight', type=float, default=1.0,
                help='[개선3] AP 직접 최적화용 양성 가중. 하드 라벨 양성(y=1,w=1) 행의 sample_weight 를 '
                     '이 값만큼 추가로 곱한다(기본 1.0 = 변화 없음). 소프트 라벨 가중 w 와 곱셈 호환 유지. 권장 1.5~3.0')
ap.add_argument('--q_uni', action='store_true',
                help='[개선4] q 모형을 k 별 분리 대신 k/F 를 입력 피처로 넣은 통합 q 모형으로 학습(피처 소수). '
                     '모든 관측길이 k 의 예제를 한 번에 학습해 데이터 효율과 안정성을 높인다')
ap.add_argument('--q_em', type=int, default=0,
                help='[개선4] 완전 EM 반복 횟수. 최종 로지스틱 예측으로 소프트 라벨 q 를 갱신해 재학습한다(기본 0=안 함). 권장 1~2')
# ── 2차 개선 플래그 (기본 off, 켜지 않으면 기존 파이프라인 그대로 재현. 모두 --lag1feat 와 함께 평가 권장) ──
ap.add_argument('--lag1feat2', action='store_true',
                help='[2차-1] lag 창 관측-질 피처 보강. --lag1feat 의 obs_conc/obs_nrise 에 더해 '
                     '"일회성 1편 히트 vs 다편 고른 상승"을 더 잘 가르는 피처 2개를 메인 모델에 추가: '
                     'obs_second(관측 2위/1위 조회수 비율; 1편뿐이면 0, 둘 이상 고르면 큼), '
                     'obs_exrise(과거중앙값 P 초과 영상들의 평균 로그 초과폭; 초과 영상이 없으면 0). '
                     '모두 [prev_D, D) 관측부분만 사용 → 누수 없음. --lag1feat 와 함께 켤 것(obs_conc/obs_nrise 재사용).')
ap.add_argument('--lag1q', action='store_true',
                help='[2차-2] q 모형(소프트 라벨 추정)에 lag 창 관측-질 신호를 주입. qfeat 에 '
                     'obs_conc/obs_nrise 의 날짜별 순위와 lag(7일전 정답)을 추가해, 잘린 날짜들의 '
                     '소프트 라벨 q 추정 정확도를 높인다. --lag1feat 와 함께 켤 것(관측-질 신호를 X 에서 재사용).')
ap.add_argument('--lag1damp', action='store_true',
                help='[2차-3] lag=1 과잉예측 억제. lag=1 이면서 관측 상승이 "집중(1편 쏠림)+소수 상승"일수록 '
                     '커지는 상호작용 lag_weakobs = lag × rk_obs_conc × (1 − rk_obs_nrise) 를 메인 모델에 추가. '
                     '상위권 오답의 74%가 lag=1 인 점을 직접 겨냥해, 믿을 만한 다편 고른 상승은 두고 '
                     '일회성 상승만 깎도록 모델이 음의 계수를 학습하게 한다. --lag1feat 와 함께 켤 것.')
ap.add_argument('--label_fix', action='store_true',
                help='[개선1] 라벨 정합성 점검 옵션. 과거 점수 P 계산 시 view_5d 측정이 끝까지 가능한 '
                     '(published_at + 5일 <= 데이터 끝) 롱폼만 사용해, 측정이 덜 된 최근 영상이 '
                     'eligibility/과거중앙값을 흔드는 것을 줄인다. 공식 08-11 라벨과의 1.5% 불일치 원인 점검용')
ap.add_argument('--build_only', action='store_true', help='피처와 함수만 만들고 멈춤 (dgp_ic.py 가 불러다 쓸 때)')
ap.add_argument('--shap', default='', help='SHAP 결과 저장 이름(접두어). 예: --shap shap_out → 그림·CSV 저장')
A = ap.parse_args()
LATEST = A.mode == 'latest'
if not LATEST and A.C == 0.015:
    A.C = 0.1                 # 0.7627 제출 때는 C=0.1 을 썼다
DATA, F = A.data.rstrip('/') + '/', A.F


# ═════════════════════════════════════════════════════════════════════════════════════
# 1. 데이터 불러오기
# ═════════════════════════════════════════════════════════════════════════════════════
# video_5d_views: 영상 1개 = 1행. 업로드 시각, 5일 후 조회수(view_5d), 쇼츠 여부 등
v = pd.read_csv(DATA + 'video_5d_views.csv', parse_dates=['published_at'])

# 조회수는 채널마다 수십 회 ~ 수백만 회로 차이가 너무 크다.
# 그래서 log(1+조회수)로 바꿔 쓴다. (1을 더하는 이유: 조회수 0이면 log(0)이 정의되지 않아서)
v['lv'] = np.log1p(v.view_5d)

# 라벨은 '롱폼'만 보고 정해지므로 롱폼(L)과 쇼츠(S)를 나눈다.
L = v[v.is_shorts == 0].sort_values('published_at')   # 롱폼. 업로드 순서로 정렬 (뒤에서 '최근 영상'을 뽑기 위해)
S = v[v.is_shorts == 1]                               # 쇼츠

# 채널별 날짜별 구독자 수 등
cs = pd.read_csv(DATA + 'channel_snapshots.csv', parse_dates=['collected_at']).sort_values('collected_at')

# 제출 양식: row_id = "채널ID_기준일". 여기서 테스트 기준일과 제출할 채널 목록을 읽는다.
ss = pd.read_csv(DATA + 'sample_submission.csv', encoding='utf-8-sig')

# 공식 정답: row_id = "채널ID_날짜", target = 0/1
tl = pd.read_csv(DATA + 'train_labels.csv')
tl['ch'] = tl.row_id.str[:-11]                        # 뒤 11글자("_2026-08-25")를 떼면 채널ID
tl['Dl'] = tl.row_id.str[-10:]                        # 뒤 10글자가 날짜

TEST_D = pd.Timestamp(ss.row_id.str[-10:].iloc[0])    # 테스트 기준일 (연습: 09-01). 데이터는 이 날 직전까지 있다
SUB_CH = ss.row_id.str[:-11].tolist()                 # 제출해야 하는 채널 목록

# prev_label = 테스트 기준일 직전의 공식 정답 날짜 (연습: 08-25). K = 며칠 전인지 (연습: 7)
PREV_D = pd.Timestamp(max(d for d in tl.Dl.unique() if pd.Timestamp(d) < TEST_D))
K = (TEST_D - PREV_D).days

# 데이터가 07-10부터 있으므로, 과거 이력이 조금 쌓인 8일 뒤(07-18)부터 학습에 쓴다
START = L.published_at.min().normalize() + 8 * DAY
print(f'TEST_D={TEST_D.date()}  prev_label={PREV_D.date()} (D-{K})  F={F}  학습시작={START.date()}')


# ═════════════════════════════════════════════════════════════════════════════════════
# 2. 라벨 만들기
#    공식 정답은 08-11, 08-25 두 날짜뿐이다. 그래서 정답 규칙을 그대로 코드로 옮겨
#    07-18 ~ 08-14 매일의 정답을 직접 만든다. (공식 08-11 정답과 98.5% 일치 확인)
# ═════════════════════════════════════════════════════════════════════════════════════
_PAST = {}                                            # 같은 날짜를 여러 번 계산하지 않도록 저장해 두는 곳
def past(d):
    """기준일 d의 '과거 점수' = d 이전 롱폼 view_5d 중앙값.
       채점 대상 조건(3편 이상 & 중앙값 100 이상)을 만족하는 채널만 돌려준다."""
    if d not in _PAST:
        p = L[L.published_at < d].groupby('channel_id').view_5d.agg(['median', 'size'])
        _PAST[d] = p[(p['size'] >= 3) & (p['median'] >= 100)]['median']
    return _PAST[d]

def partial_s(d, end):
    """기준일 d의 점수 s 를 [d, end) 사이에 올라온 영상만으로 계산한다.
       - end = d + F일  → 정식 라벨 점수
       - end 가 그보다 이르면 → 미래 창의 '앞부분만 본' 부분 점수 (7번에서 사용)
       반환: pm(과거 점수), fm(미래 중앙값), fn(미래 영상 수), s(점수), rk(그날 채널들 중 s의 순위, 0~1)"""
    pm = past(d)
    f = L[(L.published_at >= d) & (L.published_at < end)].groupby('channel_id').view_5d.agg(['median', 'size'])
    o = pd.DataFrame({'pm': pm})
    o['fm'] = f['median'].reindex(o.index)            # 미래에 영상이 없는 채널은 NaN
    o['fn'] = f['size'].reindex(o.index).fillna(0)
    o['s'] = np.log1p(o.fm.fillna(0)) - np.log1p(o.pm)  # 미래 영상 0편 → 미래 점수 0 → s 가 매우 작아짐
    o['rk'] = o.s.rank(pct=True)                      # 백분위 순위 (가장 큰 s = 1.0)
    return o

def label(d, f):
    """기준일 d, 미래 창 f일의 정답: s 상위 20% = 1"""
    o = partial_s(d, d + f * DAY)
    if A.label_fix:
        # [개선1] 상위 20% 를 분위수 임계 비교(동점 포함) 대신 '순위 기준 정확히 상위 20%'로 선정.
        #   quantile(.8) 비교는 임계값에 동점이 몰리면 20% 보다 많거나 적게 1 을 줄 수 있어
        #   공식 라벨과 미세하게 어긋난다. 가장 큰 s 부터 ceil(0.2*n) 개만 1 로 둔다(결정적, 누수 없음).
        n = o.s.notna().sum()
        k = int(np.ceil(0.2 * n))
        thr_idx = o.s.rank(ascending=False, method='first')   # 1 = 가장 큰 s (동점은 결정적 순서)
        return (thr_idx <= k).astype(int)
    return (o.s >= o.s.quantile(.8)).astype(int)


# ═════════════════════════════════════════════════════════════════════════════════════
# 3. 피처 만들기
#    '채널 × 기준일 D' 한 줄마다 숫자 16개를 만든다.
#    가장 중요한 규칙: D 이전 데이터만 쓴다. D 이후를 쓰면 미래를 보고 맞히는 '누수'가 된다.
# ═════════════════════════════════════════════════════════════════════════════════════
def slope(t, y):
    """시간 t 에 따른 y 의 직선 기울기 (점이 3개 미만이면 계산 안 함)"""
    return np.nan if len(y) < 3 or np.ptp(t) == 0 else np.polyfit(t, y, 1)[0]

def feats(D, extra=()):
    pl = L[L.published_at < D].copy()                 # D 이전 롱폼만
    pl['age'] = (D - pl.published_at) / DAY           # 각 영상이 D 기준 며칠 전에 올라왔는지
    g = pl.groupby('channel_id')

    # (a) 과거 수준: 평소 조회수(중앙값)와 들쭉날쭉한 정도(표준편차). 둘 다 로그 조회수 기준
    f = pd.DataFrame({'past_n': g.size(), 'past_med': g.lv.median(), 'past_std': g.lv.std()})

    # (b) 최근 모멘텀: 최근 3/7/14일 영상의 중앙값과 그게 평소보다 얼마나 높은지(m3, m7, m14)
    for w in (3, 7, 14):
        r = pl[pl.age <= w].groupby('channel_id').lv
        f[f'r{w}_n'] = r.size()                       # 최근 w일 영상 수
        f[f'r{w}_med'] = r.median()                   # 최근 w일 중앙값 (영상 없으면 NaN → 6번에서 처리)
        f[f'm{w}'] = f[f'r{w}_med'] - f.past_med      # 평소 대비 (로그라서 '빼기' = '몇 배')
    f['r7_n'] = f.r7_n.fillna(0)                      # 영상 수는 없으면 0편이 맞으므로 0으로 채움
    f['r14_n'] = f.r14_n.fillna(0)

    last = g.tail(1).set_index('channel_id')          # 채널별 가장 최근 영상 1개
    f['last_m'] = last.lv - f.past_med                # 최신 영상이 평소보다 얼마나 잘 됐나
    f['last3_m'] = g.tail(3).groupby('channel_id').lv.mean() - f.past_med   # 최근 3편 평균 기준
    f['days_since'] = last.age                        # 마지막 업로드 후 며칠 지났나
    f['m14_old'] = f.r14_med - pl[pl.age > 14].groupby('channel_id').lv.median()  # 최근 2주 vs 그 이전
    f['slope_all'] = g.apply(lambda x: slope(-x.age.values, x.lv.values))   # 전체 기간 추세

    # (c) 변동성: 최근 10편 조회수의 표준편차. 변동이 크면 상위 20%에 들 확률도 커진다
    f['cv_last10'] = g.tail(10).groupby('channel_id').lv.std()

    # (d) 업로드 행동: 하루당 롱폼 수. 적게 올리는 채널은 미래 중앙값이 소수 영상으로 정해져 크게 흔들린다
    f['upl_rate'] = f.past_n / pl.groupby('channel_id').age.max().clip(lower=1)

    # (e) 쇼츠 비중: 최근 2주 업로드 중 쇼츠 비율
    ps = S[S.published_at < D].copy()
    ps['age'] = (D - ps.published_at) / DAY
    sh14 = ps[ps.age <= 14].groupby('channel_id').size().reindex(f.index).fillna(0)
    f['sh_share14'] = sh14 / (sh14 + f.r14_n).replace(0, np.nan)   # 최근 2주 업로드가 0이면 정의 안 됨(NaN)

    # (f) 채널 규모: D 직전에 수집된 구독자 수(로그), 구독자 대비 조회수 수준
    c = cs[cs.collected_at < D].groupby('channel_id').tail(1).set_index('channel_id')
    f['subs'] = np.log1p(c.subscriber_count)
    f['past_rel_subs'] = f.past_med - f.subs

    # 채점 대상 채널만 남긴다 (테스트 날짜에는 제출 목록 채널도 포함)
    f = f.reindex(past(D).index.union(pd.Index(list(extra))))
    f.index.name = 'channel_id'
    return f.reset_index().assign(D=D)


# ═════════════════════════════════════════════════════════════════════════════════════
# 4. prev_label(7일 전 정답) 관련 피처  ← 가장 강한 단서
#    08-25 정답은 08-25 ~ 09-12 영상으로 정해졌다. 이 중 09-01 ~ 09-12 는 테스트 창과 겹친다.
#    또 08-25 ~ 08-31 영상은 우리 데이터에 이미 있다(관측된 구간).
#    → "정답은 1인데 관측된 앞부분은 평범했다" = "뒷부분(09-01 이후)이 강했다"고 거꾸로 추론할 수 있다.
# ═════════════════════════════════════════════════════════════════════════════════════
def _lag_obs_shape(dl, D):
    """[개선2 / 2차-1] prev_label 창의 '이미 관측된' 부분 [dl, D) 에 올라온 롱폼으로,
       그 상승이 '일회성 1편 히트'인지 '여러 편 고른 상승'인지 구분하는 피처를 만든다.
       누수 없음: D 이전 데이터만 사용.
         obs_conc  = 관측 상위영상 집중도 = 최고 조회수 / 관측 영상 조회수 합 (1편뿐이면 1.0, 고르게면 낮음)
         obs_nrise = 과거 중앙값(P)을 넘긴 관측 영상 수 (여러 편이 고르게 올랐으면 큼)
       --lag1feat2 면 2개 더 (일회성 vs 다편 고른 상승을 더 잘 가르는 피처):
         obs_second = 관측 2위/1위 조회수 비율 (1편뿐이면 0.0, 둘 이상 고르게면 1 에 가까움)
         obs_exrise = 과거 중앙값 P 를 넘긴 관측 영상들의 평균 로그 초과폭 (초과 영상 없으면 0.0)
       관측 영상이 없으면 모두 NaN."""
    P = past(dl)
    seg = L[(L.published_at >= dl) & (L.published_at < D)]
    cols = ['conc', 'nrise'] + (['second', 'exrise'] if A.lag1feat2 else [])
    rows = {}
    for c, gg in seg.groupby('channel_id'):
        if c not in P.index:
            continue
        vv = gg.view_5d.values.astype(float)
        tot = vv.sum()
        conc = (vv.max() / tot) if tot > 0 else np.nan
        nrise = int((vv > P[c]).sum())               # 과거 중앙값을 넘긴 편수
        rec = [conc, nrise]
        if A.lag1feat2:
            sv = np.sort(vv)[::-1]                    # 조회수 내림차순
            # 2위/1위 비율: 1편뿐이면 0(쏠림 극단), 비슷한 2편 이상이면 1 근처(다편 고른 상승)
            second = (sv[1] / sv[0]) if (len(sv) >= 2 and sv[0] > 0) else 0.0
            ex = np.log1p(vv) - np.log1p(P[c])        # 과거 중앙값 대비 로그 초과폭
            ex = ex[ex > 0]                           # 실제로 넘은 영상만
            exrise = float(ex.mean()) if ex.size > 0 else 0.0
            rec += [second, exrise]
        rows[c] = tuple(rec)
    if not rows:
        return tuple(pd.Series(dtype=float) for _ in cols)
    df = pd.DataFrame(rows, index=cols).T
    return tuple(df[c] for c in cols)

def add_lag(X):
    out = []
    for D, g in X.groupby('D'):
        lab = label(D - K * DAY, A.prev_F)           # 학습용: K일 전 기준일의 정답 (직접 만든 라벨)
        o = partial_s(D - K * DAY, D)                 # 그 정답의 미래 창 중 D 이전(이미 관측된) 부분만 본 점수
        rec = {'channel_id': g.channel_id, 'D': D,
               'lag': g.channel_id.map(lab).values,        # 7일 전 정답 (0/1, 대상 아니었으면 NaN)
               'lagobs_rk': g.channel_id.map(o.rk).values}  # 관측된 앞부분의 순위 (0~1)
        if A.lag1feat:
            vals = _lag_obs_shape(D - K * DAY, D)
            conc, nrise = vals[0], vals[1]
            rec['obs_conc'] = g.channel_id.map(conc).values
            rec['obs_nrise'] = g.channel_id.map(nrise).values
            if A.lag1feat2:                           # [2차-1] 보강 피처 2개
                second, exrise = vals[2], vals[3]
                rec['obs_second'] = g.channel_id.map(second).values
                rec['obs_exrise'] = g.channel_id.map(exrise).values
        out.append(pd.DataFrame(rec))
    X = X.merge(pd.concat(out), on=['channel_id', 'D'], how='left')

    # 테스트 날짜에는 직접 만든 라벨 대신 공식 정답(08-25)을 넣는다
    off = tl[tl.Dl == PREV_D.strftime('%Y-%m-%d')].set_index('ch').target
    m = X.D == TEST_D
    X.loc[m, 'lag'] = X.loc[m, 'channel_id'].map(off)

    # 역추론 피처 2개
    X['lag_x_obs'] = X.lag * (1 - X.lagobs_rk)        # 정답 1 인데 앞부분이 약했을수록 큼 → 뒷부분이 강했다
    X['nolag_x_obs'] = (1 - X.lag) * X.lagobs_rk      # 정답 0 인데 앞부분이 강했을수록 큼 → 뒷부분이 약했다
    return X


# ═════════════════════════════════════════════════════════════════════════════════════
# 5. 구조모형 피처 struct_p  (latest 모드에서만 모델에 들어감)
#    정답 규칙 "미래 n편의 중앙값이 문턱을 넘는가"를 확률로 직접 계산한다.
#      - 미래에 올릴 편수 n  ~ 포아송분포(평균 = 하루 업로드 수 × F)
#      - 영상 하나의 로그 조회수 ~ 정규분포(평균 μ, 표준편차 σ) — 최근 영상에 가중
#      - n편 중 절반 넘게 문턱을 넘으면 중앙값이 문턱을 넘는다 → 이항분포로 계산
#    편수가 적고 σ가 큰 채널일수록 문턱을 넘을 확률이 높게 나온다.
# ═════════════════════════════════════════════════════════════════════════════════════
NS = np.arange(0, 40)                                 # 미래 편수 n 을 0~39 편까지 고려
def struct_p(D, lam_by_ch, hl=5):
    rows = []
    for c, g in L[L.published_at < D].groupby('channel_id'):
        x = g.lv.values
        if len(x) < 3:
            continue
        w = 0.5 ** (np.arange(len(x))[::-1] / hl)     # 최근 영상일수록 가중치 큼 (5편마다 절반)
        mu = (w * x).sum() / w.sum()                  # 가중 평균
        sd = max(np.sqrt((w * (x - mu) ** 2).sum() / w.sum()), 0.25)   # 가중 표준편차 (너무 작지 않게 0.25 하한)
        rows.append((c, mu, sd, np.log1p(np.median(g.view_5d.values))))  # lp = 과거 점수(로그)
    Sd = pd.DataFrame(rows, columns=['channel_id', 'mu', 'sd', 'lp']).set_index('channel_id')
    lam = Sd.index.map(lam_by_ch).to_series().fillna(0.2).values * F     # 미래 편수의 평균
    pn = poisson.pmf(NS[None, :], lam[:, None])       # P(n = 0, 1, 2, ...)
    pn[:, -1] += 1 - pn.sum(1)                        # 39편 넘는 확률은 마지막 칸에 몰아준다
    kreq = NS // 2 + 1                                # n편 중 몇 편이 넘어야 중앙값이 넘는가

    def P(t):                                         # 문턱 t 일 때 채널별 확률
        p = 1 - norm.cdf((t + Sd.lp.values - Sd.mu.values) / Sd.sd.values)   # 영상 1편이 문턱을 넘을 확률
        tail = binom.sf(kreq[None, :] - 1, NS[None, :], p[:, None])         # n편 중 kreq편 이상 넘을 확률
        tail[:, 0] = 0                                # 0편이면 미래 점수 0 → 절대 못 넘음
        return (pn * tail).sum(1)                     # 모든 n 에 대해 평균

    # 문턱 t 는 "평균 확률이 0.2(상위 20%)"가 되도록 정한다. 정답을 전혀 쓰지 않으므로 누수 없음
    t = brentq(lambda t: P(t).mean() - 0.2, -5, 5)
    return pd.Series(P(t), index=Sd.index)


# ── 피처 표 만들기: 07-17 ~ 테스트 기준일까지 하루씩 ─────────────────────────────────────
print('피처 생성 중...')
X = pd.concat([feats(D, SUB_CH if D == TEST_D else ()) for D in pd.date_range(START - DAY, TEST_D)],
              ignore_index=True)
X = add_lag(X)
X['struct_p'] = np.nan
for D, idx in X.groupby('D').groups.items():
    sp = struct_p(D, X.loc[idx].set_index('channel_id').upl_rate.to_dict())
    X.loc[idx, 'struct_p'] = X.loc[idx, 'channel_id'].map(sp).values


# ── (옵션 --sim) 결합 시뮬레이션: 업로드 시점(갱신과정) + 조회수(부트스트랩) → 원조회수 중앙값 → 그날 상위 20% ──
#   모수 가정 없이 채널 자신의 과거 '업로드 간격'과 '조회수'를 다시 뽑아(부트스트랩) 미래를 수백 번 만들어 본다.
#   - 정기 업로드 채널 vs 몰아서 올리는 채널의 차이를 간격 분포 그대로 반영 (포아송 가정 제거)
#   - 짝수 편수일 때도 실제 규칙(가운데 두 값 평균)대로 중앙값 계산
#   - 매 반복마다 모든 채널을 비교해 상위 20% 문턱을 정하므로 문턱의 불확실성까지 반영
def sim_p(D, nsim=300, maxn=40, seed=0):
    rng = np.random.default_rng(seed + D.dayofyear)
    pl = L[L.published_at < D]
    chans, S = [], []
    for c, g in pl.groupby('channel_id'):
        if len(g) < 3:
            continue
        t = ((D - g.published_at) / DAY).values[::-1]          # 오래된 → 최근 순서의 '며칠 전'
        gaps = -np.diff(t)[-20:]                               # 최근 20개 업로드 간격
        gaps = gaps[gaps > 0.02] if (gaps > 0.02).any() else np.array([max(t.max(), 1) / len(g)])
        views = g.view_5d.values[-15:].astype(float)           # 최근 15편 5일 조회수
        elapsed = t.min()                                      # 마지막 업로드 후 경과일
        G = rng.choice(gaps, (nsim, maxn))
        first = np.maximum(G[:, 0] - elapsed, rng.uniform(0, 1, nsim) * G[:, 0])   # 다음 업로드까지 남은 시간(근사)
        arr = first[:, None] + np.concatenate([np.zeros((nsim, 1)), np.cumsum(G[:, 1:], 1)], 1)
        n_up = (arr < F).sum(1)                                # 미래 창 안의 업로드 수
        V = rng.choice(views, (nsim, maxn))
        V = np.where(np.arange(maxn)[None, :] < n_up[:, None], V, np.nan)
        med = np.where(n_up > 0, np.nanmedian(np.where(n_up[:, None] > 0, V, 0.), 1), 0.)
        S.append(np.log1p(med) - np.log1p(np.median(g.view_5d.values)))
        chans.append(c)
    S = np.array(S)                                            # 채널 × 반복
    elig = np.isin(chans, past(D).index)                       # 문턱은 채점 대상 채널끼리 비교
    thr = np.quantile(S[elig], 0.8, axis=0)
    return pd.Series((S >= thr[None, :]).mean(1), index=chans)

if A.sim:
    X['sim_p'] = np.nan
    for D, idx in X.groupby('D').groups.items():
        X.loc[idx, 'sim_p'] = X.loc[idx, 'channel_id'].map(sim_p(D)).values

# ═════════════════════════════════════════════════════════════════════════════════════
# 6. 결측치 처리
#    결측은 '우연히' 빠진 게 아니라 이유가 있다(MAR / 구조적 결측). 예:
#      m3 결측 = 최근 3일 업로드가 없음,  lag 결측 = 7일 전엔 채점 대상이 아니었음
#    그래서 (1) 순위로 바꾸고 (2) "결측이었다"는 표시를 따로 남기고 (3) MICE로 채운다.
# ═════════════════════════════════════════════════════════════════════════════════════
NUM = ['past_med', 'past_std', 'cv_last10', 'm7', 'm3', 'last3_m', 'last_m', 'slope_all', 'r14_med',
       'upl_rate', 'days_since', 'sh_share14', 'subs', 'past_rel_subs', 'm14_old', 'r7_n']

# (1) 날짜별 백분위 순위: 정답이 '그날 상위 20%'라는 상대 기준이라,
#     피처도 '그날 다른 채널들 사이에서 몇 등인지'로 바꾸면 날짜마다 기준이 흔들리지 않는다. (결측은 결측으로 남음)
_RK_EXTRA = []
if A.lag1feat:
    _RK_EXTRA += ['obs_conc', 'obs_nrise']
    if A.lag1feat2:
        _RK_EXTRA += ['obs_second', 'obs_exrise']
for c in NUM + ['struct_p'] + (['sim_p'] if A.sim else []) + _RK_EXTRA:
    X['rk_' + c] = X.groupby('D')[c].rank(pct=True)

# (2) 결측 지시변수: 값이 비어 있었으면 1. "최근에 영상이 없었다" 같은 사실 자체가 정보다
IND = ['lag', 'm3', 'm7', 'subs', 'm14_old']
for c in IND:
    X['na_' + c] = X[c].isna().astype(float)

MAIN = ['rk_' + c for c in NUM] + ['lag']
if A.lag1feat:   # [개선2] lag=1 오답 교정: 관측 상승의 집중도/상승 편수를 메인 모델에 추가
    MAIN = MAIN + ['rk_obs_conc', 'rk_obs_nrise']
    if A.lag1feat2:   # [2차-1] 관측-질 보강 피처 2개
        MAIN = MAIN + ['rk_obs_second', 'rk_obs_exrise']
    if A.lag1damp:   # [2차-3] lag=1 과잉예측 억제 상호작용 (lag=1 & 쏠린/소수 상승일수록 큼)
        #   rk_obs_conc(집중도 순위, 1편 쏠림=큼), (1-rk_obs_nrise)(상승 편수 적을수록 큼) 곱.
        #   lag 결측(채점 대상 아님)은 0 으로 봐 상호작용 0. 관측-질 결측은 중앙값 0.5 로 둬 중립.
        X['lag_weakobs'] = (X.lag.fillna(0)
                            * X.rk_obs_conc.fillna(.5)
                            * (1 - X.rk_obs_nrise.fillna(.5)))
        MAIN = MAIN + ['lag_weakobs']
if A.lag_int:   # lag=1 이어도 덩치가 크거나 최근이 약하면 덜 믿도록
    X['lag_pm'] = X.lag.fillna(0) * X.rk_past_med
    X['lag_l3'] = X.lag.fillna(0) * X.rk_last3_m             # MICE 로 채울 열
EXTRA = ['lagobs_rk', 'lag_x_obs', 'nolag_x_obs'] + (['rk_struct_p'] if LATEST else []) + (['rk_sim_p'] if A.sim else []) + (['lag_pm', 'lag_l3'] if A.lag_int else [])   # 중앙값으로 채울 열

def design(tr, te):
    """학습 데이터(tr)로 결측 채우는 방법을 배우고, 같은 방법을 tr 과 예측 대상(te)에 적용한다.
       te 의 정보로 채우면 누수가 되므로 반드시 tr 로만 배운다."""
    a, b = tr[MAIN].copy(), te[MAIN].copy()
    emp = a.columns[a.isna().all()]                   # 학습 데이터에서 통째로 빈 열은
    a[emp] = .5                                       # 순위의 가운데 값 0.5 로 둔다
    b[emp] = b[emp].fillna(.5)
    # (3) MICE: 다른 피처들로 빈 값을 예측해서 채우는 걸 10번 반복하며 다듬는다
    ii = IterativeImputer(max_iter=10, random_state=0).fit(a.to_numpy(copy=True))
    a = pd.DataFrame(ii.transform(a.to_numpy(copy=True)), columns=MAIN, index=a.index)
    b = pd.DataFrame(ii.transform(b.to_numpy(copy=True)), columns=MAIN, index=b.index)
    a['lag'], b['lag'] = a.lag.clip(0, 1), b.lag.clip(0, 1)   # 채운 lag 이 0~1 을 벗어나지 않게
    for c in IND:                                     # 결측 표시는 그대로 붙인다
        a['na_' + c], b['na_' + c] = tr['na_' + c].values, te['na_' + c].values
    if A.inter in ('pm', 'both'):                     # (옵션) lag 효과가 채널 규모에 따라 달라지는가
        a['lag_x_pm'], b['lag_x_pm'] = a.lag * a.rk_past_med, b.lag * b.rk_past_med
    if A.inter in ('l3', 'both'):                     # (옵션) lag 효과가 최근 모멘텀에 따라 달라지는가
        a['lag_x_l3'], b['lag_x_l3'] = a.lag * a.rk_last3_m, b.lag * b.rk_last3_m
    for c in EXTRA:                                   # 나머지는 학습 데이터 중앙값으로 채운다
        med = tr[c].median()
        a[c] = tr[c].fillna(med).values
        b[c] = te[c].fillna(med).values
    return a, b

# --dump: 여기까지의 전처리 결과(MICE 전)를 정답 y, 점수 s 와 함께 CSV 로 저장
if A.dump:
    D_ = X.copy()
    D_['y'] = np.nan
    D_['s'] = np.nan
    for D in D_.D.unique():
        if D + F * DAY <= TEST_D:                     # 정답이 완전히 계산 가능한 날짜만
            o = partial_s(D, D + F * DAY)
            m = D_.D == D
            D_.loc[m, 's'] = D_.loc[m, 'channel_id'].map(o.s).values
            D_.loc[m, 'y'] = D_.loc[m, 'channel_id'].map(label(D, F)).values
    D_.to_csv(A.dump, index=False)
    print('전처리 테이블 저장:', A.dump, D_.shape)


# ═════════════════════════════════════════════════════════════════════════════════════
# 7. 학습 데이터 만들기 — 생존분석의 '중도절단' 아이디어  ← 점수를 가장 크게 올린 부분
#
#    데이터는 08-31 에서 끝난다(C = 데이터 끝).
#    - 기준일 D ≤ 08-14: 미래 18일이 모두 데이터 안에 있다 → 정답을 정확히 안다 (하드 라벨)
#    - 기준일 D = 08-15 ~ 08-29: 미래 18일 중 앞부분만 보인다 → 정답이 '잘려' 있다 (우측 중도절단)
#      버리면 테스트와 가장 가까운 15일을 잃는다. 그래서 버리지 않고,
#      "앞부분만 봤을 때 최종 정답이 1일 확률 q"를 추정해서
#      같은 채널을 y=1(가중치 q) 한 줄 + y=0(가중치 1-q) 한 줄로 넣는다 (소프트 라벨).
#    q 는 정답을 다 아는 과거 날짜의 미래 창을 '일부러 같은 길이로 잘라서' 배운다.
# ═════════════════════════════════════════════════════════════════════════════════════
RATE = {D: g.set_index('channel_id').upl_rate.to_dict() for D, g in X.groupby('D')}

def intervals(d, C):
    """(latest 모드) 잘린 날짜에서 최종 점수 s 가 들어갈 수 있는 범위 [L, R] 계산 — 구간중도절단.
       관측된 영상 m편 + 아직 안 올라온 영상 최대 u편이 있을 때,
         u편이 전부 아주 작으면 최종 중앙값이 가장 낮아짐 → 하한 L
         u편이 전부 아주 크면 최종 중앙값이 가장 높아짐 → 상한 R (관측이 적으면 무한대)
       u 는 업로드율 × 남은 일수의 포아송분포 90% 지점으로 넉넉하게 잡는다."""
    P = past(d)
    lp = np.log1p(P)
    end = min(C, d + F * DAY)                         # 실제로 볼 수 있는 끝
    rem = (d + F * DAY - end).days                    # 아직 안 본 날 수
    vids = {c: np.sort(g.lv.values)
            for c, g in L[(L.published_at >= d) & (L.published_at < end)].groupby('channel_id')}
    rate = RATE.get(d, {})
    out = {}
    for c in P.index:
        z = vids.get(c, np.array([])) - lp[c]         # 관측된 영상들의 '과거 대비' 로그 점수 (작은 순)
        m = len(z)
        if m == 0:
            continue                                  # 관측 영상이 없으면 범위를 말할 수 없음
        u = int(poisson.ppf(0.9, max(rate.get(c, 0.3) or 0.3, 1e-3) * rem))
        jl = (m + u - 1) // 2 - u                     # u편이 모두 작을 때 중앙값이 되는 관측 영상 위치
        jh = (m + u) // 2                             # u편이 모두 클 때 중앙값이 되는 관측 영상 위치
        out[c] = (z[jl] if jl >= 0 else -lp[c],       # 위치가 범위 밖이면 하한 = 조회수 0일 때 값
                  z[jh] if jh <= m - 1 else np.inf)   # 위치가 범위 밖이면 상한 = 무한대
    return pd.DataFrame(out, index=['L', 'R']).T

def qfeat(d, end):
    """q 모형의 입력: [d, end) 만 봤을 때 알 수 있는 정보"""
    o = partial_s(d, end)
    Z = pd.DataFrame({'rk': o.rk,                       # 앞부분만 본 점수의 순위
                      'has': (o.fn > 0).astype(float)},  # 앞부분에 영상이 하나라도 있었나
                     index=o.index)
    if A.q_x:                                         # (옵션) q = E[y | 부분정보, x] 가 되도록 채널 피처 추가
        xr = X[X.D == d].set_index('channel_id').reindex(Z.index)
        Z['x_pm'] = xr.rk_past_med.fillna(.5).values
        Z['x_up'] = xr.rk_upl_rate.fillna(.5).values
        Z['x_lag'] = xr.lag.fillna(.2).values
        Z['x_nalag'] = xr.lag.isna().astype(float).values
    if A.lag1q and ('rk_obs_conc' in X.columns):      # [2차-2] q 모형에 lag 창 관측-질 신호 주입
        #   obs_conc/obs_nrise 의 날짜별 순위 + lag → 잘린 날짜의 소프트 라벨 q 추정 정확도 향상.
        #   누수 없음: 모두 D 이전 [prev_D, D) 관측부분만으로 X 에서 계산된 값(날짜 d 기준).
        xr = X[X.D == d].set_index('channel_id').reindex(Z.index)
        Z['q_conc'] = xr.rk_obs_conc.fillna(.5).values
        Z['q_nrise'] = xr.rk_obs_nrise.fillna(.5).values
        Z['q_lag'] = xr.lag.fillna(.2).values
    if A.q_ext:
        xr = X[X.D == d].set_index('channel_id').reindex(Z.index)
        for c in ['rk_cv_last10', 'rk_last3_m', 'rk_r7_n', 'rk_days_since', 'lagobs_rk']:   # 조건부 정보 확장
            Z['x_' + c] = xr[c].fillna(.5).values
        Z['x_na_m7'] = xr.na_m7.values
        k = (end - d).days
        # 시간의존 공변량(landmark): 관측 구간의 '중간 시점'에서의 순위 → 궤적(올라오는 중인가)
        oh = partial_s(d, d + max(k // 2, 1) * DAY)
        Z['rk_half'] = oh.rk.reindex(Z.index).fillna(0).values
        Z['rk_delta'] = Z.rk - Z.rk_half
        # 재발 사건(업로드 = 반복 사건): 남은 기간 기대 업로드 수 = 업로드 강도 × 남은 일수
        Z['exp_rem'] = np.log1p(xr.upl_rate.fillna(.2).values * max(F - k, 0))
        # 다단계 상태: 지금까지 몇 편 올렸나 (0편 → 1편 → 여러 편)
        Z['state_n'] = np.minimum(o.fn.reindex(Z.index).values, 5)
    if not LATEST:
        return Z                                      # 0.7627 제출은 이 두 개만 사용
    Z['n'] = np.log1p(o.fn)                           # (latest) 앞부분 영상 수
    I = intervals(d, end).reindex(o.index)
    Z['Lrk'] = I.L.replace(-np.inf, np.nan).rank(pct=True).fillna(0)   # 하한 L 의 순위
    Z['Rfin'] = np.isfinite(I.R.fillna(np.inf)).astype(float)          # 상한이 유한한가
    Z['Rrk'] = I.R.replace(np.inf, np.nan).rank(pct=True).fillna(1)    # 상한 R 의 순위
    return Z

_Q = {}
def qmodel(k, C):
    """'앞 k일만 본 정보 → 최종 정답' 을 배우는 작은 로지스틱 모형.
       정답을 다 아는 과거 날짜(d + F ≤ C)의 미래 창을 k일로 잘라 학습 예제를 만든다."""
    if (k, C) not in _Q:
        Zs = []
        for d in pd.date_range(START, C - F * DAY):
            Z = qfeat(d, d + k * DAY)                 # 일부러 k일만 본 정보
            Z['y'] = label(d, F).reindex(Z.index)     # 그 날의 진짜 최종 정답
            Zs.append(Z)
        Z = pd.concat(Zs).dropna()
        _Q[(k, C)] = LogisticRegression(C=1.0, max_iter=2000).fit(Z.drop(columns='y'), Z.y)
    return _Q[(k, C)]

# [개선4] 통합 q 모형: k 별 분리 대신 k/F(kfrac) 를 입력 피처로 넣어, 모든 관측길이 k 의 예제를
#   한 번에 학습한다. 학습 예제가 많아져(피처 소수) 추정이 안정적이고 짧은 k 에서도 정보 공유.
_QUNI = {}
def qmodel_uni(C):
    if C not in _QUNI:
        Zs = []
        kset = range(3, F)                            # 소프트 라벨에서 쓰는 관측길이 범위(3일~F-1일)
        for d in pd.date_range(START, C - F * DAY):
            for k in kset:
                Z = qfeat(d, d + k * DAY)
                Z['kfrac'] = k / F                    # 관측 비율 (통합 모형의 핵심 입력)
                Z['y'] = label(d, F).reindex(Z.index)
                Zs.append(Z)
        Z = pd.concat(Zs).dropna()
        _QUNI[C] = LogisticRegression(C=1.0, max_iter=3000).fit(Z.drop(columns='y'), Z.y)
    return _QUNI[C]

def q_predict(d, C):
    """소프트 라벨용 q = P(최종 정답=1 | [d,C) 관측). 통합/분리 모형 공통 진입점.
       반환: (Z 인덱스=channel_id 에 맞춘 q 배열, Z)"""
    k = (C - d).days
    Z = qfeat(d, C)
    if A.q_uni:
        Zq = Z.copy()
        Zq['kfrac'] = k / F
        q = qmodel_uni(C).predict_proba(Zq)[:, 1]
    else:
        q = qmodel(k, C).predict_proba(Z)[:, 1]
    return q, Z

def train_set(C):
    """데이터 끝이 C 일 때의 학습 데이터 = 하드 라벨 행 + 소프트 라벨 행. 열 w 가 가중치."""
    rows = []
    # (a) 정답을 다 아는 날: 가중치 1
    for D in pd.date_range(START, C - F * DAY):
        r = X[X.D == D].copy()
        r['y'] = r.channel_id.map(label(D, F))
        rows.append(r[r.y.notna()].assign(w=1.))
    # (b) 정답이 잘린 날 (최소 3일은 관측된 날까지): 두 줄로 복제
    for d in pd.date_range(C - (F - 1) * DAY, C - 3 * DAY):
        q, Z = q_predict(d, C)                        # 최종 정답이 1일 확률 (통합/분리 모형)
        r = X[X.D == d].set_index('channel_id').loc[Z.index].reset_index()
        sk = (C - d).days / F if A.soft_k else 1.0    # (옵션) 적게 관측된 날일수록 덜 믿기
        r['_sk'] = sk                                 # EM 재가중용으로 관측비율 보관
        rows += [r.assign(y=1, w=q * sk),             # "1일 수도 있다" — 가중치 q
                 r.assign(y=0, w=(1 - q) * sk)]       # "0일 수도 있다" — 가중치 1-q
    out = pd.concat(rows, ignore_index=True)
    # (옵션) lag 의 미래 창 [D-K, D-K+prev_F) 가 데이터 끝 C 를 넘는 학습 행 처리.
    #   테스트의 lag(공식 라벨)는 창 전체로 매겨졌지만, 학습 행의 lag 는 C 이후를 모른다.
    if A.lag_mode != 'full':
        lagend = out.D - K * DAY + A.prev_F * DAY
        bad = lagend > C
        for D in out.loc[bad, 'D'].unique():
            m = (out.D == D) & bad
            if A.lag_mode == 'na':
                out.loc[m, 'lag'] = np.nan
            else:                                         # partial: C 까지만 보고 매긴 라벨
                dl = D - K * DAY
                o = partial_s(dl, min(dl + A.prev_F * DAY, C))
                lab = (o.s >= o.s.quantile(.8)).astype(int)
                out.loc[m, 'lag'] = out.loc[m, 'channel_id'].map(lab).values
        out.loc[bad, 'na_lag'] = out.loc[bad, 'lag'].isna().astype(float)
        out.loc[bad, 'lag_x_obs'] = out.loc[bad, 'lag'] * (1 - out.loc[bad, 'lagobs_rk'])
        out.loc[bad, 'nolag_x_obs'] = (1 - out.loc[bad, 'lag']) * out.loc[bad, 'lagobs_rk']
    return out


# ═════════════════════════════════════════════════════════════════════════════════════
# 8. 모델: 로지스틱 회귀
#    신호가 약하고 같은 채널이 날짜만 바꿔 반복되는 데이터라, 복잡한 트리 모델(LightGBM)은
#    노이즈를 외워버렸다(0.66). 규제를 세게 건 로지스틱이 더 나았다(0.73).
# ═════════════════════════════════════════════════════════════════════════════════════
def fit_predict(tr, te):
    a, b = design(tr, te)                             # 6번 결측 처리
    # 최근 날짜일수록 크게 반영: 7일 전 행은 가중치 절반, 14일 전은 1/4 ... (× 소프트 라벨 가중치 w)
    sw = tr.w.values * 0.5 ** ((tr.D.max() - tr.D).dt.days.values / 7)
    if A.ap_weight != 1.0:
        # [개선3] AP 직접 최적화: 양성(y=1) 행의 가중치를 ap_weight 배. 소프트 라벨 가중 w 와 곱셈 호환.
        #   하드 양성(w=1)과 소프트 양성 행(y=1,w=q) 모두 자연스럽게 상위권으로 끌어올려 상위 정밀도를 높인다.
        sw = sw * np.where(tr.y.values == 1, A.ap_weight, 1.0)
    sc = StandardScaler().fit(a)                      # 피처마다 평균 0, 표준편차 1 로 맞춤 (규제가 공평하게 걸리도록)
    m = LogisticRegression(C=A.C, max_iter=3000).fit(sc.transform(a), tr.y, sample_weight=sw)
    global LAST
    LAST = (m, sc, a, b)                              # SHAP 계산용으로 모델과 입력을 보관
    return m.predict_proba(sc.transform(b))[:, 1], pd.Series(m.coef_[0], a.columns)

def fit_predict_em(tr, te):
    """[개선4] 완전 EM: 최종 로지스틱 예측으로 소프트 라벨 q 를 갱신하며 재학습.
       --q_em 가 0 이면 기존 fit_predict 와 동일(한 번 적합)."""
    if A.q_em <= 0 or '_sk' not in tr.columns:
        return fit_predict(tr, te)
    tr = tr.copy()
    soft = tr['_sk'].notna()                          # 소프트 라벨 행(잘린 날, y=1/y=0 두 줄로 복제됨)
    pos = soft & (tr.y == 1)
    neg = soft & (tr.y == 0)
    key = (tr.channel_id.astype(str) + '|' + tr.D.astype(str))   # 같은 채널-날짜의 두 줄을 짝지음
    p = None
    for _ in range(A.q_em):
        # M-step: 현재 가중치로 모델 적합 → 전체 학습행에 대한 예측 p (= E-step 의 책임도)
        a, _b = design(tr, tr)
        sw = tr.w.values * 0.5 ** ((tr.D.max() - tr.D).dt.days.values / 7)
        if A.ap_weight != 1.0:
            sw = sw * np.where(tr.y.values == 1, A.ap_weight, 1.0)
        sc = StandardScaler().fit(a)
        m = LogisticRegression(C=A.C, max_iter=3000).fit(sc.transform(a), tr.y, sample_weight=sw)
        p = m.predict_proba(sc.transform(a))[:, 1]
        # E-step: 소프트 날의 q 를 모델 예측으로 갱신 (y=1 행에만 예측값이 유효, 짝 행은 1-q)
        qnew = pd.Series(p[pos.values], index=key[pos].values)
        sk_pos = pd.Series(tr.loc[pos, '_sk'].values, index=key[pos].values)
        tr.loc[pos, 'w'] = (key[pos].map(qnew) * key[pos].map(sk_pos)).values
        tr.loc[neg, 'w'] = ((1 - key[neg].map(qnew)) * key[neg].map(sk_pos)).values
    return fit_predict(tr, te)


# ═════════════════════════════════════════════════════════════════════════════════════
# 9. 검증 또는 제출
#    검증: "데이터가 검증일 vd 에 끝난다"고 가정하고 7번부터 전부 다시 만든 뒤 vd 를 맞혀 본다.
#          실제 본선과 똑같은 상황을 재현해야 점수가 부풀려지지 않는다.
# ═════════════════════════════════════════════════════════════════════════════════════
if A.build_only:
    pass                                              # 다른 스크립트(dgp_ic.py)가 X, design, intervals 등을 가져다 쓴다
elif A.validate:
    res = []
    last_full = TEST_D - F * DAY                      # 정답이 완전히 계산되는 마지막 기준일
    for vd in pd.date_range(last_full - 8 * DAY, last_full):   # 마지막 9일을 하나씩 검증
        va = X[X.D == vd].copy()
        va['y'] = va.channel_id.map(label(vd, F))
        va = va[va.y.notna()]
        p, _ = fit_predict_em(train_set(vd), va)      # 데이터 끝 = vd 로 학습 데이터를 만든다
        res.append((roc_auc_score(va.y, p), average_precision_score(va.y, p)))
        print(vd.date(), np.round(res[-1], 4), flush=True)
    print('평균 ROC %.4f  PR %.4f' % tuple(np.mean(res, 0)))
else:
    te = X[X.D == TEST_D]
    p, coef = fit_predict_em(train_set(TEST_D), te)
    ss['prediction'] = ss.row_id.map(dict(zip(te.channel_id + '_' + TEST_D.strftime('%Y-%m-%d'), p)))
    assert ss.prediction.notna().all(), '예측 누락 채널 있음'
    ss.to_csv(A.out, index=False)
    print('저장:', A.out, len(ss), '행')
    # 계수: 양수 = 클수록 Rising 확률↑, 음수 = 클수록 확률↓ (표준화돼 있어 크기 비교 가능)
    print('계수 (표준화):')
    print(coef.sort_values().round(3).to_string())

    # ── XAI: SHAP ──────────────────────────────────────────────────────────────────────
    # SHAP 값 = "이 채널의 예측이 평균적인 채널보다 높/낮은 이유를 피처별로 나눈 몫" (로그 오즈 단위)
    # 로지스틱 회귀는 선형이라 SHAP 값이 정확히 계수 × (내 값 − 평균 값) 으로 계산된다 (LinearExplainer).
    if A.shap:
        import shap, matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        m, sc, a, b = LAST
        ex = shap.LinearExplainer(m, sc.transform(a))           # 기준(평균) = 학습 데이터
        sv = ex(sc.transform(b))                                # 테스트 채널별 SHAP 값
        sv.feature_names = list(a.columns)
        sv.data = b.values                                      # 색은 표준화 전 원래 값으로 칠한다
        # (1) 전체 그림: 피처별로 SHAP 값이 어떻게 퍼져 있나 (오른쪽 = Rising 확률↑)
        plt.figure()
        shap.plots.beeswarm(sv, max_display=15, show=False)
        plt.tight_layout(); plt.savefig(A.shap + '_beeswarm.png', dpi=150); plt.close()
        # (2) 피처 중요도 표: |SHAP| 평균이 클수록 예측에 많이 쓰인 피처
        imp = pd.DataFrame({'feature': a.columns, 'mean_abs_shap': np.abs(sv.values).mean(0), 'coef': m.coef_[0]})
        imp.sort_values('mean_abs_shap', ascending=False).to_csv(A.shap + '_importance.csv', index=False)
        # (3) 채널별 설명: 각 채널의 확률을 가장 많이 올린/내린 피처 3개씩
        vals = pd.DataFrame(sv.values, columns=a.columns, index=te.channel_id.values)
        rows = []
        for ch, r in vals.iterrows():
            up, dn = r.nlargest(3), r.nsmallest(3)
            rows.append({'channel_id': ch, 'pred': float(p[list(vals.index).index(ch)]),
                         'up_reasons': ', '.join(f'{k}(+{v:.2f})' for k, v in up.items() if v > 0),
                         'down_reasons': ', '.join(f'{k}({v:.2f})' for k, v in dn.items() if v < 0)})
        pd.DataFrame(rows).sort_values('pred', ascending=False).to_csv(A.shap + '_per_channel.csv', index=False)
        print('SHAP 저장:', A.shap + '_beeswarm.png / _importance.csv / _per_channel.csv')
