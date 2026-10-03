# -*- coding: utf-8 -*-
"""
BEST(submission_best.csv, 본선 리더보드 0.7343) 이후 개선 후보 생성
  BEST    = T01 .50 + lr3 .25 + sim .25        (실측 0.7343)
  BEST_v2 = sim .80 + T01 .20                   (1순위: 두 검증이 엇갈리는 상황에서 가장 덜 흔들리는 지점)
  BEST_v3 = sim 단독                            (급등 국면 검증에서 가장 높은 단일 모델)
  BEST_v4 = sim .65 + T01 .35                   (BEST 와 v3 사이 지점)
  T01 = 본선제출/T01_ranklearn50_S01x50.csv (리더보드 0.7032), sim = final_run/sim.py 의 조건부 확률(09-01, S=8000, seed 0)
BEST 원본은 읽기만 하고 쓰지 않는다. 실행: python final_run/make_v2.py
"""
import sys, os, pickle, shutil, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "final_run"); sys.path.insert(0, "work")
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score as AP
r01 = lambda x: (rankdata(x) - 1) / (len(x) - 1)
OUT = "final_run/submissions"; SS = "data/5th-Datathon/data/epoch_data/sample_submission.csv"
CAND = {"BEST_v2.csv": 0.20, "BEST_v3.csv": 0.0, "BEST_v4.csv": 0.35}          # 값 = T01 비중 (v2 가 1순위)

def check(path):
    ref = pd.read_csv(SS); s = pd.read_csv(path); raw = open(path, "rb").read()
    c = dict(file=os.path.basename(path), rows=len(s), rows_ok=len(s) == len(ref), cols_ok=list(s.columns) == ["row_id", "prediction"], idset_ok=set(s.row_id) == set(ref.row_id),
             order_ok=bool((s.row_id.values == ref.row_id.values).all()), dup=int(s.row_id.duplicated().sum()), nan=int(s.prediction.isna().sum()), inf=int(np.isinf(s.prediction).sum()),
             min=float(s.prediction.min()), max=float(s.prediction.max()), in_range=bool(s.prediction.between(0, 1).all()), unique=int(s.prediction.nunique()), bom=raw[:3] == b"\xef\xbb\xbf", crlf=b"\r\n" in raw)
    assert c["rows_ok"] and c["cols_ok"] and c["idset_ok"] and c["dup"] == 0 and c["nan"] == 0 and c["inf"] == 0 and c["in_range"], c
    return c

if __name__ == "__main__":
    ss = pd.read_csv(SS)
    rd = lambda f: pd.read_csv(f).set_index("row_id").prediction.reindex(ss.row_id).values          # row_id 기준 정렬
    t01 = rd("본선제출/T01_ranklearn50_S01x50.csv"); best = rd("final_run/BEST_preserved/BEST_original_LB0.7343.csv")
    comp = pd.read_parquet("final_run/final_components.parquet").set_index("row_id").reindex(ss.row_id)
    assert comp.sim.notna().all()
    S = pd.read_parquet("final_run/sim_feats_final.parquet"); S = S[S.D == "2026-09-01"].set_index("row_id").reindex(ss.row_id)
    sim = comp.sim.values; tie = S.sim_g.fillna(S.sim_g.median()).values                               # 동률은 시뮬레이션의 평균 성장 점수로 정렬
    T = r01(t01)
    # BEST 재현 확인 (생성 방식이 기록과 같은지)
    re = 0.5 * T + 0.25 * comp.lr3.values + 0.25 * sim
    print("BEST 재현: 순위 상관 %.6f, 상위139 겹침 %d/139" % (spearmanr(re, best)[0], len(set(np.argsort(-re)[:139]) & set(np.argsort(-best)[:139]))))

    # ---- 검증 (같은 조건: 급등 국면 pseudo fold 업로드 채널 / 08-25 운영진 정답 폴드) ----
    pse = pickle.load(open("final_run/pseudo_cache.pkl", "rb")); fc = pickle.load(open("final_run/fold_components.pkl", "rb")); cv = fc["comp"]; y = fc["y"]
    def val(wt, lr=0.0):
        a = [AP(r["y"][r["up"]], (wt * r["C"]["base6"] + lr * r["C"]["lr3"] + (1 - wt - lr) * r["C"]["sim26"])[r["up"]]) for r in pse.values()]
        b = []
        for k in ["C", "D", "E"]:
            t = r01(0.5 * r01(cv[k]["base6"]) + 0.5 * r01(cv[k]["old"])); b.append(AP(y, wt * t + lr * r01(cv[k]["lr3"]) + (1 - wt - lr) * r01(cv[k]["sim"])))
        h = AP(y, wt * r01(cv["H(정직)"]["base6"]) + lr * r01(cv["H(정직)"]["lr3"]) + (1 - wt - lr) * r01(cv["H(정직)"]["sim"]))
        return a, b, h
    rows = []
    print("\n검증 PR-AUC   급등 국면(08-28 / 08-29 / 08-30 → 평균) | 08-25 폴드 C / D / E → 평균 | H(정직)")
    for name, (wt, lr) in {"T01형 (리더보드 0.7032)": (1.0, 0.0), "BEST형 (리더보드 0.7343)": (0.5, 0.25), "BEST_v4형 sim .65 + T01 .35": (0.35, 0.0),
                           "BEST_v2형 sim .80 + T01 .20": (0.20, 0.0), "BEST_v3형 sim 단독": (0.0, 0.0)}.items():
        a, b, h = val(wt, lr)
        rows.append(dict(model=name, surge_0828=a[0], surge_0829=a[1], surge_0830=a[2], surge_mean=np.mean(a), C=b[0], D=b[1], E=b[2], cde_mean=np.mean(b), H=h))
        print(f"  {name:30s} " + " / ".join(f"{x:.3f}" for x in a) + f" → {np.mean(a):.3f} | " + " / ".join(f"{x:.3f}" for x in b) + f" → {np.mean(b):.3f} | {h:.3f}")
    V = pd.DataFrame(rows); V["surge_vs_BEST"] = V.surge_mean - V.surge_mean.iloc[1]; V["cde_vs_BEST"] = V.cde_mean - V.cde_mean.iloc[1]
    V.to_csv("final_run/v2_validation.csv", index=False, encoding="utf-8-sig")

    # ---- 파일 생성 ----
    checks = []; top = lambda a, k: set(np.argsort(-a)[:k])
    print("\n후보 파일 (BEST 대비)")
    for fn, wt in CAND.items():
        score = (1 - wt) * sim + wt * T + 1e-9 * rankdata(tie, method="ordinal")
        rr = (rankdata(score, method="ordinal") - 1) / (len(score) - 1)
        sub = pd.read_csv(SS); sub["prediction"] = np.round(1 / (1 + np.exp(-8 * (rr - 0.8))), 8)       # 순위 보존 변환, 0/1 절단 없음
        path = os.path.join(OUT, fn); sub.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        shutil.copyfile(path, os.path.join("본선제출", fn)); c = check(path); p = sub.prediction.values
        c.update(t01_weight=wt, spearman_vs_BEST=spearmanr(p, best)[0], top20_vs_BEST=len(top(p, 20) & top(best, 20)), top50_vs_BEST=len(top(p, 50) & top(best, 50)), top139_vs_BEST=len(top(p, 139) & top(best, 139)))
        checks.append(c)
        print(f"  {fn}: T01 비중 {wt:.2f} | BEST 와 순위 상관 {c['spearman_vs_BEST']:.3f}, 상위20 겹침 {c['top20_vs_BEST']}, 상위50 {c['top50_vs_BEST']}, 상위139 {c['top139_vs_BEST']}")
    C = pd.DataFrame(checks); C.to_csv("final_run/v2_submission_checks.csv", index=False, encoding="utf-8-sig")
    print("\n형식 점검:"); print(C[["file", "rows", "rows_ok", "cols_ok", "idset_ok", "order_ok", "dup", "nan", "inf", "min", "max", "unique", "bom", "crlf"]].to_string(index=False))
    import hashlib
    print("BEST 원본 md5:", hashlib.md5(open("final_run/submissions/submission_best.csv", "rb").read()).hexdigest(), "(보존본과 동일:", hashlib.md5(open("final_run/BEST_preserved/BEST_original_LB0.7343.csv", "rb").read()).hexdigest() == hashlib.md5(open("final_run/submissions/submission_best.csv", "rb").read()).hexdigest(), ")")
