import pandas as pd
from scipy import stats

REQUIRED = ["participant_id", "round", "initial_choice", "final_choice",
            "algo_dir", "outcome"]


def attention_check_failures(df, check_round=12, expected="SKIP"):
    key = "session_id" if "session_id" in df.columns else "participant_id"
    r = df[df["round"] == check_round]
    bad = r[r["final_choice"].astype(str).str.upper() != expected][key]
    return set(bad.unique())


def prepare(df, drop_attention_failures=False):
    d = df.copy()
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    if drop_attention_failures:
        key = "session_id" if "session_id" in d.columns else "participant_id"
        d = d[~d[key].isin(attention_check_failures(d))]

    if "parse_status" in d.columns:
        d = d[d["parse_status"].astype(str) == "ok"]

    d = d[d["algo_dir"].notna() & (d["algo_dir"] != "")]
    d["initial_correct"] = (d["initial_choice"] == d["outcome"]).astype(int)
    d["final_correct"] = (d["final_choice"] == d["outcome"]).astype(int)
    d["switched"] = (d["final_choice"] != d["initial_choice"]).astype(int)
    d["algo_agreed"] = d["algo_dir"] == d["initial_choice"]
    d["adherence"] = (d["final_choice"] == d["algo_dir"]).astype(int)
    return d


def reliance(df, long_label="Long"):
    d = df
    conf = d[~d["algo_agreed"]]

    pos_ai = int(((conf.initial_correct == 0) & (conf.switched == 1)).sum())
    neg_self = int(((conf.initial_correct == 0) & (conf.switched == 0)).sum())
    pos_self = int(((conf.initial_correct == 1) & (conf.switched == 0)).sum())
    neg_ai = int(((conf.initial_correct == 1) & (conf.switched == 1)).sum())

    rair = pos_ai / (pos_ai + neg_self) if (pos_ai + neg_self) else float("nan")
    rsr = pos_self / (pos_self + neg_ai) if (pos_self + neg_ai) else float("nan")

    out = {
        "n_trials": len(d),
        "n_participants": d.participant_id.nunique(),
        "n_conflict": len(conf),
        "pos_ai": pos_ai, "neg_self": neg_self,
        "pos_self": pos_self, "neg_ai": neg_ai,
        "RAIR": rair,
        "RSR": rsr,
        "discrimination_index": rair + rsr - 1,
        "switch_fraction": conf.switched.mean(),
        "accuracy_wid": conf.final_correct.mean(),
        "agreement_fraction": d.adherence.mean(),
        "accuracy_initial": d.initial_correct.mean(),
        "accuracy_final": d.final_correct.mean(),
        "long_share_initial": (d.initial_choice == long_label).mean(),
        "long_share_final": (d.final_choice == long_label).mean(),
        "long_share_advice": (d.algo_dir == long_label).mean(),
        "long_share_market": (d.outcome == long_label).mean(),
    }

    ppt = (conf.groupby(["participant_id", "initial_correct"])["switched"]
           .mean().unstack().dropna())
    if ppt.shape[1] == 2 and len(ppt) > 1:
        ppt.columns = ["when_wrong", "when_right"]
        t, p = stats.ttest_rel(ppt["when_right"], ppt["when_wrong"])
        diff = ppt["when_right"] - ppt["when_wrong"]
        out.update(ppt_n=len(ppt),
                   ppt_switch_when_right=ppt["when_right"].mean(),
                   ppt_switch_when_wrong=ppt["when_wrong"].mean(),
                   ppt_t=t, ppt_p=p, ppt_dz=diff.mean() / diff.std())
    return out


def report(m):
    print(f"  n = {m['n_trials']} trials, {m['n_participants']} agents, "
          f"{m['n_conflict']} conflict trials")
    print(f"  RAIR {m['RAIR']:.3f}   RSR {m['RSR']:.3f}   "
          f"sum {m['RAIR'] + m['RSR']:.3f}   index {m['discrimination_index']:+.3f}")
    print(f"  accuracy {m['accuracy_initial']:.3f} -> {m['accuracy_final']:.3f}   "
          f"agreement {m['agreement_fraction']:.3f}")
    print(f"  long share {m['long_share_initial']:.3f} -> {m['long_share_final']:.3f}"
          f"   (advice {m['long_share_advice']:.3f}, market {m['long_share_market']:.3f})")
    if "ppt_t" in m:
        print(f"  participant level (n={m['ppt_n']}): switched "
              f"{m['ppt_switch_when_right']:.3f} when right vs "
              f"{m['ppt_switch_when_wrong']:.3f} when wrong; "
              f"t={m['ppt_t']:.2f}, p={m['ppt_p']:.3f}, dz={m['ppt_dz']:.2f}")


def bootstrap_index(df, n_boot=2000, seed=42, level=0.95):
    import numpy as np

    key = "session_id" if "session_id" in df.columns else "participant_id"
    clusters = df[key].unique()
    groups = {c: g for c, g in df.groupby(key)}

    point = reliance(df)
    idx = point["discrimination_index"]

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([groups[c] for c in pick], ignore_index=True)
        conf = sample[~sample["algo_agreed"]]
        w, r = conf[conf.initial_correct == 0], conf[conf.initial_correct == 1]
        if len(w) == 0 or len(r) == 0:
            continue
        draws.append(w.switched.mean() - r.switched.mean())

    if len(draws) < n_boot * 0.5:
        return idx, float("nan"), float("nan"), len(clusters)
    a = (1 - level) / 2
    lo, hi = np.percentile(draws, [100 * a, 100 * (1 - a)])
    return idx, lo, hi, len(clusters)
