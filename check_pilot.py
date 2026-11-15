#!/usr/bin/env python3
import sys

import pandas as pd

import metrics

HUMAN = {"initial_acc": 0.450, "final_acc": 0.419, "long_init": 0.574,
         "long_final": 0.682, "rair": 0.467, "rsr": 0.462}


def main(path):
    df = pd.read_csv(path)
    ok = True

    print(f"=== {path} ===\n")

    n_sess = df.session_id.nunique()
    per = df.groupby("session_id")["round"].nunique()
    print(f"sessions           {n_sess}")
    print(f"rounds per session {per.min()}–{per.max()} (expect 15)")
    if per.min() != 15 or per.max() != 15:
        print("  [!] incomplete sessions — aborted mid-run, or round 12 dropped too early")
        ok = False

    missing = [c for c in ("round", "ticker", "algo_dir", "initial_choice",
                           "initial_wager", "final_choice", "final_wager",
                           "pnl", "balance", "outcome") if c not in df.columns]
    if missing:
        print(f"  [!] missing columns: {missing}")
        ok = False

    if "parse_status" in df.columns:
        bad = (df.parse_status.astype(str) != "ok").mean()
        print(f"malformed rate     {bad:.1%}")
        if bad > 0.05:
            print("  [!] above 5% — tighten the format instruction before the full run")
    else:
        print("  [!] no parse_status column — non-response is not being recorded")
        ok = False

    w = pd.concat([df.initial_wager, df.final_wager]).dropna()
    w = pd.to_numeric(w, errors="coerce").dropna()
    print(f"stakes             {w.min():.0f}–{w.max():.0f}, "
          f"{(w % 10 == 0).mean():.0%} multiples of 10")
    if w.min() < 10 or w.max() > 1000:
        print("  [!] outside the $10–$1000 slider range")
        ok = False

    d = df[df["round"] != 12].dropna(subset=["final_choice"]).copy()
    print(f"\nscored trials      {len(d)}")

    m = metrics.reliance(metrics.prepare(df))
    print()
    print(f"accuracy before    {m['accuracy_initial']:.3f}   "
          f"(humans {HUMAN['initial_acc']:.3f})")
    print(f"accuracy after     {m['accuracy_final']:.3f}   "
          f"(humans {HUMAN['final_acc']:.3f})")
    print(f"long share before  {m['long_share_initial']:.3f}   "
          f"(humans {HUMAN['long_init']:.3f}, market 0.357, advice 0.786)")
    print(f"long share after   {m['long_share_final']:.3f}   "
          f"(humans {HUMAN['long_final']:.3f})")

    a = m["accuracy_initial"]
    print()
    if a > 0.60:
        print(f"  [!] unaided accuracy {a:.1%} is well above the human 45.0%.")
        print("      Run the ticker-masked control before trusting anything else.")
        ok = False
    elif a < 0.30:
        print(f"  [!] unaided accuracy {a:.1%} is far below chance — check that")
        print("      outcome and choice are not inverted somewhere.")
        ok = False
    else:
        print(f"  unaided accuracy {a:.1%} sits in the expected band. No sign of leakage.")

    ls = m["long_share_final"]
    if ls > 0.95 or ls < 0.05:
        print(f"  [!] the model answered one direction on {max(ls, 1-ls):.0%} of trials.")
        print("      Accuracy is meaningless here; report the direction share with it.")

    print(f"\nconflict trials    {m['n_conflict']}")
    if m["n_conflict"] < 150:
        print("  reliance index is not readable at this n — pilot checks the pipeline,")
        print("  not the finding. Do not interpret RAIR/RSR yet.")
    else:
        print(f"  RAIR {m['RAIR']:.3f}  RSR {m['RSR']:.3f}  "
              f"sum {m['RAIR']+m['RSR']:.3f}")

    print("\n" + ("PIPELINE OK — safe to scale up" if ok else
                  "FIX THE FLAGGED ITEMS BEFORE THE FULL RUN"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python check_pilot.py <results.csv>")
    sys.exit(main(sys.argv[1]))
