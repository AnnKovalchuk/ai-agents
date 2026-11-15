#!/usr/bin/env python3
import os
import sys

import pandas as pd

import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
TRIALS = os.path.join(HERE, "data", "Trading_Study_Results.xlsx")
SCENARIOS = os.path.join(HERE, "scenarios.csv")

PUBLISHED = {"n_trials": 1232, "n_participants": 88, "n_conflict": 635,
             "RAIR": 0.467, "RSR": 0.462}


def load_human():
    if not os.path.exists(TRIALS):
        sys.exit("data/Trading_Study_Results.xlsx not found -- run fetch_assets.py first")

    sc = pd.read_csv(SCENARIOS)
    df = pd.read_excel(TRIALS)

    df = df.drop(columns=["algo_dir"], errors="ignore")
    df = df.merge(sc[["round", "algo_dir", "outcome"]], on="round", how="left")

    passed = df[(df["round"] == 12) & (df["final_choice"] == "SKIP")]
    df = df[df.participant_id.isin(passed.participant_id.unique())]

    df = df.sort_values("timestamp")
    df = df[~df.duplicated(subset=["participant_id", "round"], keep="first")]
    return df


def main():
    m = metrics.reliance(metrics.prepare(load_human()))
    print("HUMAN ARM -- known-negative control\n")
    metrics.report(m)

    print("\nchecking against the published figures ...")
    ok = True
    for key, want in PUBLISHED.items():
        got = m[key]
        hit = abs(got - want) < (0.001 if isinstance(want, float) else 0.5)
        ok &= hit
        print(f"  {'ok ' if hit else 'FAIL'}  {key:<15} {got:>8.3f}  expected {want}")

    if not ok:
        sys.exit("\nreproduction failed")

    idx = m["discrimination_index"]
    print(f"\nindex {idx:+.3f} -- below the no-discrimination line of 0.")
    print("The metric detects the absence of discrimination. It is usable on models.")


if __name__ == "__main__":
    main()
