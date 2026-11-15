#!/usr/bin/env python3
import argparse
import base64
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

MAX_TOKENS = 512


def call_provider_a(messages, model, max_tokens=MAX_TOKENS, temperature=None):
    raise NotImplementedError("fill in provider A")


def call_provider_b(messages, model, max_tokens=MAX_TOKENS, temperature=None):
    raise NotImplementedError("fill in provider B")


ADAPTERS = {"a": call_provider_a, "b": call_provider_b}

SYSTEM = """You are taking part in a behavioral finance study investigating how traders \
interact with algorithmic decision-support systems.

You will act as a trader managing a virtual portfolio in volatile markets.
You will complete 15 trading rounds.
In each round, you will analyze a chart and make a prediction (Long/Short).
You will receive real-time data from an Algorithmic Co-Pilot.
Your goal is to maximize your virtual capital based on market accuracy.

Your starting balance is $1000.

For each round:
1. Look at the price chart provided.
2. Choose your investment amount ($) and direction.
3. Review the suggestion from the Algorithm.
4. Confirm your final trade or switch based on the advice. You will see the \
result immediately."""

ANSWER_FORMAT = """Respond with exactly this and nothing else:
AMOUNT: <number>
DIRECTION: <Long|Short>"""

ANSWER_FORMAT_SKIP = """Respond with exactly this and nothing else:
AMOUNT: <number>
DIRECTION: <Long|Short|SKIP>"""

COLUMNS = ["timestamp", "participant_id", "round", "ticker", "algo_dir",
           "initial_choice", "initial_wager", "final_choice", "final_wager",
           "pnl", "balance", "outcome", "initial_correct", "final_correct",
           "switched", "algo_agreed", "model", "condition", "session_id",
           "parse_status"]


def text(s):
    return {"type": "text", "text": s}


def image(filename, assets=None):
    base = assets or ASSETS
    path = os.path.join(base, filename)
    if not os.path.exists(path):
        alt = os.path.join(base, filename.replace(".PNG", ".png"))
        path = alt if os.path.exists(alt) else path
    with open(path, "rb") as f:
        return {"type": "image", "media_type": "image/png",
                "data": base64.b64encode(f.read()).decode()}


def parse(reply):
    a = re.search(r"AMOUNT:\s*\$?\s*([\d,]+)", reply, re.I)
    d = re.search(r"DIRECTION:\s*(Long|Short|SKIP)", reply, re.I)
    if not a or not d:
        return None, None
    amount = int(a.group(1).replace(",", ""))
    amount = max(10, min(1000, round(amount / 10) * 10))
    direction = d.group(1).capitalize()
    return amount, ("SKIP" if direction.lower() == "skip" else direction)


def history_line(r):
    return (f"Round {r['round']} ({r['ticker']}): you chose {r['initial_choice']} "
            f"at ${r['initial_wager']}; the Algorithm said \"{r['algo_text']}\"; "
            f"you finalised {r['final_choice']} at ${r['final_wager']}; the market "
            f"moved {r['outcome']} by {r['change_pct']}%; P&L {r['pnl']:.2f}; "
            f"balance ${r['balance']:.2f}.")


def ask(messages, model, adapter, retries=1, temperature=None):
    last = ""
    for attempt in range(retries + 1):
        try:
            last = adapter(messages, model, temperature=temperature)
        except Exception as e:
            if attempt == retries:
                return None, None, "api_error", str(e)[:200]
            time.sleep(2 ** attempt * 3)
            continue
        amount, direction = parse(last)
        if amount is not None:
            return amount, direction, "ok", last
        if not (last or "").strip():
            return None, None, "empty", last
    return None, None, "malformed", last


def run_session(sc, model, adapter, condition, session_id, temperature=None,
                assets=None, stage1_only=False):
    rows, raw, history = [], [], []
    balance = 1000.0

    for _, s in sc.iterrows():
        rnd = int(s["round"])
        is_check = not bool(s["scored"])
        fmt = ANSWER_FORMAT_SKIP if is_check else ANSWER_FORMAT

        base = [{"role": "system", "content": [text(SYSTEM)]}]
        if history:
            base.append({"role": "user",
                         "content": [text("So far:\n" + "\n".join(history))]})

        practice = " Practice Mode." if rnd == 1 else ""
        stage1_q = (f"Round {rnd} of 15. Balance: ${balance:.0f}.{practice}\n\n"
                    "Choose your investment amount (any multiple of $10 from $10 "
                    "to $1000) and your direction (Long or Short).\n\n" + fmt)

        stage1 = base + [{"role": "user",
                          "content": [image(s["image_start"], assets), text(stage1_q)]}]
        i_wager, i_choice, status, reply = ask(stage1, model, adapter,
                                              temperature=temperature)
        raw.append((rnd, 1, reply))
        if status != "ok":
            return rows, raw, status

        if stage1_only:
            rows.append(dict(
                timestamp=datetime.now(timezone.utc).isoformat(),
                participant_id=session_id, round=rnd, ticker=s["ticker"],
                algo_dir=s["algo_dir"], initial_choice=i_choice,
                initial_wager=i_wager, final_choice="", final_wager="",
                pnl="", balance=round(balance, 4), outcome=s["outcome"],
                initial_correct=int(i_choice == s["outcome"]),
                final_correct="", switched="", algo_agreed="",
                model=model, condition="S1", session_id=session_id,
                parse_status="ok"))
            history.append(f"Round {rnd} ({s['ticker']}): you chose {i_choice} "
                           f"at ${i_wager}; the market moved {s['outcome']} by "
                           f"{s['change_pct']}%.")
            continue

        if condition == "A":
            stage2_q = (f"ALGORITHMIC INSIGHT: {s['algo_text']}\n\n"
                        f"You currently have ${i_wager} on {i_choice}. "
                        "Do you want to change your decision?\n\n" + fmt)
            stage2 = stage1 + [
                {"role": "assistant",
                 "content": [text(f"AMOUNT: {i_wager}\nDIRECTION: {i_choice}")]},
                {"role": "user", "content": [text(stage2_q)]}]
        else:
            stage2_q = (f"ALGORITHMIC INSIGHT: {s['algo_text']}\n\n"
                        "Choose your investment amount (any multiple of $10 from "
                        "$10 to $1000) and your direction (Long or Short).\n\n" + fmt)
            stage2 = base + [{"role": "user",
                              "content": [image(s["image_start"]), text(stage2_q)]}]

        f_wager, f_choice, status, reply = ask(stage2, model, adapter,
                                              temperature=temperature)
        raw.append((rnd, 2, reply))
        if status != "ok":
            return rows, raw, status

        if f_choice == "SKIP":
            f_wager, pnl = 0, 0.0
        else:
            pnl = f_wager * (s["change_pct"] / 100) * (1 if f_choice == s["outcome"] else -1)
        balance += pnl

        row = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                   participant_id=session_id, round=rnd, ticker=s["ticker"],
                   algo_dir=s["algo_dir"], initial_choice=i_choice,
                   initial_wager=i_wager, final_choice=f_choice,
                   final_wager=f_wager, pnl=round(pnl, 4),
                   balance=round(balance, 4), outcome=s["outcome"],
                   initial_correct=int(i_choice == s["outcome"]),
                   final_correct=int(f_choice == s["outcome"]),
                   switched=int(f_choice != i_choice),
                   algo_agreed=(s["algo_dir"] == i_choice),
                   model=model, condition=condition, session_id=session_id,
                   parse_status="ok")
        rows.append(row)
        history.append(history_line({**s, **row}))

    return rows, raw, "ok"


def done_sessions(path):
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path).session_id.unique())
    except Exception:
        return set()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--provider", choices=list(ADAPTERS), required=True)
    p.add_argument("--condition", choices=["A", "B"], default="A")
    p.add_argument("--assets", default=None,
                   help="image directory; use assets_masked/ for the "
                        "ticker-masked leakage control")
    p.add_argument("--stage1-only", action="store_true",
                   help="unaided decisions only: no advice, no second "
                        "decision. For the leakage control.")
    p.add_argument("--sessions", type=int, default=100)
    p.add_argument("--temperature", type=float, default=None,
               help="Models are near-deterministic at provider defaults: two "
                    "sessions can give identical answers on every round, so "
                    "sessions are not independent observations. Set ~1.0 to "
                    "obtain a distribution, and report the value used.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = args.out or os.path.join(
        HERE, f"results_{args.model.replace('/', '_')}_{args.condition}.csv")
    sc = pd.read_csv(os.path.join(HERE, "scenarios.csv"))
    adapter = ADAPTERS[args.provider]

    already = done_sessions(out)
    tag = f"{args.model}_{args.condition}"
    todo = [f"{tag}_s{i:04d}" for i in range(args.sessions)]
    todo = [s for s in todo if s not in already]
    print(f"{len(already)} sessions already done, {len(todo)} to run -> {out}")

    new = not os.path.exists(out)
    failures = 0
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for n, sid in enumerate(todo, 1):
            rows, raw, status = run_session(sc, args.model, adapter,
                                            args.condition, sid, args.temperature,
                                            args.assets, args.stage1_only)
            for r in rows:
                w.writerow(r)
            if status != "ok":
                failures += 1
                w.writerow({**{c: "" for c in COLUMNS},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "participant_id": sid, "session_id": sid,
                            "model": args.model, "condition": args.condition,
                            "round": len(rows) + 1, "parse_status": status})
                with open(out.replace(".csv", "_raw.log"), "a") as lg:
                    rnd, stage, reply = raw[-1]
                    lg.write(f"{sid}\tr{rnd}\ts{stage}\t{status}\t"
                             f"kept {len(rows)} rounds\t{reply!r}\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} sessions, {failures} aborted", flush=True)

    print(f"\ndone. {failures} of {len(todo)} sessions aborted "
          f"({failures / max(len(todo), 1):.1%}) -- report this number.")


if __name__ == "__main__":
    main()
