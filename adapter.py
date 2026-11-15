#!/usr/bin/env python3
from pathlib import Path

HERE = Path(__file__).resolve().parent


OPENROUTER_API_KEY = ""
OPENROUTER_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT = 180
TRANSPORT_RETRIES = 4

MODELS = [
    # "google/gemini-4-hydrogen",
    # "openai/gpt-6.1-terra-pro",
    "anthropic/claude-sonnet-5.5",
    # "x-ai/grok-4.5",
    # "typesafe/jev-router",
    # "~deepseek/deepseek-pro-latest"

]
CONDITIONS = ["A"]
SESSIONS = 5

LEAKAGE = False
LEAKAGE_MODELS = ["google/gemini-4-hydrogen"]
LEAKAGE_SESSIONS = 1
PILOT_UNAIDED_ACCURACY = 0.61

TEMPERATURE = 1.0
MAX_TOKENS = 4096

PARALLEL = 5
RESULTS_DIR = HERE
SAVE_TRANSCRIPTS = True
RUN_VERIFY = True
RUN_CHECK_PILOT = True

HUMAN_RAIR, HUMAN_RSR = 0.467, 0.462


import argparse
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional

import requests


def tag_for(model: str, condition: str) -> str:
    return f"{model.replace('/', '_')}_{condition}"


_max_tokens = MAX_TOKENS
_calls_log: Optional[Path] = None
_last_data: Optional[dict] = None

_transcript_dir: Optional[Path] = None
_session_id: Optional[str] = None
_round: Optional[int] = None


def _to_openai(messages: List[dict]) -> List[dict]:
    out = []
    for m in messages:
        if m["role"] == "user":
            content = [
                {"type": "text", "text": c["text"]} if c["type"] == "text" else
                {"type": "image_url", "image_url": {"url": f"data:{c['media_type']};base64,{c['data']}"}}
                for c in m["content"]
            ]
        else:
            content = "\n".join(c["text"] for c in m["content"] if c["type"] == "text")
        out.append({"role": m["role"], "content": content})
    return out


def _log_call(model: str, data: dict) -> None:
    if _calls_log is None:
        return
    choice = (data.get("choices") or [{}])[0]
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "model": model,
             "served_by": data.get("provider"), "finish_reason": choice.get("finish_reason"),
             "usage": data.get("usage")}
    with _calls_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def call_openrouter(messages: List[dict], model: str, max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None) -> str:
    global _last_data
    _last_data = None
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("__"):
        raise RuntimeError("OPENROUTER_API_KEY is not set in the SETTINGS block of adapter.py")
    payload = {"model": model, "messages": _to_openai(messages),
               "max_tokens": max_tokens or _max_tokens, "usage": {"include": True}}
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    url = f"{OPENROUTER_URL.rstrip('/')}/chat/completions"

    for attempt in range(1, TRANSPORT_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            if attempt == TRANSPORT_RETRIES:
                raise
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in (408, 429, 500, 502, 503, 504) and attempt < TRANSPORT_RETRIES:
            time.sleep(2 ** attempt)
            continue
        if not resp.ok:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        _last_data = data
        _log_call(model, data)
        return data["choices"][0]["message"].get("content") or ""
    raise RuntimeError("unreachable")


def _last_user_text(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return "\n".join(c["text"] for c in m["content"] if c["type"] == "text")
    return ""


def _save_transcript(messages: List[dict], model: str, temperature: Optional[float],
                     reply: Optional[str], error: Optional[str] = None) -> None:
    global _round
    if _transcript_dir is None or _session_id is None:
        return
    prompt = _last_user_text(messages)
    found = re.search(r"Round (\d+) of 15", prompt)
    if found:
        _round, stage = int(found.group(1)), 1
    else:
        stage = 2 if "ALGORITHMIC INSIGHT" in prompt else 0
    data = _last_data or {}
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(), "session_id": _session_id,
        "round": _round, "stage": stage, "model": model, "temperature": temperature,
        "served_by": data.get("provider"), "finish_reason": choice.get("finish_reason"),
        "prompt": prompt, "reply": reply, "reasoning": message.get("reasoning"), "error": error,
    }
    name = _session_id.replace("/", "_")
    with (_transcript_dir / f"{name}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if error:
        answer = f"[error: {error}]"
    elif not (reply or "").strip():
        answer = f"[empty reply, finish: {entry['finish_reason']}]"
    else:
        answer = reply.strip()
    with (_transcript_dir / f"{name}.md").open("a", encoding="utf-8") as f:
        f.write(f"Q: {prompt.strip()}\nA: {answer}\n\n")


def logged_call(messages: List[dict], model: str, max_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> str:
    try:
        reply = call_openrouter(messages, model, max_tokens=max_tokens, temperature=temperature)
    except Exception as exc:
        _save_transcript(messages, model, temperature, None, error=f"{type(exc).__name__}: {exc}")
        raise
    _save_transcript(messages, model, temperature, reply)
    return reply


def worker(model: str, condition: str, sessions: int, max_tokens: int, results_dir: Path,
           temperature: Optional[float], stage1_only: bool) -> None:
    global _max_tokens, _calls_log, _transcript_dir
    _max_tokens = max_tokens
    label = "S1" if stage1_only else condition
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    _calls_log = results_dir / "logs" / f"calls_{tag_for(model, label)}.jsonl"

    import run

    adapter = call_openrouter
    if SAVE_TRANSCRIPTS:
        _transcript_dir = results_dir / "transcripts" / tag_for(model, label)
        _transcript_dir.mkdir(parents=True, exist_ok=True)
        adapter = logged_call
        original_run_session = run.run_session

        def run_session(sc, model_, adapter_, condition_, session_id, *args, **kwargs):
            global _session_id, _round
            _session_id, _round = session_id, None
            return original_run_session(sc, model_, adapter_, condition_, session_id, *args, **kwargs)

        run.run_session = run_session

    run.ADAPTERS["openrouter"] = adapter
    out = results_dir / f"results_{tag_for(model, label)}.csv"
    sys.argv = ["run.py", "--provider", "openrouter", "--model", model,
                "--condition", condition, "--sessions", str(sessions), "--out", str(out)]
    if temperature is not None:
        sys.argv += ["--temperature", str(temperature)]
    if stage1_only:
        sys.argv += ["--stage1-only", "--assets", str(HERE / "assets_masked")]
    run.main()


def run_jobs(models: List[str], conditions: List[str], sessions: int, max_tokens: int,
             parallel: int, results_dir: Path, temperature: Optional[float],
             stage1_only: bool) -> None:
    jobs = [(m, c) for m in models for c in conditions]
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    with (results_dir / "run_meta.jsonl").open("a", encoding="utf-8") as f:
        for m, c in jobs:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "model": m,
                                "condition": "S1" if stage1_only else c, "sessions": sessions,
                                "max_tokens": max_tokens,
                                "temperature": temperature if temperature is not None else "provider default",
                                "assets": "assets_masked" if stage1_only else "assets",
                                "via": OPENROUTER_URL}) + "\n")

    def one(job):
        m, c = job
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
               "--models", m, "--conditions", c, "--sessions", str(sessions),
               "--max-tokens", str(max_tokens), "--results-dir", str(results_dir),
               "--temperature", "default" if temperature is None else str(temperature)]
        if stage1_only:
            cmd += ["--leakage"]
        c = "S1" if stage1_only else c
        if parallel == 1:
            print(f"\n=== {m}  condition {c}", flush=True)
            return job, subprocess.run(cmd).returncode
        log_path = results_dir / "logs" / f"{tag_for(m, c)}.log"
        print(f"started {m} {c} -> {log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            return job, subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for (m, c), rc in pool.map(one, jobs):
            c = "S1" if stage1_only else c
            if rc != 0:
                print(f"! {m} {c} exited with code {rc}", file=sys.stderr, flush=True)
            elif parallel > 1:
                print(f"finished {m} {c}", flush=True)


SUMMARY_KEYS = ["n_trials", "n_conflict", "RAIR", "RSR", "discrimination_index",
                "accuracy_initial", "accuracy_final", "agreement_fraction",
                "long_share_initial", "long_share_final", "long_share_advice", "long_share_market",
                "ppt_n", "ppt_switch_when_right", "ppt_switch_when_wrong", "ppt_t", "ppt_p", "ppt_dz"]


def _call_stats(results_dir: Path, tag: str):
    path = results_dir / "logs" / f"calls_{tag}.jsonl"
    if not path.exists():
        return float("nan"), 0
    total, truncated = 0.0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        call = json.loads(line)
        total += float((call.get("usage") or {}).get("cost") or 0)
        truncated += call.get("finish_reason") == "length"
    return total, truncated


def _temperatures(results_dir: Path, model: str, cond: str) -> List[str]:
    path = results_dir / "run_meta.jsonl"
    if not path.exists():
        return []
    seen = []
    for line in path.read_text(encoding="utf-8").splitlines():
        meta = json.loads(line)
        if meta.get("model") == model and meta.get("condition") == cond:
            t = str(meta.get("temperature"))
            if t not in seen:
                seen.append(t)
    return seen


def _independence(ok):
    import itertools
    piv = (ok[ok["round"] != 12]
           .pivot_table(index="round", columns="session_id", values="initial_choice", aggfunc="first"))
    sids = sorted(piv.columns)
    if len(sids) < 2:
        return None, float("nan")
    first = piv[[sids[0], sids[1]]].dropna()
    pair_shares = []
    for a, b in itertools.combinations(sids, 2):
        both = piv[[a, b]].dropna()
        if len(both):
            pair_shares.append((both[a] == both[b]).mean())
    same = int((first[sids[0]] == first[sids[1]]).sum())
    return (same, len(first)), (sum(pair_shares) / len(pair_shares) if pair_shares else float("nan"))


def _summarize_stage1(ok) -> dict:
    scored = ok[ok["round"] != 12]
    acc = scored["initial_correct"].mean() if len(scored) else float("nan")
    long_share = (scored["initial_choice"] == "Long").mean() if len(scored) else float("nan")
    print(f"  unaided accuracy on masked charts {acc:.3f} over {len(scored)} probes "
          f"(same model unmasked in the pilot: {PILOT_UNAIDED_ACCURACY}; humans 0.450)")
    print(f"  long share {long_share:.3f}  (market 0.357; always-Short scores 0.643)")
    print("  ~0.6 -> reads the chart, no leakage;  toward 0.5 -> it was identifying the instrument")
    return {"unaided_accuracy_masked": acc, "long_share_initial": long_share, "n_trials": len(scored)}


def summarize(results_dir: Path, run_check_pilot: bool = True) -> None:
    import pandas as pd
    import metrics

    rows = []
    for f in sorted(results_dir.glob("results_*.csv")):
        df = pd.read_csv(f)
        if df.empty:
            continue
        model, cond = df["model"].dropna().iloc[0], df["condition"].dropna().iloc[0]
        tag = tag_for(model, cond)
        sessions = df["session_id"].nunique()
        bad = df[df["parse_status"] != "ok"]
        ok = df[df["parse_status"] == "ok"]
        r12 = ok[ok["round"] == 12]
        cost, truncated = _call_stats(results_dir, tag)
        row = {
            "model": model, "condition": cond, "sessions": sessions,
            "aborted": bad["session_id"].nunique(),
            "aborted_rate": bad["session_id"].nunique() / sessions if sessions else float("nan"),
            "abort_reasons": json.dumps(bad["parse_status"].value_counts().to_dict()),
            "truncated_calls": truncated,
            "attention_check_skip_rate": (r12["final_choice"] == "SKIP").mean() if len(r12) else float("nan"),
            "cost_usd": cost,
            "cost_per_session_usd": cost / sessions if sessions and not math.isnan(cost) else float("nan"),
        }
        print(f"\n--- {model}  condition {cond}: {sessions} sessions, "
              f"{row['aborted']} aborted ({row['aborted_rate']:.1%}) {row['abort_reasons']}, "
              f"{truncated} calls cut off by max_tokens, "
              f"cost ${cost:.4f}")
        temps = _temperatures(results_dir, model, cond)
        row["temperature"] = ",".join(temps)
        if len(temps) > 1:
            print(f"  [!] this file mixes runs with different temperatures: {temps}")

        if cond == "S1":
            if len(ok):
                row.update(_summarize_stage1(ok))
            rows.append(row)
            continue

        first_pair, mean_pair = _independence(ok)
        row["s0000_s0001_same_rounds"] = f"{first_pair[0]}/{first_pair[1]}" if first_pair else ""
        row["mean_pairwise_same_share"] = mean_pair
        if first_pair:
            flag = "  [!] identical - temperature had no effect" if first_pair[0] == first_pair[1] else ""
            print(f"  check 1: s0000 vs s0001 initial_choice same on {first_pair[0]}/{first_pair[1]} "
                  f"rounds; all session pairs {mean_pair:.0%}{flag}")
        if row["aborted"]:
            kept = ok[ok["session_id"].isin(bad["session_id"])].groupby("session_id")["round"].nunique()
            kept = kept.reindex(bad["session_id"].unique(), fill_value=0)
            print(f"  check 2: aborted sessions kept {kept.min()}-{kept.max()} completed rounds each")

        if len(ok):
            try:
                m = metrics.reliance(metrics.prepare(ok))
                metrics.report(m)
                row.update({k: m.get(k) for k in SUMMARY_KEYS})
                row["RAIR_plus_RSR"] = m["RAIR"] + m["RSR"]
            except Exception as exc:
                print(f"  metrics not computable yet: {type(exc).__name__}: {exc}")
        if run_check_pilot:
            subprocess.run([sys.executable, "check_pilot.py", str(f)], cwd=HERE)
        rows.append(row)

    if not rows:
        print(f"no results_*.csv in {results_dir}")
        return
    out = results_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nHuman reading of the same instrument in this protocol: RAIR {HUMAN_RAIR} + "
          f"RSR {HUMAN_RSR} = {HUMAN_RAIR + HUMAN_RSR:.3f} (no-discrimination line = 1.000).")
    print(f"summary -> {out}")


def _temperature_arg(value: str) -> Optional[float]:
    return None if value.lower() in ("default", "none") else float(value)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None, help="override MODELS / LEAKAGE_MODELS")
    p.add_argument("--conditions", nargs="+", choices=["A", "B"], default=CONDITIONS)
    p.add_argument("--sessions", type=int, default=None, help="override SESSIONS / LEAKAGE_SESSIONS")
    p.add_argument("--temperature", type=_temperature_arg, default=TEMPERATURE,
                   help="override TEMPERATURE ('default' = provider default)")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--parallel", type=int, default=PARALLEL)
    p.add_argument("--leakage", action="store_true", default=LEAKAGE,
                   help="run 2: stage 1 only, on assets_masked/ (results_<model>_S1.csv)")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--metrics-only", action="store_true", help="skip the run, only recompute the summary")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    results_dir = args.results_dir.resolve()
    models = args.models or (LEAKAGE_MODELS if args.leakage else MODELS)
    sessions = args.sessions or (LEAKAGE_SESSIONS if args.leakage else SESSIONS)

    if args.worker:
        worker(models[0], args.conditions[0], sessions, args.max_tokens, results_dir,
               args.temperature, args.leakage)
        return 0

    if not args.metrics_only:
        if not (HERE / "assets").is_dir():
            print("assets/ missing, running fetch_assets.py (downloads from Zenodo) ...")
            if subprocess.run([sys.executable, "fetch_assets.py"], cwd=HERE).returncode != 0:
                return 1
        if RUN_VERIFY:
            print("verify.py - the human figures must reproduce before anything runs")
            if subprocess.run([sys.executable, "verify.py"], cwd=HERE).returncode != 0:
                sys.exit("verify.py failed: the environment or data are wrong, not the run. Stopping.")
        if args.leakage and not (HERE / "assets_masked").is_dir():
            subprocess.run([sys.executable, "mask_tickers.py"], cwd=HERE)
            sys.exit("Masked charts created. Look at assets_masked/1_start.PNG, then rerun.")
        results_dir.mkdir(parents=True, exist_ok=True)
        run_jobs(models, args.conditions, sessions, args.max_tokens,
                 max(1, args.parallel), results_dir, args.temperature, args.leakage)

    summarize(results_dir, run_check_pilot=RUN_CHECK_PILOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
