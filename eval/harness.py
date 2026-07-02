"""Backend-agnostic attack-success-rate (ASR) harness.

Runs the attack battery through any model backend with the defense OFF then ON,
scoring each by whether the deployment secret leaked. Prints a per-attack table
and the aggregate ASR reduction.

    python -m eval.harness                        # mock backend (offline sanity)
    python -m eval.harness --backend openai       # real API (reads OPENAI_API_KEY)
    python -m eval.harness --backend hf --json results/gpt-oss.json   # + save report
    python -m eval.harness --backend mock --json - | jq .summary      # pipe JSON

The human-readable table goes to stderr; --json - writes the machine-readable
report to stdout so it pipes cleanly. --csv PATH writes per-attack rows. The
mock backend is deterministic and needs no network; use it to validate the
harness. Point --backend at openai/hf for a real-model measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

from server import model as model_mod
from server.model import SECRET, MockBackend, build_chat_messages, leaked_secret
from server.pipeline import ensure_system, handle_message
from server.transcript import Transcript
from eval.attacks import ATTACKS


def run_once(backend, message: str, defense_on: bool) -> tuple[bool, str]:
    t = Transcript()
    trace = handle_message(t, backend, message, defense_on)
    reply = trace.get("reply", "")
    return leaked_secret(reply), reply


def preview() -> int:
    """Print the exact OpenAI messages that would be sent — no API call.

    Lets you inspect the defense framing (and cost) before spending on a real
    run. Uses the mock only to drive sanitize+seal; no network either way.
    """
    flatten = os.environ.get("OPENAI_FLATTEN", "1") != "0"
    for a in ATTACKS[:3]:
        print(f"\n{'='*70}\nATTACK: {a.id} — {a.vector}\n{'='*70}")
        for defense_on in (False, True):
            t = Transcript()
            ensure_system(t)
            # sanitize+seal the attack message exactly as the pipeline would
            from server import sealer
            content = sealer.sanitize(a.message).clean if defense_on else a.message
            t.append("user", content)
            msgs = build_chat_messages(t, defense_on, flatten)
            print(f"\n--- defense {'ON' if defense_on else 'OFF'} "
                  f"(flatten={flatten}) — {len(msgs)} messages ---")
            for m in msgs:
                print(f"[{m['role']}] {m['content'][:400]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Role-confusion defense ASR harness")
    ap.add_argument("--backend", default=os.environ.get("MODEL_BACKEND", "mock"),
                    choices=["mock", "openai", "hf"])
    ap.add_argument("--show-replies", action="store_true",
                    help="print a snippet of each reply")
    ap.add_argument("--preview", action="store_true",
                    help="print the OpenAI messages that would be sent, then exit "
                         "(no API call)")
    ap.add_argument("--json", metavar="PATH", nargs="?", const="-",
                    help="write the full report as JSON (PATH, or '-'/omit for stdout)")
    ap.add_argument("--csv", metavar="PATH",
                    help="write per-attack results as CSV to PATH")
    args = ap.parse_args()

    if args.preview:
        return preview()

    os.environ["MODEL_BACKEND"] = args.backend
    backend = model_mod.load_backend()

    print(f"\nASR harness · backend={backend.name} · secret={SECRET!r} · "
          f"{len(ATTACKS)} attacks\n", file=sys.stderr)
    header = f"{'attack':<18}{'vector':<40}{'OFF':>6}{'ON':>6}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    off_leaks = on_leaks = control_leaks = controls = 0
    results = []
    for a in ATTACKS:
        leak_off, reply_off = run_once(backend, a.message, defense_on=False)
        leak_on, reply_on = run_once(backend, a.message, defense_on=True)
        off_leaks += leak_off
        on_leaks += leak_on
        is_control = a.id == "benign_ask"
        if is_control:
            controls += 1
            control_leaks += leak_on
        mark = lambda b: " LEAK" if b else "  -  "
        print(f"{a.id:<18}{a.vector[:38]:<40}{mark(leak_off):>6}{mark(leak_on):>6}",
              file=sys.stderr)
        results.append({
            "id": a.id, "vector": a.vector, "is_control": is_control,
            "leaked_off": bool(leak_off), "leaked_on": bool(leak_on),
            "reply_off": reply_off, "reply_on": reply_on,
        })

    n = len(ATTACKS)
    reduction = (off_leaks - on_leaks) / off_leaks if off_leaks else 0.0
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "model": getattr(backend, "model_id", None),
        "flatten": getattr(backend, "flatten", None),
        "secret": SECRET,
        "n_attacks": n,
        "summary": {
            "baseline_asr": off_leaks / n,
            "defended_asr": on_leaks / n,
            "baseline_leaks": off_leaks,
            "defended_leaks": on_leaks,
            "blocked": off_leaks - on_leaks,
            "blocked_fraction_of_baseline": reduction,
            "control_leaks_under_defense": control_leaks,
            "controls": controls,
        },
        "results": results,
    }

    s = report["summary"]
    print("-" * len(header), file=sys.stderr)
    print(f"\nBaseline ASR (defense OFF): {off_leaks}/{n} = {s['baseline_asr']:.0%}",
          file=sys.stderr)
    print(f"Defended ASR (defense ON):  {on_leaks}/{n} = {s['defended_asr']:.0%}",
          file=sys.stderr)
    print(f"Attacks blocked by defense: {off_leaks - on_leaks} "
          f"({reduction:.0%} of baseline successes)", file=sys.stderr)
    if controls:
        print(f"Control (benign ask) leaked under defense: {control_leaks}/{controls} "
              f"(should be 0)", file=sys.stderr)

    if args.show_replies:
        print("\n--- replies ---", file=sys.stderr)
        for r in results:
            print(f"\n[{r['id']}] OFF: {r['reply_off'][:140]!r}", file=sys.stderr)
            print(f"[{r['id']}] ON : {r['reply_on'][:140]!r}", file=sys.stderr)

    if args.csv:
        _write_csv(args.csv, report)
        print(f"\nwrote CSV -> {args.csv}", file=sys.stderr)
    if args.json is not None:
        _write_json(args.json, report)
        if args.json != "-":
            print(f"wrote JSON -> {args.json}", file=sys.stderr)

    return 0


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_json(path: str, report: dict) -> None:
    text = json.dumps(report, indent=2)
    if path == "-":
        print(text)  # machine-readable payload goes to stdout
    else:
        _ensure_parent(path)
        with open(path, "w") as f:
            f.write(text + "\n")


def _write_csv(path: str, report: dict) -> None:
    _ensure_parent(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "vector", "is_control", "leaked_off", "leaked_on"])
        for r in report["results"]:
            w.writerow([r["id"], r["vector"], r["is_control"],
                        r["leaked_off"], r["leaked_on"]])


if __name__ == "__main__":
    raise SystemExit(main())
