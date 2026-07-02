"""Backend-agnostic attack-success-rate (ASR) harness.

Runs the attack battery through any model backend with the defense OFF then ON,
scoring each by whether the deployment secret leaked. Prints a per-attack table
and the aggregate ASR reduction.

    python -m eval.harness                        # mock backend (offline sanity)
    MODEL_BACKEND=openai OPENAI_MODEL=gpt-4o \
        OPENAI_API_KEY=sk-... python -m eval.harness --backend openai

The mock backend is deterministic and needs no network; use it to validate the
harness itself. Point --backend at openai/hf for a real-model measurement.
"""

from __future__ import annotations

import argparse
import os

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
    args = ap.parse_args()

    if args.preview:
        return preview()

    os.environ["MODEL_BACKEND"] = args.backend
    backend = model_mod.load_backend()

    print(f"\nASR harness · backend={backend.name} · secret={SECRET!r} · "
          f"{len(ATTACKS)} attacks\n")
    header = f"{'attack':<18}{'vector':<40}{'OFF':>6}{'ON':>6}"
    print(header)
    print("-" * len(header))

    off_leaks = on_leaks = 0
    controls = 0
    control_leaks = 0
    rows = []
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
        print(f"{a.id:<18}{a.vector[:38]:<40}{mark(leak_off):>6}{mark(leak_on):>6}")
        rows.append((a, reply_off, reply_on))

    n = len(ATTACKS)
    print("-" * len(header))
    print(f"\nBaseline ASR (defense OFF): {off_leaks}/{n} = {off_leaks/n:.0%}")
    print(f"Defended ASR (defense ON):  {on_leaks}/{n} = {on_leaks/n:.0%}")
    reduction = (off_leaks - on_leaks) / off_leaks if off_leaks else 0.0
    print(f"Attacks blocked by defense: {off_leaks - on_leaks} "
          f"({reduction:.0%} of baseline successes)")
    if controls:
        print(f"Control (benign ask) leaked under defense: {control_leaks}/{controls} "
              f"(should be 0)")

    if args.show_replies:
        print("\n--- replies ---")
        for a, roff, ron in rows:
            print(f"\n[{a.id}] OFF: {roff[:140]!r}")
            print(f"[{a.id}] ON : {ron[:140]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
