"""Run the ASR harness against a local OpenAI-compatible model server.

The measurement route for a workstation with a consumer GPU, where the
transformers ``hf`` backend is impractical (MXFP4/Triton is unreliable on native
Windows; recent consumer cards need newer CUDA than the pinned wheels target).
Instead, serve a non-hardened open model with any OpenAI-compatible local server
(KoboldCpp, llama.cpp's server, vLLM, LM Studio, Ollama, ...) and point the
existing ``openai`` backend at it. Same ``build_chat_messages`` framing, so the
numbers stay comparable to the OpenAI / HF backends.

If the machine runs a GPU-arbitration service (a lease API), this optionally
leases the GPU for the duration of the run. See ``eval/gpu_lease.py`` and
docs/LOCAL_GPU.md. Use ``--no-lease`` if there is no arbiter.

Flow:
    1. (optional) Acquire a GPU lease.
    2. Launch the model server with the configured command (or ``--no-launch``
       to attach to one you started yourself).
    3. Wait for the server, discover the served model id.
    4. Run ``python -m eval.harness --backend openai`` against it, forwarding any
       extra args (--json, --csv, --show-replies, ...).
    5. On exit (or Ctrl-C / error): stop the server we launched and release the
       lease.

Usage:
    python -m eval.local_gpu                                  # table to stderr
    python -m eval.local_gpu --json results/run.json --csv results/run.csv
    python -m eval.local_gpu --no-lease --no-launch           # attach to a running server

Configure the model server via env / .env (see docs/LOCAL_GPU.md):
    COT_MODEL_GGUF   path to the GGUF to serve (for the default KoboldCpp command)
    KOBOLDCPP_EXE    KoboldCpp executable (default: koboldcpp.exe on PATH)
    KOBOLD_CMD       full launch command line (overrides the two above)
    KOBOLD_URL       server base URL (default: http://localhost:5001)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# server/__init__ loads .env (GPU_LEASE_API_KEY, COT_MODEL_GGUF, ...) on import.
import server  # noqa: F401
from eval.gpu_lease import GpuLease, GpuLeaseError

DEFAULT_URL = "http://localhost:5001"


def split_cmd(cmd: str) -> list[str]:
    """Split a command line into argv, preserving Windows backslashes.

    ``posix=False`` keeps backslashes intact (posix=True would treat them as
    escapes) but leaves surrounding quotes in each token, so strip a matched
    pair afterwards.
    """
    tokens = shlex.split(cmd, posix=False)
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] == '"' else t for t in tokens]


def kobold_url() -> str:
    return os.environ.get("KOBOLD_URL", DEFAULT_URL).rstrip("/")


def build_launch_cmd() -> str:
    """Resolve the model-server launch command from env.

    KOBOLD_CMD wins; otherwise build a default KoboldCpp command from
    COT_MODEL_GGUF (+ KOBOLDCPP_EXE, KOBOLD_PORT, KOBOLD_DEVICE).
    """
    explicit = os.environ.get("KOBOLD_CMD")
    if explicit:
        return explicit
    model = os.environ.get("COT_MODEL_GGUF")
    if not model:
        raise SystemExit(
            "No model configured. Set COT_MODEL_GGUF to your GGUF path (and "
            "optionally KOBOLDCPP_EXE), or set KOBOLD_CMD to a full launch "
            "command, or pass --no-launch to attach to a running server. See "
            "docs/LOCAL_GPU.md."
        )
    exe = os.environ.get("KOBOLDCPP_EXE", "koboldcpp.exe")
    port = os.environ.get("KOBOLD_PORT", "5001")
    device = os.environ.get("KOBOLD_DEVICE", "0")
    return (
        f'"{exe}" --model "{model}" --port {port} '
        f"--usecublas normal {device} mmq --gpulayers 99 --contextsize 8192 "
        f"--flashattention --quantkv 1"
    )


def _get_json(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def wait_for_server(base: str, timeout_s: float = 240.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # /v1/models is the generic OpenAI-compatible readiness probe.
        if _get_json(f"{base}/v1/models") is not None:
            return
        time.sleep(2)
    raise RuntimeError(f"model server did not come up on {base} within {timeout_s:.0f}s")


def served_model_id(base: str) -> str:
    info = _get_json(f"{base}/v1/models")
    try:
        return info["data"][0]["id"]  # type: ignore[index]
    except (TypeError, KeyError, IndexError):
        return "local-model"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the ASR harness against a local OpenAI-compatible model server",
        epilog="Unrecognized args are forwarded to eval.harness (e.g. --json, --show-replies).",
    )
    ap.add_argument("--kobold-cmd", help="override: full model-server launch command line")
    ap.add_argument("--duration", type=int, default=60,
                    help="lease minutes (auto-heartbeated; default 60)")
    ap.add_argument("--force", action="store_true",
                    help="cancel the arbiter's active GPU jobs when acquiring the lease")
    ap.add_argument("--no-lease", action="store_true",
                    help="don't use a GPU lease (no arbiter on this machine)")
    ap.add_argument("--no-launch", action="store_true",
                    help="assume the model server is already running; don't launch/stop it")
    args, passthrough = ap.parse_known_args()

    base = kobold_url()
    cmd = args.kobold_cmd or (None if args.no_launch else build_launch_cmd())

    server_proc: subprocess.Popen | None = None
    log_path = Path(__file__).resolve().parent.parent / "logs" / "model-server.log"
    log_path.parent.mkdir(exist_ok=True)

    # Optionally wrap the whole run in a GPU lease.
    lease_ctx = None if args.no_lease else GpuLease(
        duration_minutes=args.duration, force=args.force
    )

    def run() -> int:
        nonlocal server_proc
        if not args.no_launch:
            argv = split_cmd(cmd)
            exe = argv[0]
            looks_like_path = (os.path.sep in exe) or ("/" in exe) or (":" in exe)
            if looks_like_path and not Path(exe).exists():
                raise SystemExit(f"model server executable not found: {exe} (check KOBOLDCPP_EXE / --kobold-cmd)")
            print(f"[local-gpu] launching model server", flush=True)
            with open(log_path, "w") as log:
                server_proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        print(f"[local-gpu] waiting for model server at {base} ...", flush=True)
        wait_for_server(base)
        model_id = served_model_id(base)
        print(f"[local-gpu] server ready, serving {model_id!r}", flush=True)

        env = dict(os.environ)
        env["MODEL_BACKEND"] = "openai"
        env["OPENAI_BASE_URL"] = f"{base}/v1"
        env.setdefault("OPENAI_API_KEY", "local")  # unused by local server, SDK needs non-empty
        env["OPENAI_MODEL"] = model_id

        harness_cmd = [sys.executable, "-m", "eval.harness", "--backend", "openai"]
        harness_cmd += passthrough
        print(f"[local-gpu] running harness: {' '.join(harness_cmd)}", flush=True)
        return subprocess.run(harness_cmd, env=env).returncode

    try:
        if lease_ctx is None:
            return run()
        with lease_ctx:
            return run()
    except GpuLeaseError as e:
        print(f"[local-gpu] GPU lease error: {e}", file=sys.stderr)
        return 2
    finally:
        if server_proc and server_proc.poll() is None:
            print("[local-gpu] stopping model server", flush=True)
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
