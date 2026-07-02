"""Cooperative GPU-lease client for a local GPU-arbitration service.

Some workstations run a service that arbitrates GPU access between applications
via a small HTTP lease API. Holding a lease tells that service to release the
GPU (stop its own GPU processes) and refuse to reclaim it until we release — so
we can run our own inference process without VRAM collisions.

Protocol (generic REST; auth via an ``X-API-Key`` header):
    GET  /api/external/gpu/status     -> {available, externalLease, activeJobs, ...}
    POST /api/external/gpu/acquire    {name, durationMinutes?, force?}
    POST /api/external/gpu/heartbeat  {name, durationMinutes?}   (extend)
    POST /api/external/gpu/release    {name}

The lease auto-expires server-side, so a long run MUST heartbeat. ``GpuLease``
is a context manager that acquires on enter, heartbeats on a daemon thread, and
releases on exit (even on error / Ctrl-C).

Config via env (also read from .env by ``server``):
    GPU_LEASE_URL      default http://127.0.0.1:3000   (the arbiter's base URL)
    GPU_LEASE_API_KEY  required — the arbiter's API key
    GPU_LEASE_NAME     default "cot-resistance"        (this holder's identifier)

Stdlib only (urllib) — no new dependency, matching the sealer's no-deps ethos.
If your machine has no such arbiter, skip this and run the model server yourself
(see docs/LOCAL_GPU.md, ``--no-lease``).
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_LEASE_URL = "http://127.0.0.1:3000"
DEFAULT_NAME = "cot-resistance"


class GpuLeaseError(RuntimeError):
    """Lease API returned an error (auth, conflict, active jobs, unreachable)."""


def _post(url: str, api_key: str, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    return _send(req, timeout)


def _get(url: str, api_key: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(
        url, method="GET", headers={"X-API-Key": api_key}
    )
    return _send(req, timeout)


def _send(req: urllib.request.Request, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        # The arbiter returns JSON error bodies (401/409/403/404) — surface them.
        try:
            payload = json.loads(e.read().decode() or "{}")
        except Exception:
            payload = {}
        msg = payload.get("error") or payload.get("message") or e.reason
        raise GpuLeaseError(f"HTTP {e.code}: {msg}") from None
    except urllib.error.URLError as e:
        raise GpuLeaseError(
            f"cannot reach the GPU lease API at {req.full_url} — is the GPU "
            f"arbitration service running? ({e.reason})"
        ) from None


@dataclass
class GpuLease:
    """Client + context manager for one GPU lease."""

    name: str = None            # type: ignore[assignment]
    api_key: str = None         # type: ignore[assignment]
    base_url: str = None        # type: ignore[assignment]
    duration_minutes: int = 60
    force: bool = False

    def __post_init__(self) -> None:
        self.name = self.name or os.environ.get("GPU_LEASE_NAME", DEFAULT_NAME)
        self.api_key = self.api_key or os.environ.get("GPU_LEASE_API_KEY", "")
        self.base_url = (self.base_url or os.environ.get("GPU_LEASE_URL", DEFAULT_LEASE_URL)).rstrip("/")
        self._stop = threading.Event()
        self._hb_thread: threading.Thread | None = None
        if not self.api_key:
            raise GpuLeaseError(
                "GPU_LEASE_API_KEY is not set. Put the GPU arbiter's API key in "
                "cot-resistance's .env as GPU_LEASE_API_KEY=... (or run without a "
                "lease; see docs/LOCAL_GPU.md)."
            )

    def _u(self, path: str) -> str:
        return f"{self.base_url}/api/external/gpu/{path}"

    # --- raw API calls -----------------------------------------------------
    def status(self) -> dict:
        return _get(self._u("status"), self.api_key)

    def acquire(self) -> dict:
        resp = _post(self._u("acquire"), self.api_key, {
            "name": self.name,
            "durationMinutes": self.duration_minutes,
            "force": self.force,
        })
        if not resp.get("success"):
            raise GpuLeaseError(resp.get("error", "acquire failed"))
        return resp

    def heartbeat(self) -> dict:
        return _post(self._u("heartbeat"), self.api_key, {
            "name": self.name, "durationMinutes": self.duration_minutes,
        })

    def release(self) -> dict:
        return _post(self._u("release"), self.api_key, {"name": self.name})

    # --- context manager with background heartbeat -------------------------
    def _heartbeat_loop(self) -> None:
        # Renew at 1/3 of the lease window so a missed beat still has slack.
        interval = max(30.0, self.duration_minutes * 60 / 3)
        while not self._stop.wait(interval):
            try:
                self.heartbeat()
            except GpuLeaseError as e:
                print(f"[gpu-lease] heartbeat failed: {e}", flush=True)

    def __enter__(self) -> "GpuLease":
        lease = self.acquire()["lease"]
        print(f"[gpu-lease] acquired as {self.name!r}, expires {lease['expiresAt']}",
              flush=True)
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=5)
        try:
            self.release()
            print("[gpu-lease] released", flush=True)
        except GpuLeaseError as e:
            print(f"[gpu-lease] release failed (lease will auto-expire): {e}",
                  flush=True)
