"""Orchestrates one chat turn and records an inspector trace of every stage."""

from __future__ import annotations

from dataclasses import asdict

from . import sealer
from .model import ModelOutput, SYSTEM_PROMPT
from .transcript import Transcript


def ensure_system(transcript: Transcript) -> None:
    if not transcript.segments:
        transcript.append("system", SYSTEM_PROMPT)


def handle_message(transcript: Transcript, backend, message: str, defense_on: bool) -> dict:
    ensure_system(transcript)
    trace: dict = {"defense_on": defense_on, "raw_input": message}

    # 1. Sanitize untrusted input (defense only)
    if defense_on:
        result = sealer.sanitize(message)
        content = result.clean
        trace["sanitization"] = {
            "skipped": False,
            "clean": result.clean,
            "modified": result.modified,
            "spans": [asdict(s) for s in result.spans],
        }
    else:
        content = message
        trace["sanitization"] = {"skipped": True, "clean": message,
                                 "modified": False, "spans": []}

    # 2. Seal the user segment onto the chain
    user_seg = transcript.append("user", content)
    trace["seal"] = {
        "role": user_seg.role,
        "seq": user_seg.seq,
        "mac": user_seg.mac,
        "content_hash": user_seg.content_hash,
        "prev_mac": user_seg.prev_mac,
    }

    # 3. Verify the whole chain before the model sees anything
    if defense_on:
        results = transcript.verify()
        trace["verification"] = {
            "skipped": False,
            "ok": all(r.ok for r in results),
            "segments": [asdict(r) for r in results],
        }
        if not trace["verification"]["ok"]:
            reply = ("⚠️ Conversation integrity check failed — a role tag could "
                     "not be authenticated. The turn was not sent to the model.")
            trace["rendered_prompt"] = None
            trace["model"] = {"backend": backend.name, "hijacked": False,
                              "notes": ["generation refused: seal verification failed"]}
            transcript.append("assistant", reply)
            trace["reply"] = reply
            return trace
    else:
        trace["verification"] = {"skipped": True, "ok": None, "segments": []}

    # 4. Render the stream exactly as the model consumes it
    rendered = transcript.render() if defense_on else transcript.render_unsealed()
    trace["rendered_prompt"] = rendered

    # 5. Generate, then seal the model's own output back onto the chain
    out: ModelOutput = backend.generate(transcript, defense_on)
    asst_seg = transcript.append("assistant", out.reply)
    trace["model"] = {"backend": backend.name, "hijacked": out.hijacked, "notes": out.notes}
    trace["reply"] = out.reply
    trace["assistant_seal"] = {"role": asst_seg.role, "seq": asst_seg.seq, "mac": asst_seg.mac}
    return trace


def transcript_view(transcript: Transcript) -> list[dict]:
    return [
        {"role": s.role, "seq": s.seq, "mac": s.mac, "content": s.content,
         "rendered": s.render()}
        for s in transcript.segments
    ]
