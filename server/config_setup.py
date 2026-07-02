"""Local, opt-in setup page for writing the OpenAI API key into .env.

Security posture:
  * Disabled unless ENABLE_KEY_SETUP=1, so the deployed image never exposes it.
  * The POST endpoint rejects any non-loopback client (local machine only).
  * The key travels form -> 127.0.0.1 -> .env on disk. It never enters the chat
    transcript, code, or git (.env is gitignored).
This is a dev convenience; production key management belongs in a secret store.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def enabled() -> bool:
    return os.environ.get("ENABLE_KEY_SETUP", "0") == "1"


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost")


def _upsert_env(updates: dict[str, str]) -> None:
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in updates:
                out.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(out) + "\n")
    os.chmod(ENV_PATH, 0o600)


SETUP_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>cot-resistance · key setup</title>
<style>
 body{font-family:-apple-system,Segoe UI,sans-serif;background:#0d1117;color:#e6edf3;
  display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}
 .card{background:#161b22;border:1px solid #2d333b;border-radius:12px;padding:28px;width:420px}
 h1{font-size:18px;margin:0 0 4px} p{color:#8b949e;font-size:13px;margin:6px 0 16px}
 label{display:block;font-size:12px;color:#8b949e;margin:12px 0 4px}
 input,select{width:100%;box-sizing:border-box;background:#0a0d12;color:#e6edf3;
  border:1px solid #2d333b;border-radius:6px;padding:9px 11px;font-size:13px;font-family:inherit}
 button{margin-top:18px;width:100%;background:#1f6feb;color:#fff;border:none;border-radius:6px;
  padding:10px;font-size:14px;font-weight:600;cursor:pointer}
 #msg{margin-top:14px;font-size:13px;min-height:18px}
 .ok{color:#3fb950} .bad{color:#f85149}
 .note{font-size:11.5px;color:#8b949e;margin-top:16px;line-height:1.5}
</style></head><body>
<div class="card">
 <h1>&#x27E6;&#x1F517;&#x27E7; API key setup</h1>
 <p>Writes to the gitignored <code>.env</code> on this machine. The key never
 leaves localhost.</p>
 <label>OpenAI API key</label>
 <input id="key" type="password" placeholder="sk-..." autocomplete="off">
 <label>Model</label>
 <select id="model"><option>gpt-4o</option><option>gpt-4o-mini</option><option>gpt-4.1</option></select>
 <button id="save">Save to .env</button>
 <div id="msg"></div>
 <div class="note">After saving, restart the server so the new key loads:
  <code>uvicorn server.main:app</code>. This page only exists because
  <code>ENABLE_KEY_SETUP=1</code>.</div>
</div>
<script>
 document.getElementById('save').onclick = async () => {
  const key = document.getElementById('key').value.trim();
  const model = document.getElementById('model').value;
  const msg = document.getElementById('msg');
  msg.className=''; msg.textContent='Saving...';
  try{
   const r = await fetch('/api/config/openai-key',{method:'POST',
     headers:{'Content-Type':'application/json'},body:JSON.stringify({key,model})});
   const d = await r.json();
   if(d.ok){ msg.className='ok'; msg.textContent='Saved to '+d.wrote+' - restart the server.';
     document.getElementById('key').value=''; }
   else { msg.className='bad'; msg.textContent=d.error||'Failed'; }
  }catch(e){ msg.className='bad'; msg.textContent=e.message; }
 };
</script></body></html>"""


def register(app) -> None:
    """Attach the setup routes. Call before mounting static files at '/'."""

    @app.get("/setup")
    def setup_page():
        if not enabled():
            raise HTTPException(status_code=404, detail="key setup disabled")
        return HTMLResponse(SETUP_HTML)

    @app.post("/api/config/openai-key")
    async def set_openai_key(request: Request) -> dict:
        if not enabled():
            raise HTTPException(status_code=404, detail="key setup disabled")
        if not _is_loopback(request):
            raise HTTPException(status_code=403, detail="local machine only")
        body = await request.json()
        key = (body.get("key") or "").strip()
        model = (body.get("model") or "gpt-4o").strip()
        if not key.startswith("sk-"):
            return {"ok": False, "error": "key should start with 'sk-'"}
        _upsert_env({"OPENAI_API_KEY": key, "OPENAI_MODEL": model,
                     "MODEL_BACKEND": "openai"})
        # Make it live for this process too (backend still needs a restart).
        os.environ.update({"OPENAI_API_KEY": key, "OPENAI_MODEL": model})
        return {"ok": True, "wrote": str(ENV_PATH), "model": model}
