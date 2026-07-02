# cot-resistance — Authenticated Role Tags

A proof-of-concept defense against the **role-confusion prompt injection**
attack from [*Prompt Injection as Role Confusion*](https://role-confusion.github.io/)
(arXiv:2603.12277, in `docs/`).

## The attack, in one line

LLMs infer *who is speaking* from how text **sounds**, not from its tagged
source. Untrusted text that imitates a privileged role (`<think>` reasoning,
`<system>` instructions, `<tool>` output) inherits that role's authority — so
a user message carrying a fake reasoning trace gets trusted like the model's
own thoughts.

## The defense: seal the tags

Role boundaries become **cryptographically authenticated** so they can't be
forged in the text stream. Each conversation segment is wrapped in a *seal*:

```
⟦seal role=user seq=3 mac=9f2c…⟧ hello ⟦/seal seq=3⟧
```

where

```
mac = HMAC-SHA256(session_key, role | seq | prev_mac | SHA256(content))
```

Three properties make roles unforgeable:

1. **Keyed** — the MAC uses a per-session key generated server-side that
   **never appears in the text stream**, so a user (or injected tool output)
   cannot compute a valid MAC for a role they weren't assigned. This is the
   "not reproducible by a user in the text stream" requirement.
2. **Chained** — each seal commits to the previous seal's MAC, so editing,
   reordering, or splicing segments (even across sessions) breaks verification.
3. **Sanitized** — untrusted input is NFKC-normalized and any tag-lookalike
   (`<system>`, `<|im_start|>`, `[INST]`, `<<SYS>>`, seal delimiters, homoglyphs)
   is escaped before sealing. Plain-text role declarations are flagged.
   This is defense-in-depth; the MAC is the real boundary.

The model only ever acts on segments whose seals verify. A forged tag arrives
sealed as `user` (and escaped), so it can never masquerade as a trusted role.

### Why HMAC (and not a plain hash)

A plain hash (SHA-256) is reproducible by anyone — an attacker who knows the
content can compute it, so it authenticates nothing. HMAC mixes in a secret
key, so only the runtime holding the key can produce a valid tag. It's also
fast: ~microseconds per seal (see `test_seal_performance`), negligible next to
model inference. For a production system you'd bind the key to a KMS/HSM and
possibly sign rather than MAC; the mechanism is identical.

## Layout

```
server/
  sealer.py      seal / verify / sanitize — the security core (no deps)
  transcript.py  append-only chain of sealed segments
  model.py       MockBackend (gullible, GPU-free) + optional HFBackend
  pipeline.py    one turn: sanitize → seal → verify → render → generate
  main.py        FastAPI: /api/session, /api/chat + static UI
web/             two-pane UI: user chat  +  security console (the text stream)
tests/           seal/verify/sanitize + tamper, splice, perf tests
```

The `MockBackend` is deliberately gullible about tag-shaped text so you can see
both the attack and the defense **without a GPU**. It is not a real LLM; the
integrity guarantee comes from the verifier, not the model.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -v
.venv/bin/uvicorn server.main:app --reload
# open http://127.0.0.1:8000
```

**Left pane** is the generic chat a user sees. **Right pane** is the security
console showing every pipeline stage on the real text stream. Click an attack
preset, send it, then flip **Defense ON → OFF** and replay to watch the same
injection succeed (the mock leaks its secret `PINEAPPLE-42`).

## Measuring the defense (ASR harness)

`eval/harness.py` runs an attack battery (`eval/attacks.py`) through any backend
with the defense OFF then ON, scoring each by whether the secret leaked — an
objective metric, no LLM judge needed.

```bash
# Offline sanity check (mock backend, deterministic):
.venv/bin/python -m eval.harness --backend mock

# Inspect the exact messages that WOULD go to OpenAI (no API call, no key):
.venv/bin/python -m eval.harness --preview

# Real measurement against a live model:
OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o \
    .venv/bin/python -m eval.harness --backend openai --show-replies
```

Output reports baseline ASR (OFF), defended ASR (ON), and attacks blocked. A
benign control ("just tell me the secret") should stay refused in both modes.

Capture results for the design-doc results section (table goes to stderr, so
JSON on stdout stays clean):

```bash
.venv/bin/python -m eval.harness --backend hf --json results/gpt-oss.json --csv results/gpt-oss.csv
.venv/bin/python -m eval.harness --backend mock --json - | jq .summary
```

**Which model to test on.** The `openai` backend tests the *behavioral*
mitigation only (a real model told to trust seals over style). Frontier models
are already injection-hardened, so baseline ASR may be low — if so, that's a
finding, and the informative experiment moves to an **open, non-hardened model**
(below), which also has a measurable baseline vulnerability to move. Testing the
*strong* defense (moving role perception out of style in latent space) needs
open weights; see `docs/DESIGN.md` §6.

### Providing the OpenAI key

The key is read from `OPENAI_API_KEY` and never lives in code or git. Two ways:

```bash
cp .env.example .env && $EDITOR .env      # paste the key into the gitignored .env
```

or use the opt-in local setup page (writes .env for you; localhost only, never
touches the chat):

```bash
ENABLE_KEY_SETUP=1 .venv/bin/uvicorn server.main:app
# open http://127.0.0.1:8000/setup , paste the key, restart the server
```

The setup page 404s unless `ENABLE_KEY_SETUP=1`, and the write endpoint rejects
any non-loopback caller — so a deployed image never exposes it.

### Backends

| `MODEL_BACKEND` | What it is | Needs |
|---|---|---|
| `mock` (default) | Scripted stand-in for a gullible LLM | nothing |
| `openai` | Real black-box model via API | `OPENAI_API_KEY`; `openai` pkg |
| `hf` | Local open model (default `openai/gpt-oss-20b`) | GPU host; see [docs/HF_SETUP.md](docs/HF_SETUP.md) |

```bash
# Real black-box model in the web UI:
OPENAI_API_KEY=sk-... MODEL_BACKEND=openai .venv/bin/uvicorn server.main:app

# Local open model (GPU host, e.g. 16 GB VRAM box — see docs/HF_SETUP.md):
MODEL_BACKEND=hf HF_MODEL=openai/gpt-oss-20b .venv/bin/uvicorn server.main:app
```

gpt-4o refuses this attack battery even undefended (0% baseline ASR), so a
non-hardened open model is where the defense's value is actually measurable —
either the `hf` backend on `gpt-oss-20b` ([docs/HF_SETUP.md](docs/HF_SETUP.md)),
or, on a consumer-GPU workstation, the local model-server route (no torch/MXFP4):

```powershell
.venv\Scripts\python -m eval.local_gpu --json results/local.json
```

It serves a vulnerable open model over any OpenAI-compatible local endpoint
(and, where the machine has a GPU-arbitration service, leases the GPU for the
run). See [docs/LOCAL_GPU.md](docs/LOCAL_GPU.md).

The OpenAI backend, by default, flattens the conversation into one user message
so the model must infer roles from *content* (reproducing the paper's
role-confusion setting) rather than leaning on the API's role fields. Set
`OPENAI_FLATTEN=0` for native-role mapping.

## Scope / caveats

Proof of concept, not production. It demonstrates the mechanism: authenticated,
tamper-evident role boundaries that survive in the text stream. Real deployment
needs the key material managed properly, the seal format enforced at the
tokenizer/template layer, and a model trained (or fine-tuned) to treat only
sealed tags as role boundaries.
