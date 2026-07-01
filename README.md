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

## Optional: a real model

Intended for GPU hosts (e.g. a 16 GB VRAM Windows box). Uncomment the
`transformers`/`torch` lines in `requirements.txt`, then:

```bash
MODEL_BACKEND=hf HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct .venv/bin/uvicorn server.main:app
```

The sealed stream is passed to the model with a header explaining the format.
A pretrained model wasn't trained on sealed tags, so answer quality degrades —
acceptable for a PoC, and the integrity guarantee is unaffected because it
lives in the verifier.

## Scope / caveats

Proof of concept, not production. It demonstrates the mechanism: authenticated,
tamper-evident role boundaries that survive in the text stream. Real deployment
needs the key material managed properly, the seal format enforced at the
tokenizer/template layer, and a model trained (or fine-tuned) to treat only
sealed tags as role boundaries.
