# Running the open-model (HF) backend

Why: frontier APIs like gpt-4o refuse the attack battery even undefended (0%
baseline ASR — nothing for the defense to block). The informative experiment
needs a **non-hardened, injection-vulnerable** model. The paper's `gpt-oss`
family shows 79–94% ASR on CoT forgery, so that's our target.

The backend is already wired (`server/model.py` `HFBackend`) and uses the exact
same seal framing as the OpenAI backend, so the harness numbers are comparable.
You only need to install the model runtime and point `HF_MODEL` at it.

## Target: openai/gpt-oss-20b on the 16 GB VRAM box

`gpt-oss-20b` ships in native **MXFP4** quantization and is designed to fit
~16 GB of VRAM. The MXFP4 fast path needs recent CUDA + Triton kernels, which
are best supported on Linux — so on the Windows box, **WSL2 (Ubuntu) + CUDA is
the smoothest path**.

### WSL2 / Linux + NVIDIA GPU (recommended)

```bash
# in the repo, in a fresh venv
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# CUDA build of torch (pick the cu-version matching your driver; cu124 shown)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# gpt-oss needs a recent transformers + the MXFP4 kernels
pip install "transformers>=4.55" accelerate "triton>=3.4" kernels

# smoke test one generation before the full harness
MODEL_BACKEND=hf HF_MODEL=openai/gpt-oss-20b \
  python -c "from server.model import load_backend as L; import os; \
    from server.transcript import Transcript; from server.pipeline import handle_message; \
    b=L(); t=Transcript(); print(handle_message(t,b,'hi',True)['reply'][:200])"

# full measurement
MODEL_BACKEND=hf HF_MODEL=openai/gpt-oss-20b \
  python -m eval.harness --backend hf --show-replies
```

First run downloads ~13 GB of weights to the HF cache.

### If the MXFP4 kernels won't build (common on native Windows)

Two fallbacks, in order of preference:

1. **4-bit bitsandbytes load** of a still-vulnerable model that fits 16 GB:
   ```bash
   pip install bitsandbytes
   # then use a model + set load-in-4bit; e.g. Qwen2.5-7B or Llama-3.1-8B.
   ```
   (bitsandbytes on native Windows can be finicky; WSL2 is again smoother.)

2. **Smaller open model**, no special kernels — quicker but a weaker/less
   certain vulnerability signal than gpt-oss:
   ```bash
   pip install "transformers>=4.44" torch accelerate
   MODEL_BACKEND=hf HF_MODEL=Qwen/Qwen2.5-7B-Instruct \
     python -m eval.harness --backend hf --show-replies
   ```

## Quick local smoke test on the Mac (no GPU)

Not for real numbers — just to confirm the HF path runs end to end. Uses a tiny
model on CPU/MPS (slow):

```bash
pip install "transformers>=4.44" torch
MODEL_BACKEND=hf HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct HF_MAX_NEW_TOKENS=128 \
  python -m eval.harness --backend hf
```

## Knobs

| Env var | Default | Meaning |
|---|---|---|
| `HF_MODEL` | `openai/gpt-oss-20b` | any HF causal-LM id |
| `HF_FLATTEN` | `1` | flatten convo into one user msg (reproduce role confusion) vs native roles (`0`) |
| `HF_MAX_NEW_TOKENS` | `300` | generation length cap |

## Reading the result

Expect **high baseline ASR (defense OFF)** on gpt-oss — that's the point; it's
the vulnerability the paper documents. Then look at **defended ASR (defense
ON)**: the seal contract + sanitization should drop it. Sanitization alone
should defeat the tag-based forgeries (they arrive escaped); the contract is
what (partly) handles plain-text declarations. Whatever remains under defense ON
is the evidence for why the *strong* (Tier-2/3) defense in `docs/DESIGN.md` §6
is needed — prompt-level mitigation only goes so far, since the model treats the
seal as style, not a trust primitive (we saw gpt-4o literally mimic the seal
syntax).
