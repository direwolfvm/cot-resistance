# Deploy

## What ships
The default image runs the **mock backend** — no GPU, no model weights, tiny
image. It demonstrates the *defense* (seal/verify/sanitize), not a real LLM.
See `docs/DESIGN.md` §6 for the path to a real, seal-honoring model.

## Local

```bash
docker build -t cot-resistance .
docker run --rm -p 8080:8080 cot-resistance
# open http://127.0.0.1:8080
```

The container binds `0.0.0.0:${PORT:-8080}` so it works both locally and on
Cloud Run (which injects `PORT`).

## Google Cloud Run (mock backend)

```bash
PROJECT=$(gcloud config get-value project)
REGION=us-central1

# Build + push with Cloud Build, then deploy.
gcloud builds submit --tag gcr.io/$PROJECT/cot-resistance
gcloud run deploy cot-resistance \
  --image gcr.io/$PROJECT/cot-resistance \
  --region $REGION \
  --allow-unauthenticated \
  --min-instances 1 --max-instances 1 \
  --memory 512Mi
```

### Why `--max-instances 1`
Sessions and the per-session seal key live **in process memory**
(`SESSIONS` in `server/main.py`). With more than one instance, a request can
land on an instance that never saw the session and 404. For the PoC, pin to a
single instance. To scale out, move session state + keys to a shared store
(Redis / Firestore) and keep the key in Secret Manager or Cloud KMS — that's
the M4 hardening item in the design doc.

## Real model (HF backend)
Not for vanilla Cloud Run — it needs torch + weights + ideally a GPU.

```bash
docker build -f Dockerfile.hf -t cot-resistance-hf .
docker run --rm -p 8080:8080 cot-resistance-hf   # CPU: slow but works for 0.5B
```

Deploy targets: a GPU-backed GCE VM, GKE with a GPU node pool, or Cloud Run
GPU (limited regions). Set `HF_MODEL` to pick the model.
