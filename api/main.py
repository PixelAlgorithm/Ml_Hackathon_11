"""PathSnap FastAPI backend (runs on Laptop 1).

    PATHSNAP_DATA_DIR=<dir with partition/trip folders>  \
    PATHSNAP_PARTITION=student_test  (or the evaluator partition folder name) \
    uvicorn api.main:app --host 0.0.0.0 --port 8000

The UI sends only a trip id. Ground-truth routes are read server-side for scoring
and are never returned.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.pipeline import Pipeline, build_scorer, load_hmm_config          # noqa: E402
from api.schemas import PredictRequest, PredictResponse, TripInfo   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("pathsnap.api")

DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", ROOT / "data" / "raw"))
PARTITION = os.environ.get("PATHSNAP_PARTITION") or None
SCORER = os.environ.get("PATHSNAP_SCORER", "distance")
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))
# comma separated list of UI origins, or "*" (default) so the UI on Laptop 2 can call us
CORS_ORIGINS = [o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip()]

pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    # a bad scorer/artifact raises here -> uvicorn refuses to start (fail loudly, no fallback)
    pipeline = Pipeline(DATA_DIR, build_scorer(SCORER, ARTIFACTS), load_hmm_config(ARTIFACTS, SCORER),
                        partition=PARTITION)
    n = len(pipeline.trip_ids())
    log.info("data=%s partition=%s scorer=%s trips=%d", DATA_DIR, PARTITION, SCORER, n)
    if n == 0:
        raise RuntimeError(f"no trips found under {DATA_DIR} (partition={PARTITION!r})")
    if n != 10:
        log.warning("expected 10 evaluation trips but found %d -- check PATHSNAP_PARTITION", n)
    if os.environ.get("PATHSNAP_PRELOAD", "1") == "1":
        pipeline.preload(warmup=os.environ.get("PATHSNAP_WARMUP", "1") == "1")
    log.info("READY")
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info("%s %s -> %d (%.0f ms) from %s", request.method, request.url.path, resp.status_code,
             (time.perf_counter() - t0) * 1e3, request.client.host if request.client else "?")
    return resp


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """Any unexpected error -> logged with traceback + JSON 500 (CORS headers still added)."""
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/health")
def health():
    return {"status": "ok", "scorer": pipeline.scorer.name, "n_trips": len(pipeline.trip_ids())}


@app.get("/trips", response_model=list[TripInfo])
def trips():
    return [TripInfo(trip_id=t) for t in pipeline.trip_ids()]


def _predict(trip_id: str) -> PredictResponse:
    try:
        out = pipeline.run(trip_id)
    except KeyError:
        raise HTTPException(404, f"unknown trip_id {trip_id!r}")
    m = out["metrics_hmm"]
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    log.info("trip %s: route F1 %s (classifier-only %s), continuity %.3f, %d breaks, %.0f ms",
             trip_id, f(m["route_f1"]), f(out["metrics_classifier_only"]["route_f1"]),
             m["route_continuity"], out["hmm_breaks"], out["timings_ms"]["predict_ms"])
    return PredictResponse(**out)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _predict(req.trip_id)


@app.get("/predict/{trip_id}", response_model=PredictResponse)
def predict_get(trip_id: str):
    return _predict(trip_id)
