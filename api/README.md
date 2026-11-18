# Decoder + API (Person 4)

Code: `src/hmm/` (graph, Viterbi, HMM decoder, scorers, tuning, comparison, Folium maps),
`api/` (FastAPI backend + smoke test), `tests/`.

## Install
Python 3.10+ (developed on 3.12).

    python -m venv .venv && source .venv/bin/activate
    pip install -r api/requirements.txt
    # extra, only for tuning / comparison plots / classifier artifacts:
    pip install optuna matplotlib scikit-learn joblib

`api/requirements.txt` is kept separate; merge it into the root `requirements.txt`
(add `optuna`, `matplotlib`, `scikit-learn`, `joblib` as well).

## Environment variables
| variable | meaning | default |
|---|---|---|
| `PATHSNAP_DATA_DIR` | folder holding the partition / trip folders | `data/raw` |
| `PATHSNAP_PARTITION` | partition folder to serve (`student_test` now, evaluator folder on demo day) | all trips found |
| `PATHSNAP_SCORER` | `distance` (no ML, dev baseline) or `classifier` (trained model) | `distance` |
| `PATHSNAP_ARTIFACTS` | folder with `model.joblib`, `hmm_config.json`, ... | `artifacts/` |
| `PATHSNAP_CORS_ORIGINS` | comma-separated UI origins, or `*` | `*` |
| `PATHSNAP_PRELOAD` / `PATHSNAP_WARMUP` | load all graphs at start / also run one prediction per trip | `1` / `1` |

## Run the API (Laptop 1)
    export PATHSNAP_DATA_DIR="<dir containing the partition folder>"
    export PATHSNAP_PARTITION=<evaluator partition folder name>
    export PATHSNAP_SCORER=classifier
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Start-up preloads all road graphs and warms each trip (log line `READY` when done). If the
scorer / artifacts are broken the server refuses to start (no silent fallback to distance
scoring). Every request prints a log line (method, path, status, ms, client) plus a per-trip
result line - keep this console visible for the evaluators.
The UI laptop calls `http://<laptop-1-ip>:8000` (same Wi-Fi; allow port 8000 in the firewall).

Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`, `GET /predict/{trip_id}`.
Responses never contain reference routes; accuracy numbers are computed server-side.

## Test
    export PATHSNAP_DATA_DIR="<.../Student folder>"
    python -m pytest tests -q                      # unit + API tests
    # smoke test: /predict for every trip, prints a table, exit code != 0 on any failure
    PATHSNAP_PARTITION=student_test python -m api.smoke_test --inprocess
    python -m api.smoke_test --url http://<laptop-1-ip>:8000     # against the running server

## Reproduce the decoder results (development data only: train + student_test)
    D="<.../Student folder>"
    python -m src.hmm.tune_decoder --data-dir "$D" --scorer distance --n-trials 60 --seed 0 \
        --tag distance --write-artifact        # -> results/decoder_tuning_distance.csv, artifacts/hmm_config.json
    python -m src.hmm.compare --data-dir "$D" --partition student_test --scorer distance --tag distance
    python -m src.hmm.make_maps --data-dir "$D" --partition student_test --scorer distance
    # once the classifier is integrated, repeat all three with --scorer classifier --tag classifier

Outputs land in `results/` (CSVs, PNGs, `results/maps/trip_<id>.html`).

### Screenshot of a Folium map for the report
Open `results/maps/trip_<id>.html` in Chrome, tick GPS + Reference + Predicted layers, zoom to
the whole trip with the legend visible, take an OS screenshot, save into `reports/`.
(`--screenshot` on `make_maps` automates it if `playwright install chromium` was run.)

## Integration points (needs teammates)
`ClassifierScorer` (`src/hmm/scoring.py`, re-exported from `api/pipeline.py`) needs:
- `artifacts/model.joblib` - fitted estimator with `predict_proba` (positive class = 1 = "candidate is the true arc"). Optional `artifacts/preprocessor.joblib` (fitted on train only) and `artifacts/manifest.json` (`{"positive_class": 1, "temperature": 1.0}`).
- `src.features.build_candidate_features(trip, cand_arcs, cand_dist) -> list of (K_t, F) arrays`, rows aligned with `cand_arcs[t]`; `trip` has `graph`, `track_xy` (UTM m), `track_lonlat`, `track_t`, no labels.
- `src/hmm/candidates.py` is a temporary candidate generator (STRtree, 50 m radius, 15 nearest); features must be computed for *those* candidates, or the team's generator must replace it.
- Pointwise accuracy: per-point arc labels from Person 1's alignment as `<trip_id>.npy` (`--point-labels-dir`).
After the real model exists, re-run `tune_decoder --scorer classifier --write-artifact`:
`hmm_config.json` tuned on the distance baseline is not tuned for classifier probabilities.

## Troubleshooting
- `ModuleNotFoundError: pyproj/shapely/...` - venv not active or requirements not installed.
- Server exits at start with `build_candidate_features ... does not exist` / `model.joblib not found` - classifier artifacts not integrated; this is intentional. Use `PATHSNAP_SCORER=distance` only for development.
- `no trips found` / `expected 10 evaluation trips but found N` - wrong `PATHSNAP_DATA_DIR` or `PATHSNAP_PARTITION`.
- UI cannot reach the API - use `--host 0.0.0.0`, the laptop's LAN IP, same network, firewall open on 8000; CORS is open by default.
- `/predict` 404 - trip id not in the served partition (check `GET /trips`).
- First request slow - ensure `PATHSNAP_PRELOAD=1` and wait for `READY`.

## UI (`ui/index.html`)
Static page (Leaflet + plain JS, no build). Start the API, then serve the page: `cd ui && python3 -m http.server 5173` and visit `http://localhost:5173`.
Do NOT double-click the file: OpenStreetMap blocks map tiles for `file://` pages (the page shows a warning).
Set the API address in the sidebar (e.g. `http://<laptop-1-ip>:8000`), pick a trip, press
Predict. It only calls `/health`, `/trips`, `/predict`; metrics come from the server and the
reference route is never sent to the browser. Map tiles and Leaflet load from the internet.
