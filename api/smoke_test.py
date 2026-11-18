"""Call /predict for EVERY trip the API lists and print a result table.

    # against a running server (Laptop 1 or from the UI laptop):
    python -m api.smoke_test --url http://localhost:8000
    # or start the app in-process (no server needed; uses the PATHSNAP_* env vars):
    python -m api.smoke_test --inprocess

Exit code 0 only if all trips return HTTP 200, a non-empty route, the expected number of
GPS points, and no ground-truth-looking field. Run it before the demo.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FORBIDDEN = ("reference", "truth", "label", "ground")


def fmt(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


def check(body: dict) -> list[str]:
    problems = [f"leaks field {k!r}" for k in body if any(w in k.lower() for w in FORBIDDEN)]
    if not body.get("predicted_route_arcs"):
        problems.append("empty predicted route")
    if len(body.get("per_point_arc", [])) != body.get("n_points"):
        problems.append("per_point_arc length != n_points")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--inprocess", action="store_true")
    ap.add_argument("--expect", type=int, default=10, help="expected number of trips")
    a = ap.parse_args()

    if a.inprocess:
        from fastapi.testclient import TestClient
        from api.main import app
        ctx = TestClient(app)
        ctx.__enter__()          # runs the startup (preload + warm-up)
        client, get, post = ctx, ctx.get, ctx.post
    else:
        import httpx
        client = httpx.Client(base_url=a.url, timeout=120)
        get, post = client.get, client.post

    health = get("/health").json()
    trips = [t["trip_id"] for t in get("/trips").json()]
    print(f"health: {health}\ntrips listed: {len(trips)}\n")
    hdr = f"{'trip':10} {'http':>4} {'ms':>7} {'points':>6} {'F1 hmm':>7} {'F1 cls':>7} {'cont.':>6} {'jumps':>5} {'breaks':>6}  status"
    print(hdr + "\n" + "-" * len(hdr))
    failures = 0
    for tid in trips:
        t0 = time.perf_counter()
        try:
            r = post("/predict", json={"trip_id": tid})
            ms = (time.perf_counter() - t0) * 1e3
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            b = r.json()
            problems = check(b)
            m, c = b["metrics_hmm"], b["metrics_classifier_only"]
            print(f"{tid:10} {r.status_code:>4} {ms:7.0f} {b['n_points']:6d} {fmt(m['route_f1']):>7} "
                  f"{fmt(c['route_f1']):>7} {m['route_continuity']:6.3f} {m['n_discontinuities']:5d} "
                  f"{b['hmm_breaks']:6d}  {'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
            failures += bool(problems)
        except Exception as e:      # keep going so one bad trip does not hide the others
            failures += 1
            print(f"{tid:10} {'-':>4} {'-':>7} {'-':>6} {'-':>7} {'-':>7} {'-':>6} {'-':>5} {'-':>6}  FAIL: {e}")
    if len(trips) != a.expect:
        failures += 1
        print(f"\nFAIL: expected {a.expect} trips, API lists {len(trips)}")
    print(f"\n{'ALL OK' if not failures else str(failures) + ' PROBLEM(S)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
