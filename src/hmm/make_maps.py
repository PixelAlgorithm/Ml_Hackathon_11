"""Final Folium maps: one interactive HTML per trip -> results/maps/trip_<id>.html

    python -m src.hmm.make_maps --data-dir "<Student folder>" --partition student_test \
        --scorer distance

Layers: raw GPS, road network (near the trace), reference route (.route), classifier-only
route (dashed, hidden by default), classifier+HMM route; legend + layer control built in.

=====================  HOW TO EXPORT A SCREENSHOT FOR THE REPORT  =====================
1. Open results/maps/trip_<id>.html in Chrome (needs internet for the OpenStreetMap tiles).
2. Tick the layers you want in the top-right layer control (e.g. GPS + Reference + Predicted).
3. Zoom so the whole trip is visible and the legend (bottom-left) is in frame.
4. Take a screenshot (macOS: Cmd+Shift+4, Windows: Win+Shift+S) and save it in reports/.
   Use a student_test trip for the report -- state the partition in the caption.
(Optional automation: `--screenshot` renders PNGs if `playwright install chromium` was run.)
========================================================================================
The reference route is shown only on these offline report maps; the live API never
returns it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import load_cfg
from .folium_map import make_route_map
from .graph import list_trip_ids
from .runner import DEV_PARTITIONS, prepare, run
from .scoring import DistanceScorer, build_scorer

SCREENSHOT_HELP = __doc__.split("=====")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trips", nargs="*")
    ap.add_argument("--scorer", choices=["distance", "classifier"], default="distance")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--hmm-config", default=None)
    ap.add_argument("--out-dir", default="results/maps")
    ap.add_argument("--screenshot", action="store_true", help="also save PNGs via playwright (optional)")
    a = ap.parse_args()
    if a.partition not in DEV_PARTITIONS:
        print(f"NOTE: partition {a.partition!r} is not a development partition; these maps are display-only "
              f"and nothing is tuned on them.")
    cfg, src = load_cfg(a.hmm_config or Path(a.artifacts) / "hmm_config.json")
    scorer = DistanceScorer(cfg.sigma_m) if a.scorer == "distance" else build_scorer("classifier", a.artifacts)
    cls_label = "Nearest-arc (distance baseline)" if a.scorer == "distance" else "Classifier-only prediction"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for tid in a.trips or list_trip_ids(a.data_dir, a.partition):
        prep = prepare(a.data_dir, tid)
        r = run(prep, cfg, scorer)
        t = prep.trip
        f = out / f"trip_{tid}.html"
        make_route_map(t.graph, t.track_lonlat, predicted_arcs=r["hmm_route"], reference_arcs=t.route,
                       extra_routes={cls_label: r["argmax_route"]}, out_html=f, classifier_label=cls_label,
                       title=f"Trip {tid} | {a.partition} | scorer: {scorer.name} | HMM F1 {r['hmm']['route_f1']:.3f}")
        paths.append(f)
        print("wrote", f)
    if a.screenshot:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1400, "height": 900})
            for f in paths:
                pg.goto(f.resolve().as_uri())
                pg.wait_for_timeout(3000)       # let tiles load
                pg.screenshot(path=str(f.with_suffix(".png")))
                print("screenshot", f.with_suffix(".png"))
            b.close()
    print("\n" + SCREENSHOT_HELP)


if __name__ == "__main__":
    main()
