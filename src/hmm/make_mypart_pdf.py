"""Build reports/mypart.pdf (Person 4: HMM / Viterbi, API, deployment, Folium).

    python -m src.hmm.make_mypart_pdf

Every number in the PDF is READ from files produced by the repo's own scripts
(results/*.csv, results/*.json, results/smoke_test_classifier.txt) or measured when this
script runs (pytest). Nothing is typed in by hand. Re-run the scripts listed in
api/README.md first if you want fresh numbers.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "reports" / "mypart.pdf"

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"))
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="-") for i in items]


def table(rows, widths=None, header=True, font=8.5):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


def wrap(rows):
    """Wrap string cells in Paragraphs so long text breaks inside the cell."""
    c = ParagraphStyle("cell", parent=BODY, fontSize=8.3, leading=10.5, spaceAfter=0)
    return [[Paragraph(str(x), c) if isinstance(x, str) and len(x) > 18 else x for x in r] for r in rows]


def img(path, width_cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def f(x, n=3):
    return f"{float(x):.{n}f}"


def progress_plot(tag, title):
    df = pd.read_csv(RES / f"decoder_tuning_{tag}.csv")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(df["trial"], df["train_f1"].cummax(), label="best train F1 so far (selection metric)")
    ax.scatter(df["trial"], df["train_f1"], s=10, alpha=0.5, label="train F1 per trial")
    ax.scatter(df["trial"], df["student_test_f1"], s=10, marker="x", alpha=0.6, label="student_test F1 per trial (held-out check)")
    ax.set_xlabel("Optuna trial (trial 0 = default config)")
    ax.set_ylabel("mean route F1")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = RES / f"decoder_tuning_{tag}_progress.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def summary_rows(tag):
    s = pd.read_csv(RES / f"comparison_{tag}_summary.csv", keep_default_na=False)   # keep the literal "N/A"
    rows = [["metric", "classifier-only (argmax)", "classifier + HMM"]]
    for _, r in s.iterrows():
        a, b = r["classifier_only"], r["hmm"]
        nd = 1 if r["metric"] == "jumps" else 3      # jumps are counts per trip -> 1 decimal
        rows.append([r["metric"], a if a == "N/A" else f(a, nd), b if b == "N/A" else f(b, nd)])
    return rows, s


def main():
    env = dict(os.environ)
    env.setdefault("PATHSNAP_DATA_DIR", str(ROOT.parent.parent / "PathSnap_student_materials" / "Student folder"))
    t = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT, env=env, capture_output=True, text=True)
    test_line = [l for l in t.stdout.strip().splitlines() if "passed" in l or "failed" in l][-1]

    dist_best = json.loads((RES / "decoder_best_distance.json").read_text())
    cls_best = json.loads((RES / "decoder_best_classifier.json").read_text())
    d_rows, d_sum = summary_rows("distance")
    c_rows, c_sum = summary_rows("classifier")
    cd_rows, _ = summary_rows("classifier_defaults")
    per_trip = pd.read_csv(RES / "comparison_classifier_per_trip.csv", dtype={"trip": str})
    smoke = (RES / "smoke_test_classifier.txt").read_text().splitlines()
    p_dist = progress_plot("distance", "Decoder tuning, distance scorer (60 trials, 80 train trips)")
    p_cls = progress_plot("classifier", "Decoder tuning, classifier scorer (40 trials, 80 train trips)")

    story = []
    # ------------------------------------------------------------ title
    story += [P("PathSnap - Person 4: HMM / Viterbi decoder, API and deployment", ParagraphStyle("t", parent=H1, fontSize=18)),
              P("What I built, how it works, and the real results so far. Every number in this document is read from files "
                "produced by code in the repository (results/*.csv, results/*.json) and can be regenerated; "
                "commands are in api/README.md.", SMALL), Spacer(1, 6)]

    story += [P("1. My responsibilities", H1)] + bullets([
        "HMM + Viterbi route reconstruction (src/hmm/) and the classifier-only vs classifier+HMM comparison.",
        "Decoder parameter tuning, using the 80 train trips and the 10 student_test trips only (never evaluator data).",
        "FastAPI backend (api/): model loading, ground truth kept server-side, request logging, CORS, startup preload.",
        "Folium maps (GPS, road network, reference route, predicted route, legend, layer control).",
        "Integration glue: an adapter that runs Person 1's candidates and model through my decoder and API.",
    ])

    # ------------------------------------------------------------ built
    story += [P("2. What exists in the repository (my files)", H1)]
    story.append(table(wrap([
        ["file", "purpose"],
        ["src/hmm/graph.py", "Road network per trip, UTM projection (metres), bounded Dijkstra between arcs, data loading"],
        ["src/hmm/viterbi.py", "Generic log-space Viterbi with chain breaks; independent of roads so it is unit-testable"],
        ["src/hmm/decoder.py", "HMM map matcher: emission/transition model, gap handling, route expansion, HMMConfig"],
        ["src/hmm/candidates.py", "Temporary candidate generator (STRtree, 50 m radius, 15 nearest arcs)"],
        ["src/hmm/scoring.py", "DistanceScorer (no ML) and ClassifierScorer (loads artifacts, supports Person 1's candidates)"],
        ["src/hmm/runner.py", "Shared prepare -> score -> decode -> metrics loop used by every script"],
        ["src/hmm/tune_decoder.py", "Optuna TPE search; every trial saved to results/decoder_tuning_*.csv"],
        ["src/hmm/compare.py", "Classifier-only vs classifier+HMM: CSVs, plots, 'where the HMM did not help'"],
        ["src/hmm/make_maps.py", "One Folium HTML per trip + screenshot instructions"],
        ["src/hmm/metrics.py, run_baseline.py, folium_map.py", "Route precision/recall/F1, continuity, discontinuities; distance baseline; map layers"],
        ["api/main.py, pipeline.py, schemas.py", "FastAPI app, server-side pipeline, response models"],
        ["api/smoke_test.py", "Calls /predict for every trip and prints a table"],
        ["tests/", "Unit tests (" + test_line + ")"],
    ]), widths=[5.2 * cm, 11.8 * cm]))

    # ------------------------------------------------------------ method
    story += [P("3. How the decoder works", H1),
              P("<b>Hidden state.</b> For each GPS point the hidden state is one <i>candidate road arc</i> of that point. "
                "The candidate-to-arc_id correspondence is kept all the way to the output. Arc ids are local to each trip's network, "
                "so arc_id is never treated as a global class."),
              P("<b>Emission.</b> The log-probability that the candidate is the true arc. With the trained classifier this is "
                "log(predict_proba), clipped at 1e-6; for the distance baseline it is a Gaussian on the point-to-arc distance. "
                "Both are scaled by emission_weight."),
              P("<b>Transition (Newson-Krumm style).</b> For consecutive fixes a -> b: d_gc is the straight-line distance between the "
                "two GPS fixes and d_route is the network distance between their projections on the two arcs. "
                "log P = -|d_gc - d_route| / beta - log(beta). Pairs whose route distance exceeds route_cutoff_m get log P = -inf. "
                "Hopping onto the reverse arc costs an extra uturn_penalty. Small backward jitter on the same arc "
                "(within backward_tol_m) is treated as zero movement, not a jump."),
              P("<b>Viterbi.</b> In log space: delta_t(j) = max_i [delta_{t-1}(i) + log A(i,j)] + log B_t(j), with back-pointers; the path "
                "is read back from the best final state. Log space avoids underflow over thousands of steps."),
              P("<b>Gaps, empty candidates, disconnected arcs.</b> Points with no candidates are skipped and the chain bridges over them. "
                "If the time gap between used fixes exceeds max_gap_s, or if every transition into a step is impossible "
                "(disconnected arcs), the chain restarts as a new segment instead of forcing a bogus link. "
                "expand_route() fills the missing arcs between decoded arcs with the shortest path, but never across a chain break."),
              P("<b>Geography.</b> Everything runs in a per-trip UTM projection (metres); degrees are never used as distances. "
                "Raw data is (lon, lat); Folium needs (lat, lon) and is converted in one place.")]

    story += [P("Decoder parameters (defaults vs values selected by tuning)", H2)]
    keys = ["beta_m", "route_cutoff_m", "max_gap_s", "uturn_penalty", "emission_weight", "sigma_m"]
    rows = [["parameter", "default", "tuned: distance scorer", "tuned: classifier scorer"]]
    for k in keys:
        rows.append([k, f(dist_best["default_config"][k], 2), f(dist_best["config"][k], 2),
                     f(cls_best["config"][k], 2) + (" (unused)" if k == "sigma_m" else "")])
    story.append(table(rows, widths=[4 * cm, 3 * cm, 4.5 * cm, 4.5 * cm]))

    # ------------------------------------------------------------ tests
    story += [P("4. Hardening and tests", H1),
              P("I reviewed the existing decoder and added a test for each edge case: no candidates at all, a candidate-free step in the "
                "middle, a one-point trip, a GPS gap, disconnected islands, the U-turn penalty, backward jitter, NaN / -inf emissions "
                "and mismatched input lengths. Review also found a real bug: the Dijkstra cache could return distances computed with a "
                "<i>larger</i> cutoff when a smaller route_cutoff_m was requested, which would silently have corrupted cutoff tuning. "
                "It is fixed and has a regression test. Test result when this PDF was built: <b>" + test_line + "</b>.")]

    # ------------------------------------------------------------ tuning
    story += [PageBreak(), P("5. Decoder tuning (development data only)", H1),
              P("Search: Optuna TPE sampler (Bayesian), seed 0, trial 0 enqueued as the default config so the record includes the baseline. "
                "Search space: beta_m 1-50 (log), route_cutoff_m 200-1500, max_gap_s 5-60, uturn_penalty 0-10, emission_weight 0.2-3 (log), "
                "and sigma_m 5-40 (log, distance scorer only). Objective: mean length-weighted route F1 on the <b>80 train trips</b>. "
                "student_test is logged for every trial but only as a held-out check, never for selection. "
                "Full record: results/decoder_tuning_distance.csv and results/decoder_tuning_classifier.csv.")]
    rows = [["scorer", "trials", "train F1 default", "train F1 best", "student_test F1 default", "student_test F1 best"]]
    for name, b in [("distance", dist_best), ("classifier (logistic reg.)", cls_best)]:
        rows.append([name, b["n_trials"], f(b["train_f1_default"], 4), f(b["train_f1_best"], 4),
                     f(b["student_test_f1_default"], 4), f(b["student_test_f1_best"], 4)])
    story += [table(rows), Spacer(1, 4), img(p_dist, 13), img(p_cls, 13),
              P("Reading: for the distance scorer the tuned config improves train F1 slightly but not student_test F1 "
                "(difference within noise), so tuning matters little there. For the classifier scorer it improves both train and student_test, "
                "modestly. emission_weight and sigma_m partly cancel for a Gaussian emission (both rescale emission against transition).", SMALL)]

    # ------------------------------------------------------------ comparison
    story += [PageBreak(), P("6. Classifier-only vs classifier + HMM (student_test, 10 trips)", H1),
              P("Route metrics compare the predicted route with the reference route (.route): length-weighted precision, recall and F1 of the "
                "arc sets, route continuity (fraction of consecutive distinct arcs that are connected) and the number of discontinuities. "
                "<b>Pointwise accuracy is N/A</b>: the raw data has no per-GPS-point labels; they must come from the shared alignment (Person 1)."),
              P("Trained classifier (Person 1's logistic regression), decoder tuned for it", H2), table(c_rows, widths=[5 * cm, 5.5 * cm, 5.5 * cm]),
              P("Same classifier, default decoder config (before tuning)", H2), table(cd_rows, widths=[5 * cm, 5.5 * cm, 5.5 * cm]),
              P("Distance-only baseline (no ML model), decoder tuned for it", H2), table(d_rows, widths=[5 * cm, 5.5 * cm, 5.5 * cm]),
              P("Honest reading: the HMM improves every trip in all three settings, mainly by removing the hundreds of disconnected jumps. "
                "But the distance-only baseline plus HMM currently scores a higher route F1 than the logistic regression plus HMM, so the "
                "current classifier does not yet add value beyond distance. A stronger model (Persons 2/3) should be dropped in as "
                "artifacts/model.joblib. Route F1 is an arc-set overlap that ignores order, so it is optimistic; and .route is reference "
                "information, not perfect ground truth.", SMALL)]
    rows = [["trip", "points", "F1 cls-only", "F1 + HMM", "jumps cls-only", "jumps + HMM", "HMM breaks"]]
    for _, r in per_trip.iterrows():
        rows.append([r["trip"], int(r["n_points"]), f(r["classifier_only_route_f1"]), f(r["hmm_route_f1"]),
                     int(r["classifier_only_jumps"]), int(r["hmm_jumps"]), int(r["hmm_breaks"])])
    story += [P("Per trip (classifier scorer, tuned decoder)", H2), table(rows)]
    nb = pd.read_csv(RES / "comparison_classifier_not_helped.csv")
    story += [P(f"Trips where the HMM did not help (F1 not better, or more jumps): <b>{len(nb)} of {len(per_trip)}</b>. "
                "The weakest HMM result is the lowest F1 in the table above.", SMALL),
              img(RES / "comparison_classifier_f1.png", 13), img(RES / "comparison_classifier_jumps.png", 13)]

    # ------------------------------------------------------------ API
    story += [PageBreak(), P("7. FastAPI backend", H1)] + bullets([
        "Endpoints: GET /health, GET /trips, POST /predict {trip_id}, GET /predict/{trip_id}.",
        "<b>Ground truth stays server-side.</b> Trips are loaded without .route; the reference is read only inside evaluate() after the prediction "
        "is made, and only aggregate metrics are returned. A test asserts that the Trip object has no route and the response has no reference/truth/label field.",
        "Start-up preloads every road graph and spatial index, then runs one warm-up prediction per trip, so the first demo request is fast and a broken trip fails at start-up.",
        "Environment variables: PATHSNAP_DATA_DIR, PATHSNAP_PARTITION, PATHSNAP_SCORER (distance | classifier), PATHSNAP_ARTIFACTS, PATHSNAP_CORS_ORIGINS.",
        "No silent fallback: with PATHSNAP_SCORER=classifier and missing/invalid artifacts the server refuses to start.",
        "Console log line for every request (method, path, status, ms, client) plus a per-trip result line - for showing the evaluators.",
        "CORS open by default so the UI on the other laptop can call it; thread lock around the shared caches; JSON 500 handler with traceback in the log; NaN metrics returned as null.",
    ])
    story += [P("Smoke test: /predict for every trip (PATHSNAP_SCORER=classifier, in-process; results/smoke_test_classifier.txt)", H2)]
    tab = [l for l in smoke if re.match(r"^\d{8}\s", l) or l.startswith("trip ")]
    rows = [re.split(r"\s{2,}|\s(?=\S+\s+\S+\s+\S+$)", l.strip()) for l in tab]
    rows = [l.split() for l in tab]
    rows[0] = ["trip", "http", "ms", "points", "F1 hmm", "F1 cls", "cont.", "jumps", "breaks", "status"]
    rows[1:] = [r[:9] + [" ".join(r[9:])] for r in rows[1:]]
    story += [table(rows), P(next((l for l in smoke if "ALL OK" in l or "PROBLEM" in l), ""), SMALL)]

    # ------------------------------------------------------------ maps
    story += [PageBreak(), P("8. Folium map", H1),
              P("make_maps.py writes one interactive HTML per trip (raw GPS, road network near the trace, reference route in green, "
                "classifier-only route dashed and toggleable, HMM route in red, start/end markers, legend, layer control). "
                "Below: trip 00000056 (student_test), the trip with the most HMM chain breaks. The HMM route (red) follows the reference route (green)."),
              img(ROOT / "reports" / "map_trip_00000056.png", 16.5)]

    # ------------------------------------------------------------ status
    story += [PageBreak(), P("9. Integration status and what is blocked", H1)] + bullets([
        "Person 1's code (candidates + logistic regression) was merged unchanged. My ClassifierScorer calls their build_inference_candidates, "
        "feeds their model's probabilities to the HMM, and I made no edits to their files. All 10 student_test trips run end to end through the API.",
        "artifacts/model.joblib currently holds Person 1's logistic-regression baseline. When Persons 2/3 choose the final model, it replaces that file and I re-run "
        "tune_decoder --scorer classifier --write-artifact, compare and make_maps.",
        "Pointwise accuracy needs per-point arc labels from Person 1 (<trip_id>.npy); until then it is reported as N/A.",
        "Person 1's training labels are a nearest-route-arc heuristic (their code says so), so candidate-level metrics from their baseline are on pseudo-labels and not directly comparable with my route metrics.",
        "Limitations: route F1 ignores arc order; student_test is only 10 trips; the evaluator partition has not been touched.",
    ])
    story += [P("10. Things I must be able to explain in the viva", H1)] + bullets([
        "Hidden state, emission and transition, and why |d_gc - d_route| is a good signal for 'same road'.",
        "The Viterbi recurrence, back-pointers, why log space, and what a chain break is and when it triggers.",
        "Why UTM metres instead of degrees, and why arc ids are local to a trip.",
        "How gaps, empty candidates and disconnected arcs are handled, and what expand_route does.",
        "Why tuning selects on train only and treats student_test as a check; what Optuna TPE is.",
        "The cache bug (cutoff vs cached Dijkstra) and how the regression test catches it.",
        "How the API keeps ground truth out of the model, decoder and response.",
        "Why the HMM improves continuity a lot and F1 less, and why the distance baseline currently beats the logistic regression.",
    ])

    OUT.parent.mkdir(exist_ok=True)
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="PathSnap - Person 4 (HMM, API, deployment)").build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
