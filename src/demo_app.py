"""Step D2: demo page for the interview (Streamlit).

The page shows a non-technical audience how the baseline method and Toto find simulated incidents
on a real metric. It calls the API like a real client; if the API cannot be reached, it uses the
precomputed results of results/demo_cache.json (offline mode), so the demo never crashes.

How to run it, from the repository root:
1. terminal 1: uvicorn api:app --app-dir src   (without --reload), then wait for "Ready"
2. terminal 2: streamlit run src/demo_app.py, then open http://localhost:8501
Without step 1, the page works in offline mode. If results/demo_cache.json is missing,
rebuild it with python src/demo_precompute.py.
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")  # no window: figures are only rendered into the page
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from data import CONTEXT, HORIZON, inject_spikes, load_series, split_history_future
from plot_data import GREY, RESULTS, SURFACE
from plot_detectors import draw

API_URL = os.environ.get("API_URL", "http://localhost:8000")
HEALTH_TIMEOUT_S = 1
REQUEST_TIMEOUT_S = 5  # per HTTP request
POLL_INTERVAL_S = 0.2
JOB_TIMEOUT_S = 60  # beyond this, the page falls back to the cache
CACHE_PATH = RESULTS / "demo_cache.json"  # written by src/demo_precompute.py

# Demo series, chosen in step D3.1 (results/demo_candidates.csv): index -> (name, explanation).
# The explanations only describe the observed shape of the curve, not what the metric measures.
DEMO_SERIES = {
    12: ("Peak every five minutes",
         "Sampled every minute, this metric spikes sharply about every five minutes, then falls back."),
    4: ("Hourly dip",
        "Sampled every five minutes, this metric drops sharply about once an hour, then returns to its usual level."),
    38: ("Recent drop to a lower level",
         "Sampled every minute, this noisy metric dropped to a lower level during the hour before the analyzed "
         "window."),
    173: ("Plateaus with level shifts",
          "Sampled every 30 minutes, this metric stays on long plateaus and occasionally shifts to a new level, "
          "including a small step up about a day before the analyzed window."),
}
# API method -> (displayed name, color, chart title); colors are the same as in the interview slides
METHODS = {
    "baseline": ("Classic method", "#B34A12", "Classic method: fixed normal range"),
    "toto": ("Toto", "#2F5FD0", "Toto: forecast range, point by point"),
}
# draw() writes technical legend labels: they are replaced with the page vocabulary
LEGEND_LABELS = {
    "forecast interval": "normal range",
    "series": "metric",
    "injected spike": "simulated incident",
    "flagged point": "alert",
}
TOTO_SAMPLES_TUNED = 256  # number of Toto samples chosen in step 3.8
FA = "False alarms / series"  # column of the results tables
LOADTEST_USERS = 32  # simulated users of the load test (step 5.3)

# Order-of-magnitude assumptions of the "Sous le capot" tab, deliberately simple
FLEET_SERIES = 1_000_000  # series to monitor
CHECK_INTERVAL_S = 5 * 60  # each series is checked every 5 minutes
CASCADE_DISCARDED = 0.95  # share of series that the baseline rules out before Toto

# Main chain from left to right; the job storage sits under the queue, linked to the API and the worker.
# Streamlit's CSS forces a small font in the diagram, and the diagram is stretched to the page width:
# a compact layout (small font, small gaps) is what makes the text big on screen.
ARCHITECTURE_DOT = """
digraph {
    rankdir=LR;
    nodesep=0.15;
    ranksep=0.25;
    node [shape=box, style="rounded,filled", fillcolor="#f4f4f1", color="#c3c2b7", fontname="Helvetica",
          fontsize=10, margin="0.08,0.04", height=0.3];
    edge [color="#52514e", arrowsize=0.8];
    client [label="Client\\n(page de démo)"];
    api [label="API\\n(FastAPI)"];
    queue [label="File d'attente"];
    worker [label="Worker\\n(traitement par lots)"];
    toto [label="Toto", fillcolor="#e3e9fa", color="#2F5FD0"];
    storage [label="Stockage des jobs\\n(statuts et résultats)", shape=cylinder];
    client -> api -> queue -> worker -> toto;
    {rank=same; queue; storage;}
    api -> storage [dir=both];
    worker -> storage [dir=back];
}
"""


@st.cache_data
def load_data():
    """Values and identifiers of the 200 series (data/series.npz)."""
    values, ids, _ = load_series()
    return values, ids


@st.cache_data
def load_csv(name):
    """A results table of results/."""
    return pd.read_csv(RESULTS / name)


@st.cache_data
def load_cache():
    """Precomputed results of the demo series, used in offline mode."""
    return json.loads(CACHE_PATH.read_text())


def num(x, digits=1):
    """A number with a fixed number of decimals: 2.25 -> "2.2"."""
    return f"{x:.{digits}f}"


def pct(x):
    """A share written as a percentage: 0.8213 -> "82%"."""
    return f"{100 * x:.0f}%"


def approx(x, significant=2):
    """x rounded to a few significant digits, with thousands separators: 72505 -> "73,000" (or "70,000")."""
    digits = significant - 1 - int(np.floor(np.log10(abs(x))))  # decimals to keep (negative: round to tens...)
    rounded = round(x, digits)
    return f"{rounded:,.0f}" if digits <= 0 else num(rounded, digits)


def api_is_up():
    """True if GET /health answers within HEALTH_TIMEOUT_S."""
    try:
        return requests.get(f"{API_URL}/health", timeout=HEALTH_TIMEOUT_S).ok
    except requests.RequestException:
        return False


def demo_series(index):
    """History, future with spikes and spike positions of a demo series (same spikes as the figures)."""
    values, _ = load_data()
    history, future = split_history_future(values[index])
    spiked, positions = inject_spikes(future, history, seed=0)
    return history, spiked, positions


# --- Figures read from results/ ---

def equal_false_alarms():
    """Validation set, at an equal false-alarm rate: false alarms per series, Toto recall, classic recall.

    Tuned Toto (results/tuning_toto_samples.csv) against the classic method's k whose false alarms
    are the closest (results/tuning_baseline_fine.csv).
    """
    samples = load_csv("tuning_toto_samples.csv")
    toto = samples[samples["Samples"] == TOTO_SAMPLES_TUNED].iloc[0]
    grid = load_csv("tuning_baseline_fine.csv")
    classic = grid.loc[(grid[FA] - toto[FA]).abs().idxmin()]
    return toto[FA], toto["Recall"], classic["Recall"]


def compute_times():
    """Mean computation time per series on the test set, in ms: (Toto, classic method)."""
    final = load_csv("eval_test.csv")
    toto_ms = final.loc[final["Detector"].str.startswith("Tuned Toto"), "ms / series"].item()
    classic_ms = final.loc[final["Detector"].str.contains("default"), "ms / series"].item()
    return toto_ms, classic_ms


def load_test():
    """results/loadtest.csv, one row per batch size."""
    return load_csv("loadtest.csv").set_index("Batch")


# --- Overview tab ---

def render_overview_tab():
    values, _ = load_data()
    n_series = len(values)
    n_spikes = len(demo_series(next(iter(DEMO_SERIES)))[2])

    st.subheader("The problem")
    st.markdown("When an incident happens, for example a faulty deployment that makes an API's error rate spike, "
                "the on-call engineer must be alerted within minutes, without false alarms that would teach them "
                "to ignore alerts.")
    st.subheader("The scale constraint")
    st.markdown("Datadog monitors millions of time series. Training one model per series is not an option. "
                "Toto is a single pretrained model that works on any series, with no retraining (151M parameters, "
                "pretrained on about 2.36 trillion data points, 43% of them Datadog internal metrics).")

    st.subheader("What I built")
    steps = [
        ("Data", f"{n_series} real Datadog metrics (public BOOM dataset)"),
        ("Incidents", f"{n_spikes} simulated spikes per series"),
        ("Evaluation", f"classic statistical method vs. Toto, tuned on {n_series // 2} series, "
                       f"tested on {n_series - n_series // 2} others"),
        ("Serving", "asynchronous API with a bounded queue and batched inference"),
        ("Load test", f"{LOADTEST_USERS} concurrent clients (Locust)"),
    ]
    quality_col, scale_col = st.columns([3, 2])
    for group_col, label, first, last in ((quality_col, "Quality: does it detect well?", 0, 3),
                                          (scale_col, "Scale: does it hold the load?", 3, 5)):
        with group_col.container(border=True):
            st.markdown(f"**{label}**")
            for col, number in zip(st.columns(last - first), range(first, last)):
                name, text = steps[number]
                col.markdown(f"**{number + 1}. {name}**  \n{text}")

    st.subheader("Key figures")
    _, toto_recall, classic_recall = equal_false_alarms()
    toto_ms, classic_ms = compute_times()
    loadtest = load_test()
    batched = loadtest.loc[loadtest.index.max(), "Series / s"]  # biggest batch
    figures = [
        (f"{toto_recall / classic_recall:.0f}×", "more incidents caught at an equal false-alarm rate"),
        (f"≈{approx(toto_ms / classic_ms, 1)}×", "more compute per series than the classic method"),
        (f"≈{approx(batched, 1)} series/s", f"on a laptop CPU, {int(loadtest['Failures'].sum())} failures"),
    ]
    for col, (value, text) in zip(st.columns(3), figures):
        with col:
            st.metric(text, value, label_visibility="collapsed")
            st.markdown(text)  # under the number rather than as its label, so that it can wrap


# --- Live detection tab ---

def run_all_with_api(status_line):
    """Submits all the jobs at once (each demo series with each method), then polls them until done,
    like src/client.py.

    status_line: an st.empty() updated live with the number of jobs in each status.
    Returns ({index: {method: {"low", "high", "anomalies"}}}, seconds), or None if anything goes wrong
    (network error, failed job, more than JOB_TIMEOUT_S): the page then uses the cache.
    """
    _, ids = load_data()
    try:
        start = time.perf_counter()
        jobs = {}  # (index, method) -> job_id
        for index in DEMO_SERIES:
            history, spiked, _ = demo_series(index)
            series = [{"id": str(ids[index]), "history": history.tolist(), "recent": spiked.tolist()}]
            for method in METHODS:
                response = requests.post(f"{API_URL}/v1/detection-jobs", json={"method": method, "series": series},
                                         timeout=REQUEST_TIMEOUT_S)
                response.raise_for_status()
                jobs[(index, method)] = response.json()["job_id"]

        statuses, results = {}, {}
        while True:
            for key, job_id in jobs.items():
                if key in results:
                    continue
                response = requests.get(f"{API_URL}/v1/detection-jobs/{job_id}", timeout=REQUEST_TIMEOUT_S)
                response.raise_for_status()
                job = response.json()
                if job["status"] == "failed":
                    return None
                statuses[key] = job["status"]
                if job["status"] == "done":
                    results[key] = job["results"][0]
            seconds = time.perf_counter() - start
            counts = " · ".join(f"{list(statuses.values()).count(s)} {s}" for s in ("pending", "running", "done"))
            status_line.markdown(f"{len(jobs)} jobs submitted · {counts} · {seconds:.1f} s")
            if len(results) == len(jobs):
                return {index: {m: results[(index, m)] for m in METHODS} for index in DEMO_SERIES}, seconds
            if seconds > JOB_TIMEOUT_S:
                return None
            time.sleep(POLL_INTERVAL_S)
    except Exception:  # any problem with the API (stopped, unreachable, unexpected answer): use the cache
        return None


def detect_all(api_up):
    """Runs the detection of all the demo series, through the API if possible, otherwise from the cache."""
    outcome = None
    if api_up:
        slot = st.empty()  # live progress, removed once the run is over
        with slot.container():
            with st.status(f"Submitting {len(DEMO_SERIES) * len(METHODS)} jobs to the API…", expanded=True):
                outcome = run_all_with_api(st.empty())
        slot.empty()
    if outcome is None:
        cache = load_cache()
        return {"source": "cache", "results": {i: {m: cache[str(i)][m] for m in METHODS} for i in DEMO_SERIES}}
    results, seconds = outcome
    return {"source": "api", "results": results, "seconds": seconds}


def plot_series(series, positions, color=GREY, low=None, high=None, anomalies=()):
    """A demo chart drawn with draw() of plot_detectors.py, relabeled with the page vocabulary.

    The title is written above the chart by the page, where it stays readable at any width.
    Without low and high, only the metric and the simulated incidents are shown.
    Returns the figure, the number of incidents caught and the number of false alarms.
    """
    fig, ax = plt.subplots(figsize=(5, 2.6), facecolor=SURFACE)
    show_interval = low is not None
    if not show_interval:  # nothing to fill: the range and the alerts are left out of the legend below
        low = high = np.full(HORIZON, np.nan)
    anomalies = np.asarray(anomalies, dtype=int)
    caught, false_alarms = draw(ax, series, positions, np.asarray(low), np.asarray(high), anomalies, color, "")

    ax.set_title("", loc="left")  # draw() writes its title on the left
    ax.set_ylabel("value", color=GREY, fontsize=9)
    ax.set_xticklabels([])  # time step numbers mean nothing to the audience
    ax.tick_params(labelsize=8)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 0.12 * (top - bottom))  # room for the circles of the highest spikes
    # Above the shaded future area, so that it never hides a spike
    ax.text(CONTEXT + HORIZON / 2, 1.01, "analyzed window", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, color=GREY)
    handles, labels = ax.get_legend_handles_labels()
    kept = [(h, LEGEND_LABELS.get(lab, lab)) for h, lab in zip(handles, labels)
            if show_interval or lab not in ("forecast interval", "flagged point")]
    # Under the chart, so that it never hides the curve
    ax.legend([h for h, _ in kept], [lab for _, lab in kept], loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=len(kept), fontsize=8.5, frameon=False, handlelength=1.5, columnspacing=1.2)
    fig.tight_layout()
    return fig, caught, false_alarms


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)  # free the memory of the figure


def caught_and_false_alarms(anomalies, positions):
    return len(np.intersect1d(anomalies, positions)), len(np.setdiff1d(anomalies, positions))


def show_summary(run):
    """Source of the results, then one row per series: incidents caught and false alarms of both methods."""
    if run["source"] == "cache":
        st.caption("Offline mode: precomputed results")
    else:
        st.caption(f"Computed live by the API: {len(DEMO_SERIES) * len(METHODS)} jobs submitted · "
                   f"pending → running → done in {run['seconds']:.1f} s")
    rows = []
    for index, (name, _) in DEMO_SERIES.items():
        _, _, positions = demo_series(index)
        classic, toto = (caught_and_false_alarms(run["results"][index][m]["anomalies"], positions) for m in METHODS)
        rows.append({
            "Metric": name,
            "Incidents caught (classic / Toto)": f"{classic[0]} / {toto[0]}",
            "False alarms (classic / Toto)": f"{classic[1]} / {toto[1]}",
        })
    st.table(pd.DataFrame(rows), hide_index=True)


def show_series(index, run):
    """Name and explanation of a series, then both charts side by side, each with its counters."""
    name, explanation = DEMO_SERIES[index]
    history, spiked, positions = demo_series(index)
    series = np.concatenate([history, spiked])
    st.subheader(name)
    st.markdown(explanation)
    for col, (method, (_, color, title)) in zip(st.columns(2), METHODS.items()):
        result = run["results"][index][method]
        with col:
            st.markdown(f"**{title}**")
            fig, caught, false_alarms = plot_series(series, positions, color,
                                                    result["low"], result["high"], result["anomalies"])
            show_figure(fig)
            for counter, (label, value) in zip(st.columns(2), (("Incidents caught", f"{caught} / {len(positions)}"),
                                                               ("False alarms", false_alarms))):
                with counter:
                    st.markdown(label)  # above the number rather than as its label, so that it can wrap
                    st.metric(label, value, label_visibility="collapsed")


def show_previews():
    """Before any run: each metric with its simulated incidents only, two per row."""
    indices = list(DEMO_SERIES)
    for pair in (indices[:2], indices[2:]):
        for col, index in zip(st.columns(2), pair):
            history, spiked, positions = demo_series(index)
            with col:
                st.markdown(f"**{DEMO_SERIES[index][0]}**")
                fig, _, _ = plot_series(np.concatenate([history, spiked]), positions)
                show_figure(fig)


def render_live_tab(api_up):
    """The live detection tab; returns True if the API stopped answering during a run."""
    st.markdown("Four real metrics, three simulated incidents each. Both detectors are called through the API, "
                "exactly like a production client would.")
    api_lost = False
    if st.button(f"Run detection on all {len(DEMO_SERIES)} metrics", type="primary"):
        # Kept in the session: the last results stay on screen until the next run
        st.session_state["demo_run"] = detect_all(api_up)
        api_lost = api_up and st.session_state["demo_run"]["source"] == "cache"

    run = st.session_state.get("demo_run")
    if run is None:
        show_previews()
    else:
        show_summary(run)
        for index in DEMO_SERIES:
            show_series(index, run)
    st.caption(f"Illustrative examples. Aggregate results across {len(load_data()[0])} series are in the Results "
               "tab.")
    return api_lost


# --- Results tab ---

def method_name(detector):
    """French name of a detector of results/eval_test.csv."""
    if detector.startswith("Tuned Toto"):
        return "Toto"
    if "default" in detector:
        return "Méthode classique (réglage par défaut)"
    return "Méthode classique (réglée sur les fausses alertes de Toto)"


def render_results_tab():
    # Equal false alarms, on the validation set: tuned Toto, and the baseline k with the closest false alarms
    samples = load_csv("tuning_toto_samples.csv")
    toto = samples[samples["Samples"] == TOTO_SAMPLES_TUNED].iloc[0]
    baseline_grid = load_csv("tuning_baseline_fine.csv")
    baseline = baseline_grid.loc[(baseline_grid[FA] - toto[FA]).abs().idxmin()]
    st.subheader(f"À nombre de fausses alertes égal ({num(toto[FA])} par série), Toto détecte "
                 f"{pct(toto['Recall'])} des incidents, contre {pct(baseline['Recall'])} pour la méthode classique.")

    st.image(str(RESULTS / "recall_vs_false_alarms.png"), width=900)
    st.caption("Horizontalement : les fausses alertes par série. Verticalement : la part des incidents détectés. "
               "Chaque point est un réglage : plus la courbe est haute, meilleure est la méthode.")

    final = load_csv("eval_test.csv")
    table = pd.DataFrame({
        "Méthode": final["Detector"].map(method_name),
        "Incidents détectés": final["Recall"].map(pct),
        "Alertes justes": final["Precision"].map(pct),
        "Fausses alertes par série": final[FA].map(num),
    })
    st.table(table, hide_index=True)  # static table: text wraps, no scrollbar or toolbar on screen
    st.caption("Réglages choisis sur 100 séries, résultats vérifiés sur 100 autres séries jamais vues.")


# --- Under the hood tab ---

def render_inside_tab():
    st.subheader("L'architecture")
    st.graphviz_chart(ARCHITECTURE_DOT)

    final = load_csv("eval_test.csv")
    toto_ms = final.loc[final["Detector"].str.startswith("Tuned Toto"), "ms / series"].item()
    baseline_ms = final.loc[final["Detector"].str.contains("default"), "ms / series"].item()
    loadtest = load_csv("loadtest.csv").set_index("Batch")["Series / s"]
    batched = loadtest[loadtest.index.max()]  # series per second with the biggest batch (16)

    st.subheader("Les chiffres")
    figures = {
        "Coût de Toto par rapport à la méthode classique": f"≈ {approx(toto_ms / baseline_ms)}×",
        "Gain de débit grâce aux lots": f"+{pct(batched / loadtest[1] - 1)}",
        f"Débit sous {LOADTEST_USERS} clients simultanés": f"{num(batched)} séries/s",
    }
    # Bottom alignment: the numbers stay on one line even when a label wraps
    for col, (label, value) in zip(st.columns(3, vertical_alignment="bottom"), figures.items()):
        with col:
            st.markdown(label)  # above the number rather than as its label, so that it can wrap
            st.metric(label, value, label_visibility="collapsed")

    st.subheader("Et à grande échelle ?")
    needed = FLEET_SERIES / CHECK_INTERVAL_S  # series per second
    machines = needed / batched
    st.markdown(
        f"{FLEET_SERIES / 1e6:g} million de séries toutes les {CHECK_INTERVAL_S // 60} minutes "
        f"= environ {approx(needed)} séries/s  \n"
        f"→ environ {approx(machines)} machines CPU à ce débit  \n"
        f"→ environ {approx(machines * (1 - CASCADE_DISCARDED))} avec une cascade où la méthode classique "
        f"écarte {pct(CASCADE_DISCARDED)} des séries."
    )
    st.caption("Ordre de grandeur, avec des hypothèses volontairement simples. Le débit vient du test de charge, "
               "avec une version allégée de Toto, sur un ordinateur portable.")


# --- Page ---

st.set_page_config(page_title="Incident detection with Toto", layout="wide")
st.title("Detecting incidents across millions of metrics")
st.markdown("An end-to-end study of Toto, Datadog AI Research's open-source forecasting model: detection quality, "
            "serving architecture and cost at scale.")
st.caption("Elias Massaro · MVA (ENS Paris-Saclay) · Mines Nancy")

api_status = st.sidebar.empty()  # a placeholder, updated if the API stops during a detection


def show_api_status(up):
    with api_status:
        if up:
            st.success("API connected")
        else:
            st.info("Offline mode: precomputed results")


api_up = api_is_up()
show_api_status(api_up)

overview_tab, live_tab, results_tab, architecture_tab, takeaways_tab = st.tabs(
    ["Overview", "Live detection", "Results", "Architecture & scalability", "Takeaways"])
with overview_tab:
    render_overview_tab()
with live_tab:
    if render_live_tab(api_up):
        show_api_status(False)
with results_tab:
    render_results_tab()
with architecture_tab:
    render_inside_tab()

st.divider()
st.caption("Data: public BOOM dataset (Datadog). Model: Toto, open source (Datadog AI Research).")
