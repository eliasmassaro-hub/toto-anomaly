"""Step D2: demo page for the interview (Streamlit).

The page shows a non-technical audience how the baseline method and Toto find simulated incidents
on a real metric. It calls the API like a real client; if the API cannot be reached, it uses the
precomputed results of results/demo_cache.json (offline mode), so the demo never crashes.
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
from plot_data import BLUE, GREY, RESULTS, SURFACE, ZOOM
from plot_detectors import draw

API_URL = os.environ.get("API_URL", "http://localhost:8000")
HEALTH_TIMEOUT_S = 1
REQUEST_TIMEOUT_S = 5  # per HTTP request
POLL_INTERVAL_S = 0.2
JOB_TIMEOUT_S = 30  # beyond this, the page falls back to the cache
CACHE_PATH = RESULTS / "demo_cache.json"  # written by src/demo_precompute.py

# Demo series: index -> (displayed name, explanation, history points shown). The first one is selected
# by default. Series 42 shows 450 points so that the previous hourly bump (~360 points earlier) is visible.
DEMO_SERIES = {
    42: ("Tâche planifiée toutes les heures", "Une bosse revient environ toutes les heures : c'est normal.", 450),
    0: ("Trafic cyclique", "Un cycle régulier, avec des pics naturels.", 200),
    3: ("Changement de niveau", "La métrique change de niveau juste avant la fenêtre analysée.", 200),
    7: ("Sauts réguliers", "Des sauts courts et réguliers, difficiles à prévoir à la minute près.", 200),
}
# API method -> (displayed name, color, chart title); colors are the same as in the interview slides
METHODS = {
    "baseline": ("Méthode classique", "#B34A12", "Méthode classique : zone normale fixe"),
    "toto": ("Toto", "#2F5FD0", "Toto : zone normale prévue point par point"),
}
# draw() writes English legend labels: they are replaced with the page vocabulary
LEGEND_LABELS = {
    "forecast interval": "zone normale prévue",
    "series": "métrique",
    "injected spike": "incident simulé",
    "flagged point": "alerte",
}
JOB_STATUSES = {"pending": "En attente", "running": "En cours"}
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


def fr(x, digits=1):
    """A number written the French way: 2.5 -> "2,5"."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x):
    """A share written as a French percentage: 0.8213 -> "82 %" (non-breaking space)."""
    return f"{100 * x:.0f} %"


def approx(x):
    """x rounded to 2 significant digits, with French thousands separators: 72505 -> "73 000"."""
    digits = 1 - int(np.floor(np.log10(abs(x))))  # decimals to keep (negative: round to tens, hundreds...)
    rounded = round(x, digits)
    return f"{rounded:,.0f}".replace(",", " ") if digits <= 0 else fr(rounded, digits)


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


# --- Live detection tab ---

def run_with_api(index, history, spiked, status_lines):
    """Creates a baseline job and a toto job, then polls both until done, like src/client.py.

    status_lines: one st.empty() per method, updated live with the job status.
    Returns {method: {"low", "high", "anomalies", "seconds"}}, or None if anything goes wrong
    (network error, failed job, more than JOB_TIMEOUT_S): the page then uses the cache.
    """
    _, ids = load_data()
    series = [{"id": str(ids[index]), "history": history.tolist(), "recent": spiked.tolist()}]
    try:
        start = time.perf_counter()
        jobs = {}
        for method in METHODS:
            response = requests.post(f"{API_URL}/v1/detection-jobs", json={"method": method, "series": series},
                                     timeout=REQUEST_TIMEOUT_S)
            response.raise_for_status()
            jobs[method] = response.json()["job_id"]

        results = {}
        while len(results) < len(jobs):
            for method, job_id in jobs.items():
                if method in results:
                    continue
                response = requests.get(f"{API_URL}/v1/detection-jobs/{job_id}", timeout=REQUEST_TIMEOUT_S)
                response.raise_for_status()
                job = response.json()
                if job["status"] == "failed":
                    return None
                if job["status"] == "done":
                    seconds = time.perf_counter() - start
                    results[method] = {**job["results"][0], "seconds": seconds}
                    text = f"Terminé en {fr(seconds)} s"
                else:
                    text = JOB_STATUSES.get(job["status"], job["status"])
                status_lines[method].markdown(f"**{METHODS[method][0]}** : {text}")
            if len(results) < len(jobs):
                if time.perf_counter() - start > JOB_TIMEOUT_S:
                    return None
                time.sleep(POLL_INTERVAL_S)
        return results
    except Exception:  # any problem with the API (stopped, unreachable, unexpected answer): use the cache
        return None


def detect(index, api_up):
    """Runs the detection of a demo series, through the API if possible, otherwise from the cache."""
    results = None
    if api_up:
        history, spiked, _ = demo_series(index)
        with st.status("Envoi de la série à l'API…", expanded=True) as status:
            status_lines = {method: st.empty() for method in METHODS}
            for method, line in status_lines.items():
                line.markdown(f"**{METHODS[method][0]}** : envoi…")
            results = run_with_api(index, history, spiked, status_lines)
            label = "Détection terminée" if results else "L'API ne répond plus : résultats précalculés"
            status.update(label=label, state="complete", expanded=results is not None)
    if results is None:
        cached = load_cache()[str(index)]
        return {"index": index, "source": "cache", "results": {method: cached[method] for method in METHODS}}
    return {"index": index, "source": "api", "results": results}


def plot_series(series, positions, title, shown, color=GREY, low=None, high=None, anomalies=()):
    """A demo chart drawn with draw() of plot_detectors.py, with French labels and bigger text.

    shown: number of history points shown before the analyzed window (draw() shows ZOOM of them).
    Without low and high, only the metric and the simulated incidents are shown.
    Returns the figure, the number of incidents found and the number of false alarms.
    """
    fig, ax = plt.subplots(figsize=(11, 3.4), facecolor=SURFACE)
    show_interval = low is not None
    if not show_interval:  # nothing to fill: the interval and the alerts are left out of the legend below
        low = high = np.full(HORIZON, np.nan)
    anomalies = np.asarray(anomalies, dtype=int)
    hits, false_alarms = draw(ax, series, positions, np.asarray(low), np.asarray(high), anomalies, color, title)

    start = CONTEXT - shown
    if shown > ZOOM:  # draw() starts the line ZOOM points before the window: the earlier part is added here
        ax.plot(np.arange(start, CONTEXT - ZOOM + 1), series[start:CONTEXT - ZOOM + 1], color=BLUE, lw=1.2)
    ax.set_xlim(start, len(series) - 1)

    ax.set_title(title, color=GREY, fontsize=13, loc="left")
    ax.set_ylabel("valeur", color=GREY, fontsize=10)
    ax.set_xlabel("temps", color=GREY, fontsize=10)
    ax.tick_params(labelsize=9)
    # Above the shaded future area, so that it never hides a spike
    ax.text(CONTEXT + HORIZON / 2, 1.01, "fenêtre analysée", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=10, color=GREY)
    handles, labels = ax.get_legend_handles_labels()
    kept = [(h, LEGEND_LABELS.get(lab, lab)) for h, lab in zip(handles, labels)
            if show_interval or lab not in ("forecast interval", "flagged point")]
    ax.legend([h for h, _ in kept], [lab for _, lab in kept], loc="upper left", fontsize=10, frameon=False)
    fig.tight_layout()
    return fig, hits, false_alarms


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)  # free the memory of the figure


def show_detection(index, detection):
    """The two charts (baseline then Toto), each with its two counters."""
    history, spiked, positions = demo_series(index)
    series = np.concatenate([history, spiked])
    if detection["source"] == "cache":
        st.caption("Mode hors ligne : résultats précalculés")
    else:
        times = " ; ".join(f"{METHODS[m][0]} : {fr(r['seconds'])} s" for m, r in detection["results"].items())
        st.caption(f"Résultats calculés en direct par l'API ({times}).")

    for method, (_, color, title) in METHODS.items():
        result = detection["results"][method]
        fig, hits, false_alarms = plot_series(series, positions, title, DEMO_SERIES[index][2], color,
                                              result["low"], result["high"], result["anomalies"])
        chart_col, counters_col = st.columns([4, 1], vertical_alignment="center")
        with chart_col:
            show_figure(fig)
        with counters_col:
            st.metric("Incidents retrouvés", f"{hits} / {len(positions)}")
            st.metric("Fausses alertes", false_alarms)


def render_live_tab(api_up):
    names = {name: index for index, (name, _, _) in DEMO_SERIES.items()}
    name = st.selectbox("Métrique", list(names))
    index = names[name]
    st.markdown(DEMO_SERIES[index][1])

    if st.button("Lancer la détection", type="primary"):
        # Kept in the session: the last result stays on screen until the next launch
        st.session_state["detection"] = detect(index, api_up)

    detection = st.session_state.get("detection")
    if detection is not None and detection["index"] == index:
        show_detection(index, detection)
    else:  # before any detection: the metric with its simulated incidents only
        history, spiked, positions = demo_series(index)
        fig, _, _ = plot_series(np.concatenate([history, spiked]), positions,
                                f"La métrique, avec {len(positions)} incidents simulés", DEMO_SERIES[index][2])
        show_figure(fig)


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
    st.subheader(f"À nombre de fausses alertes égal ({fr(toto[FA])} par série), Toto détecte "
                 f"{pct(toto['Recall'])} des incidents, contre {pct(baseline['Recall'])} pour la méthode classique.")

    st.image(str(RESULTS / "recall_vs_false_alarms.png"), width=900)
    st.caption("Horizontalement : les fausses alertes par série. Verticalement : la part des incidents détectés. "
               "Chaque point est un réglage : plus la courbe est haute, meilleure est la méthode.")

    final = load_csv("eval_test.csv")
    table = pd.DataFrame({
        "Méthode": final["Detector"].map(method_name),
        "Incidents détectés": final["Recall"].map(pct),
        "Alertes justes": final["Precision"].map(pct),
        "Fausses alertes par série": final[FA].map(fr),
    })
    st.dataframe(table, hide_index=True)
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
        f"Débit sous {LOADTEST_USERS} clients simultanés": f"{fr(batched)} séries/s",
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

st.set_page_config(page_title="Détection d'incidents avec Toto", layout="wide")
st.title("Détecter les incidents sur des métriques")
st.markdown("Une vraie métrique, trois incidents simulés : la méthode classique et Toto doivent les retrouver, "
            "sans fausses alertes.")

api_up = api_is_up()
with st.sidebar:
    if api_up:
        st.success("API connectée")
    else:
        st.info("Mode hors ligne : résultats précalculés")

live_tab, results_tab, inside_tab = st.tabs(["Détection en direct", "Résultats sur 200 séries", "Sous le capot"])
with live_tab:
    render_live_tab(api_up)
with results_tab:
    render_results_tab()
with inside_tab:
    render_inside_tab()

st.divider()
st.caption("Données : dataset public BOOM (Datadog). Modèle : Toto, open source (Datadog AI Research).")
