"""Step D2: demo page for the interview (Streamlit).

The page shows a non-technical audience how the baseline method and Toto find simulated incidents
on a real metric. It calls the API like a real client; if the API cannot be reached, it uses the
precomputed results of results/demo_cache.json (offline mode), so the demo never crashes.
"""

import os

import pandas as pd
import requests
import streamlit as st

from data import load_series
from plot_data import RESULTS

API_URL = os.environ.get("API_URL", "http://localhost:8000")
HEALTH_TIMEOUT_S = 1


@st.cache_data
def load_data():
    """Values and identifiers of the 200 series (data/series.npz)."""
    values, ids, _ = load_series()
    return values, ids


@st.cache_data
def load_csv(name):
    """A results table of results/."""
    return pd.read_csv(RESULTS / name)


def api_is_up():
    """True if GET /health answers within HEALTH_TIMEOUT_S."""
    try:
        return requests.get(f"{API_URL}/health", timeout=HEALTH_TIMEOUT_S).ok
    except requests.RequestException:
        return False


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

st.divider()
st.caption("Données : dataset public BOOM (Datadog). Modèle : Toto, open source (Datadog AI Research).")
