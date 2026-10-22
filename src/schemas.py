"""Formats des données échangées avec l'API."""

from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat

from data import CONTEXT, HORIZON  # 1024 et 64


class SeriesIn(BaseModel):
    """Une série envoyée par le client."""

    id: str
    # FiniteFloat refuse NaN et Infinity, que le JSON de Python accepte : sinon le calcul
    # produirait des NaN, impossibles à renvoyer en JSON
    history: list[FiniteFloat] = Field(min_length=CONTEXT)  # au moins 1024 points
    recent: list[FiniteFloat] = Field(min_length=1, max_length=HORIZON)  # entre 1 et 64 points


class JobRequest(BaseModel):
    """Le corps du POST /v1/detection-jobs."""

    method: Literal["toto", "baseline"] = "toto"
    series: list[SeriesIn] = Field(min_length=1, max_length=100)
