"""Formats of the data exchanged with the API."""

from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat

from data import CONTEXT, HORIZON  # 1024 and 64


class SeriesIn(BaseModel):
    """A series sent by the client."""

    id: str
    # FiniteFloat rejects NaN and Infinity, which Python's JSON accepts: otherwise the computation
    # would produce NaN, which cannot be returned as JSON
    history: list[FiniteFloat] = Field(min_length=CONTEXT)  # at least 1024 points
    recent: list[FiniteFloat] = Field(min_length=1, max_length=HORIZON)  # between 1 and 64 points


class JobRequest(BaseModel):
    """The body of POST /v1/detection-jobs."""

    method: Literal["toto", "baseline"] = "toto"
    series: list[SeriesIn] = Field(min_length=1, max_length=100)
