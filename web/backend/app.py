"""API REST opcional que expone el pipeline existente (src/, experiments/) a
la interfaz web (web/frontend/). No contiene lógica de negocio propia: cada
endpoint llama directamente a las mismas funciones que ya usan `demo.py`,
`gui.py` y `experiments/*.py`, para no duplicar ni divergir del pipeline
principal (ver README.md, sección "Arquitectura").

Esta capa SÍ tiene dependencias de terceros (FastAPI, uvicorn) a propósito
aislado en `web/backend/requirements.txt`: el pipeline principal
(`src/`, `experiments/`, `demo.py`, `gui.py`) sigue funcionando sin ninguna
dependencia externa, con o sin esta API.

Uso:
    pip install -r web/backend/requirements.txt
    uvicorn web.backend.app:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from experiments import run_experiment  # noqa: E402
from experiments.analyze_external_data import (  # noqa: E402
    DATA_PATH as EXTERNAL_DATA_PATH,
)
from experiments.analyze_external_data import (  # noqa: E402
    distribution_by_group,
    independence_test_by_group,
    level_distribution,
    load_rows,
    time_to_attention_minutes,
)
from src.baseline import RuleBasedBaseline  # noqa: E402
from src.ingest import load_corpus  # noqa: E402
from src.pipeline import CopilotoPipeline  # noqa: E402

app = FastAPI(
    title="TDSE Copiloto de Triage RAG - API",
    description="API opcional que expone el pipeline de triage a la interfaz web.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # Orígenes por defecto de Vite (dev) y de un build servido localmente.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_AUDIT_PATH = ROOT / "results" / "audit_web.jsonl"
_baseline = RuleBasedBaseline()


class ConsultaRequest(BaseModel):
    texto: str
    corpus_format: Literal["raw", "reformatted"] = "reformatted"
    embedding_mode: Literal["generic", "clinical_es"] = "clinical_es"
    backend: Literal["gemini", "deterministic"] = "gemini"


class RetrievedFragment(BaseModel):
    chunk_id: str
    score: float
    text: str
    level: Optional[int]


class SuggestionOut(BaseModel):
    level: Optional[int]
    citation: Optional[str]
    confidence: float
    abstained: bool
    reason: str


class BaselineOut(BaseModel):
    level: int
    citation: str
    matched_keyword: Optional[str]


class ConsultaResponse(BaseModel):
    suggestion: SuggestionOut
    retrieved: list[RetrievedFragment]
    baseline: BaselineOut
    backend_used: str
    disclaimer: str = (
        "Esta sugerencia asiste, no decide. La clasificación final es "
        "responsabilidad del personal de salud."
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/consulta", response_model=ConsultaResponse)
def consulta(req: ConsultaRequest) -> ConsultaResponse:
    texto = req.texto.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="El relato no puede estar vacío.")

    pipeline = CopilotoPipeline(
        corpus_format=req.corpus_format,
        embedding_mode=req.embedding_mode,
        backend=req.backend,
        audit_path=DEFAULT_AUDIT_PATH,
    )
    suggestion = pipeline.run_case("web", texto)
    backend_used = suggestion.backend_used
    baseline_result = _baseline.classify(texto)

    return ConsultaResponse(
        suggestion=SuggestionOut(
            level=suggestion.level,
            citation=suggestion.citation,
            confidence=suggestion.confidence,
            abstained=suggestion.abstained,
            reason=suggestion.reason,
        ),
        retrieved=[
            RetrievedFragment(
                chunk_id=rc.chunk.chunk_id, score=rc.score, text=rc.chunk.text, level=rc.chunk.level
            )
            for rc in suggestion.retrieved
        ],
        baseline=BaselineOut(
            level=baseline_result.level,
            citation=baseline_result.citation,
            matched_keyword=baseline_result.matched_keyword,
        ),
        backend_used=backend_used,
    )


@app.get("/api/experimento")
def experimento() -> dict:
    """Mismo cómputo que `python -m experiments.run_experiment`, en JSON
    crudo en vez de Markdown, para que el frontend pueda graficarlo."""
    cases = run_experiment.load_cases()
    output = run_experiment.run_all(cases)
    return run_experiment._nan_to_none(
        {
            "kappa": output["kappa_intra_equipo"],
            "baseline": output["baseline_C0"],
            "cells": output["celdas"],
            "n_total_cases": len(cases),
        }
    )


@app.get("/api/dataset")
def dataset() -> dict:
    """Mismo cómputo que `python -m experiments.analyze_external_data`, en
    JSON crudo. Ver README.md, sección 'Fuentes de datos reales'."""
    if not EXTERNAL_DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró {EXTERNAL_DATA_PATH.name}. Ver README.md.",
        )
    rows = load_rows()
    distribution = level_distribution(rows)
    durations = time_to_attention_minutes(rows)
    by_red = {red: dict(counts) for red, counts in distribution_by_group(rows, "red").items()}
    red_test = independence_test_by_group(rows, "red")

    duration_stats = {}
    for level, values in durations.items():
        if not values:
            continue
        sorted_values = sorted(values)
        n = len(sorted_values)
        duration_stats[level] = {
            "n": n,
            "median": sorted_values[n // 2],
            "mean": sum(values) / n,
        }

    return {
        "total": len(rows),
        "distribution": dict(distribution),
        "duration_stats": duration_stats,
        "by_red": by_red,
        "chi2_independence_test": {
            "chi2": red_test["chi2"],
            "df": red_test["df"],
            "p_value": red_test["p_value"],
        },
    }


@app.get("/api/casos")
def casos() -> dict:
    """Devuelve el conjunto de 74 viñetas del gold standard, tal como están
    en data/cases.json, para que el frontend pueda explorarlas."""
    cases = run_experiment.load_cases()
    return {"total": len(cases), "cases": cases}


@app.get("/api/corpus")
def corpus(corpus_format: Literal["raw", "reformatted"] = "reformatted") -> dict:
    """Devuelve los fragmentos citables del corpus normativo."""
    chunks = load_corpus(corpus_format)
    return {
        "corpus_format": corpus_format,
        "total": len(chunks),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "level": c.level,
            }
            for c in chunks
        ],
    }
