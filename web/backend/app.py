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

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
# Módulos hermanos (cds.py, fhir_client.py), tanto con `uvicorn web.backend.app`
# como importando `app` directamente desde las pruebas.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field, model_validator  # noqa: E402

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
from src.audit import AuditLog, completeness  # noqa: E402
from src.baseline import RuleBasedBaseline  # noqa: E402
from src.ingest import load_corpus  # noqa: E402
from src.pipeline import CopilotoPipeline  # noqa: E402

import cds  # noqa: E402
import fhir_client  # noqa: E402

app = FastAPI(
    title="TDSE Copiloto de Triage RAG - API",
    description="API opcional que expone el pipeline de triage a la interfaz web.",
    version="1.0.0",
)

# Orígenes por defecto de Vite (dev) y de un build servido localmente. En
# despliegue, el frontend (repositorio aparte, en Vercel) se agrega con:
#   CORS_ORIGINS="https://mi-app.vercel.app,https://otro-dominio"   (exactos)
#   CORS_ORIGIN_REGEX="https://mi-app-.*\.vercel\.app"         (previews)
_DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"]
_extra_origins = [o.strip().rstrip("/") for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _extra_origins,
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX") or None,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro v2 (hash encadenado) de la API web.
DEFAULT_AUDIT_PATH = ROOT / "results" / "audit_web_v2.jsonl"
_baseline = RuleBasedBaseline()


@lru_cache(maxsize=None)
def _pipeline(corpus_format: str, embedding_mode: str, backend: str, audit_path: str) -> CopilotoPipeline:
    """Un pipeline por configuración (construirlo carga el corpus y la caché
    de embeddings; no tiene sentido repetirlo en cada consulta)."""
    return CopilotoPipeline(
        corpus_format=corpus_format, embedding_mode=embedding_mode, backend=backend, audit_path=Path(audit_path)
    )


@lru_cache(maxsize=None)
def _chunk_texts(corpus_format: str) -> dict[str, str]:
    return {c.chunk_id: c.text for c in load_corpus(corpus_format)}


def _fragment_text(citation: Optional[str], corpus_format: str) -> Optional[str]:
    """Texto íntegro del fragmento citado. Las citas de las reglas de alarma
    son ids semánticos del corpus reformateado (el crudo usa ids
    posicionales), así que se buscan también allí."""
    if citation is None:
        return None
    return _chunk_texts(corpus_format).get(citation) or _chunk_texts("reformatted").get(citation)


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


class AlarmOut(BaseModel):
    rule_id: str
    level: int
    citation: str
    matched_pattern: str
    text: Optional[str] = None


class ConsultaResponse(BaseModel):
    suggestion: SuggestionOut
    retrieved: list[RetrievedFragment]
    baseline: BaselineOut
    backend_used: str
    record_hash: str
    degraded: bool
    escalated: bool
    n_modelo: Optional[int]
    n_reglas: Optional[int]
    n_final: Optional[int]
    model_citation: Optional[str]
    alarms: list[AlarmOut]
    # Texto íntegro del fragmento citado por la sugerencia final (o, en una
    # abstención con alarma, el de la alarma de referencia).
    cited_fragment_text: Optional[str]
    disclaimer: str = (
        "Sugerencia de apoyo. La clasificación final es responsabilidad del "
        "profesional de salud."
    )


class DecisionRequest(BaseModel):
    record_hash: str = Field(min_length=64, max_length=64)
    action: Literal["confirma", "corrige"]
    final_level: Optional[int] = Field(default=None, ge=1, le=5)
    reason: Optional[str] = None
    review_seconds: float = Field(ge=0)
    user: Optional[str] = None

    @model_validator(mode="after")
    def _correction_needs_level_and_reason(self):
        if self.action == "corrige":
            if self.final_level is None:
                raise ValueError("'corrige' exige final_level (1-5).")
            if not self.reason or not self.reason.strip():
                raise ValueError("'corrige' exige un motivo.")
        return self


class DecisionResponse(BaseModel):
    event_hash: str
    ref_hash: str
    action: str
    final_level: int
    suggested_level: Optional[int]
    fhir: dict = Field(default_factory=lambda: {"enabled": False})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/consulta", response_model=ConsultaResponse)
def consulta(req: ConsultaRequest) -> ConsultaResponse:
    texto = req.texto.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="El relato no puede estar vacío.")
    return _run_consulta(texto, req.corpus_format, req.embedding_mode, req.backend, case_id="web")


def _run_consulta(texto: str, corpus_format: str, embedding_mode: str, backend: str, case_id: str) -> ConsultaResponse:
    pipeline = _pipeline(corpus_format, embedding_mode, backend, str(DEFAULT_AUDIT_PATH))
    suggestion = pipeline.run_case(case_id, texto)
    record = pipeline.last_record
    baseline_result = _baseline.classify(texto)

    reference_citation = suggestion.citation
    if reference_citation is None and suggestion.alarms:
        reference_citation = min(suggestion.alarms, key=lambda a: a["level"])["citation"]

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
        backend_used=suggestion.backend_used,
        record_hash=record["hash"],
        degraded=suggestion.degraded,
        escalated=suggestion.escalated,
        n_modelo=suggestion.n_modelo,
        n_reglas=suggestion.n_reglas,
        n_final=suggestion.n_final,
        model_citation=suggestion.model_citation,
        alarms=[
            AlarmOut(**a, text=_fragment_text(a["citation"], corpus_format)) for a in suggestion.alarms
        ],
        cited_fragment_text=_fragment_text(reference_citation, corpus_format),
    )


@app.post("/api/decision", response_model=DecisionResponse)
def decision(req: DecisionRequest) -> DecisionResponse:
    """Registra la decisión del profesional como un evento nuevo, encadenado,
    que referencia la sugerencia (nunca reescribe el evento original)."""
    log = AuditLog(DEFAULT_AUDIT_PATH)
    try:
        event = log.append_decision(
            req.record_hash,
            req.action,
            final_level=req.final_level,
            reason=req.reason,
            user=req.user,
            review_seconds=req.review_seconds,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="No existe una sugerencia con ese record_hash.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    # Escritura opcional en FHIR R4 (FHIR_ENABLED=false por defecto). Nunca
    # falla la petición: la auditoría local ya quedó registrada.
    fhir = fhir_client.write_decision(log.find(req.record_hash), event)
    return DecisionResponse(
        event_hash=event["hash"],
        ref_hash=event["ref_hash"],
        action=event["action"],
        final_level=event["final_level"],
        suggested_level=event["suggested_level"],
        fhir=fhir,
    )


# --- CDS Hooks (https://cds-hooks.org) ---------------------------------------


class CDSRequest(BaseModel):
    hook: str = cds.HOOK
    hookInstance: Optional[str] = None
    context: dict


@app.get("/cds-services")
def cds_discovery() -> dict:
    return cds.DISCOVERY


@app.post(f"/cds-services/{cds.SERVICE_ID}")
def cds_triage(req: CDSRequest) -> dict:
    """Recibe el relato en ``context.relato`` y responde con una tarjeta.
    Usa la ruta principal por defecto (con respaldo marcado) y deja el
    evento en la misma auditoría que la API web."""
    texto = str(req.context.get("relato") or "").strip()
    if not texto:
        raise HTTPException(status_code=400, detail="context.relato es obligatorio.")
    case_id = str(req.context.get("encounterId") or "cds-hooks")
    response = _run_consulta(texto, "reformatted", "clinical_es", "gemini", case_id=case_id)
    return {"cards": [cds.build_card(response.model_dump())]}


@app.get("/api/auditoria")
def auditoria(limit: int = Query(50, ge=1, le=1000)) -> dict:
    """Últimos eventos del registro web, estado de la cadena de hashes y
    completitud del registro (métrica del paper)."""
    records = AuditLog(DEFAULT_AUDIT_PATH).read_all()
    ok, first_bad = AuditLog(DEFAULT_AUDIT_PATH).verify_chain()
    return run_experiment._nan_to_none(
        {
            "total": len(records),
            "events": records[-limit:],
            "cadena_valida": ok,
            "primer_evento_invalido": first_bad,
            "completitud": completeness(records),
        }
    )


@app.get("/api/experimento")
def experimento() -> dict:
    """Mismo cómputo que `python -m experiments.run_experiment`, en JSON
    crudo en vez de Markdown, para que el frontend pueda graficarlo."""
    cases = run_experiment.load_cases()
    output = run_experiment.run_all(cases, gemini_first_mode="readonly")
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
