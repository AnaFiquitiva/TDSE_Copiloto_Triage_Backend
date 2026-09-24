"""Registro de auditoría v2 (Sección 4.3 del documento del proyecto).

Cada sugerencia generada por el pipeline queda registrada con el relato de
entrada, los fragmentos recuperados, la versión exacta del corpus (hash), de
los modelos y del prompt, los umbrales vigentes y la sugerencia final
(incluida la abstención). El registro es un archivo JSON Lines de solo
anexado, con **hash encadenado**:

    hash = sha256(prev_hash + json_canónico(registro_sin_hash))

donde el primer evento encadena con ``GENESIS_HASH``. Modificar, borrar o
reordenar cualquier evento rompe la cadena a partir de ese punto, y
``verify_chain()`` lo detecta.

La decisión del personal de salud NUNCA reescribe el evento original: se
anexa como un evento nuevo de tipo ``"decision"`` que referencia el hash de
la sugerencia (``append_decision``). El campo ``human_decision`` del evento
original se conserva en ``null`` a propósito.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .generator import Suggestion

AUDIT_SCHEMA_VERSION = "2"
GENESIS_HASH = "0" * 64

# Identificador del generador determinista (ruta de respaldo y celdas E1-E4).
DETERMINISTIC_GENERATOR_VERSION = "minimal-generator-v1"

EVENT_SUGGESTION = "sugerencia"
EVENT_DECISION = "decision"
DECISION_ACTIONS = ("confirma", "corrige")

# Campos que todo evento de sugerencia debe tener (la clave debe existir).
SUGGESTION_FIELDS = (
    "schema_version",
    "event_type",
    "timestamp_utc",
    "case_id",
    "input_text",
    "corpus_format",
    "corpus_hash",
    "embedding_mode",
    "backend_used",
    "degraded",
    "generation_model",
    "embedding_model",
    "prompt_version",
    "thresholds",
    "retrieved_fragments",
    "suggested_level",
    "citation",
    "confidence",
    "abstained",
    "abstain_reason",
    "use_guardrail",
    "alarms",
    "n_modelo",
    "n_reglas",
    "n_final",
    "escalated",
    "human_decision",
    "prev_hash",
    "hash",
)
# Subconjunto que además no puede ser null. Los demás pueden serlo de forma
# legítima (p. ej. suggested_level en una abstención); esos casos se validan
# con las reglas condicionales de `_missing_fields`.
SUGGESTION_NON_NULL = (
    "schema_version",
    "event_type",
    "timestamp_utc",
    "case_id",
    "input_text",
    "corpus_format",
    "corpus_hash",
    "embedding_mode",
    "backend_used",
    "degraded",
    "generation_model",
    "embedding_model",
    "thresholds",
    "retrieved_fragments",
    "confidence",
    "abstained",
    "use_guardrail",
    "alarms",
    "escalated",
    "prev_hash",
    "hash",
)
DECISION_FIELDS = (
    "schema_version",
    "event_type",
    "timestamp_utc",
    "ref_hash",
    "case_id",
    "action",
    "suggested_level",
    "final_level",
    "reason",
    "user",
    "review_seconds",
    "prev_hash",
    "hash",
)
DECISION_NON_NULL = (
    "schema_version",
    "event_type",
    "timestamp_utc",
    "ref_hash",
    "action",
    "final_level",
    "review_seconds",
    "prev_hash",
    "hash",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj) -> str:
    """Serialización canónica (llaves ordenadas, sin espacios, UTF-8) usada
    para calcular el hash de cada evento."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_hash(prev_hash: str, record_without_hash: dict) -> str:
    return hashlib.sha256(
        (prev_hash + canonical_json(record_without_hash)).encode("utf-8")
    ).hexdigest()


def build_audit_record(
    case_id: str,
    patient_text: str,
    corpus_format: str,
    embedding_mode: str,
    suggestion: Suggestion,
    corpus_hash: str,
    generation_model: str,
    embedding_model: str,
    prompt_version: str | None,
    thresholds: dict,
    use_guardrail: bool = False,
    alarm_rules_version: str | None = None,
) -> dict:
    """Construye un evento de sugerencia (sin ``prev_hash`` ni ``hash``, que
    los asigna ``AuditLog.append``)."""
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_type": EVENT_SUGGESTION,
        "timestamp_utc": _now(),
        "case_id": case_id,
        "input_text": patient_text,
        "corpus_format": corpus_format,
        "corpus_hash": corpus_hash,
        "embedding_mode": embedding_mode,
        # "gemini" | "deterministic_fallback" | "deterministic"
        "backend_used": suggestion.backend_used,
        "degraded": suggestion.degraded,
        "generation_model": generation_model,
        "embedding_model": embedding_model,
        "prompt_version": prompt_version,
        "thresholds": thresholds,
        "retrieved_fragments": [
            {
                "chunk_id": rc.chunk.chunk_id,
                "score": round(rc.score, 6),
                "rank": rank,
            }
            for rank, rc in enumerate(suggestion.retrieved, start=1)
        ],
        "suggested_level": suggestion.level,
        "citation": suggestion.citation,
        "confidence": round(suggestion.confidence, 6),
        "abstained": suggestion.abstained,
        "abstain_reason": suggestion.reason if suggestion.abstained else None,
        # Componente 1 + guardarraíl (src/extraction.py, src/guardrail.py).
        "use_guardrail": use_guardrail,
        "alarm_rules_version": alarm_rules_version,
        "alarms": list(suggestion.alarms),
        "n_modelo": suggestion.n_modelo,
        "n_reglas": suggestion.n_reglas,
        "n_final": suggestion.n_final,
        "model_citation": suggestion.model_citation,
        "escalated": suggestion.escalated,
        # La decisión final del personal de salud se registra como un evento
        # aparte (AuditLog.append_decision), nunca sobrescribiendo este campo,
        # como lo exige el principio de "asistencia, no decisión".
        "human_decision": None,
    }


def _missing_fields(record: dict) -> list[str]:
    """Campos obligatorios ausentes (o null cuando no pueden serlo)."""
    if record.get("event_type") == EVENT_DECISION:
        fields, non_null = DECISION_FIELDS, DECISION_NON_NULL
    else:
        fields, non_null = SUGGESTION_FIELDS, SUGGESTION_NON_NULL

    missing = [f for f in fields if f not in record]
    missing += [f for f in non_null if f in record and record[f] is None]

    if record.get("event_type") == EVENT_SUGGESTION:
        # Reglas condicionales: una sugerencia con nivel exige cita; una
        # abstención exige su motivo; la ruta Gemini exige versión de prompt.
        if record.get("abstained") is False:
            for f in ("suggested_level", "citation", "n_modelo", "n_final"):
                if f in record and record[f] is None:
                    missing.append(f)
        if record.get("abstained") is True and record.get("abstain_reason") is None:
            if "abstain_reason" in record:
                missing.append("abstain_reason")
        if record.get("backend_used") == "gemini" and record.get("prompt_version") is None:
            if "prompt_version" in record:
                missing.append("prompt_version")
    return missing


def completeness(records: list[dict]) -> dict:
    """Métrica "completitud del registro" del paper: porcentaje de eventos
    con todos sus campos obligatorios presentes (según su tipo)."""
    if not records:
        return {"n_registros": 0, "n_completos": 0, "completitud": float("nan"), "faltantes": {}}
    missing_counter: Counter = Counter()
    n_complete = 0
    for r in records:
        missing = _missing_fields(r)
        if missing:
            missing_counter.update(missing)
        else:
            n_complete += 1
    return {
        "n_registros": len(records),
        "n_completos": n_complete,
        "completitud": n_complete / len(records),
        "faltantes": dict(missing_counter),
    }


def verify_chain(records: list[dict]) -> tuple[bool, int | None]:
    """Recalcula la cadena de hashes. Devuelve ``(True, None)`` si está
    íntegra, o ``(False, i)`` con el índice del primer evento inválido."""
    prev = GENESIS_HASH
    for i, record in enumerate(records):
        if record.get("prev_hash") != prev:
            return False, i
        body = {k: v for k, v in record.items() if k != "hash"}
        if compute_hash(prev, body) != record.get("hash"):
            return False, i
        prev = record["hash"]
    return True, None


class AuditLog:
    """Registro JSONL de solo anexado con hash encadenado.

    El candado es por ARCHIVO (compartido entre instancias), porque varias
    instancias pueden escribir el mismo registro desde hilos distintos (p. ej.
    la API web: una por consulta y otra por decisión), y dos anexos
    simultáneos leerían el mismo ``prev_hash`` y romperían la cadena."""

    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.path.resolve())
        with AuditLog._locks_guard:
            self._lock = AuditLog._locks.setdefault(key, threading.Lock())

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _last_hash(self) -> str:
        records = self.read_all()
        if not records:
            return GENESIS_HASH
        return records[-1].get("hash") or GENESIS_HASH

    def append(self, record: dict) -> dict:
        """Encadena y escribe ``record``. Devuelve el evento tal como quedó
        escrito (con ``prev_hash`` y ``hash``)."""
        with self._lock:
            event = {k: v for k, v in record.items() if k not in ("prev_hash", "hash")}
            event["prev_hash"] = self._last_hash()
            event["hash"] = compute_hash(event["prev_hash"], event)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            return event

    def find(self, record_hash: str) -> dict | None:
        return next((r for r in self.read_all() if r.get("hash") == record_hash), None)

    def append_decision(
        self,
        record_hash: str,
        action: str,
        final_level: int | None = None,
        reason: str | None = None,
        user: str | None = None,
        review_seconds: float = 0.0,
    ) -> dict:
        """Registra la decisión del profesional sobre una sugerencia como un
        evento NUEVO que referencia su hash (no reescribe el original).

        - ``"confirma"``: acepta el nivel sugerido; solo es posible si la
          sugerencia tiene nivel (una abstención no se puede "confirmar").
        - ``"corrige"``: exige ``final_level`` (1-5) y ``reason``.

        Lanza ``KeyError`` si ``record_hash`` no existe y ``ValueError`` si la
        decisión no es válida.
        """
        original = self.find(record_hash)
        if original is None or original.get("event_type") != EVENT_SUGGESTION:
            raise KeyError(f"No existe una sugerencia con hash {record_hash}")
        if action not in DECISION_ACTIONS:
            raise ValueError(f"action debe ser uno de {DECISION_ACTIONS}")

        # suggested_level es el nivel FINAL presentado (n_final tras el guardarraíl).
        suggested = original.get("suggested_level")
        if action == "confirma":
            if suggested is None:
                raise ValueError(
                    "La sugerencia no tiene nivel (abstención): use 'corrige' con un nivel y un motivo."
                )
            if final_level is not None and final_level != suggested:
                raise ValueError("Para cambiar el nivel use 'corrige', no 'confirma'.")
            final_level = suggested
        else:
            if final_level not in (1, 2, 3, 4, 5):
                raise ValueError("'corrige' exige final_level entre 1 y 5.")
            if not reason or not reason.strip():
                raise ValueError("'corrige' exige un motivo.")

        event = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "event_type": EVENT_DECISION,
            "timestamp_utc": _now(),
            "ref_hash": record_hash,
            "case_id": original.get("case_id"),
            "action": action,
            "suggested_level": suggested,
            "final_level": final_level,
            "reason": reason.strip() if reason else None,
            "user": user,
            "review_seconds": float(review_seconds),
        }
        return self.append(event)

    def verify_chain(self) -> tuple[bool, int | None]:
        return verify_chain(self.read_all())
