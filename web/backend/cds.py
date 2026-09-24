"""Tarjetas CDS Hooks (https://cds-hooks.org) a partir de una sugerencia del
copiloto. Módulo puro, sin FastAPI, para poder probarlo directamente.

Servicio: ``triage-copiloto`` sobre el hook ``encounter-start``. El relato del
paciente llega en ``context.relato``, que es una extensión de contexto propia
de este prototipo. La tarjeta NUNCA propone una acción automática: es
información para el profesional, coherente con "asiste, no decide".
"""
from __future__ import annotations

import uuid

SERVICE_ID = "triage-copiloto"
HOOK = "encounter-start"
SOURCE_LABEL = "TDSE Copiloto de Triage RAG (prototipo académico)"
DISCLAIMER = "Sugerencia de apoyo. La clasificación final es responsabilidad del profesional de salud."
LEVEL_NAMES = {1: "I · Resucitación", 2: "II · Emergencia", 3: "III · Urgencia", 4: "IV · Urgencia menor", 5: "V · No urgencia"}

DISCOVERY = {
    "services": [
        {
            "hook": HOOK,
            "id": SERVICE_ID,
            "title": "Copiloto de triage (Res. 5596/2015)",
            "description": (
                "Sugiere un nivel de triage I-V citando el fragmento normativo que lo respalda, "
                "o se abstiene y escala. Requiere el relato libre del paciente en context.relato."
            ),
            "prefetch": {},
        }
    ]
}


def indicator_for(n_final: int | None, escalated: bool) -> str:
    """``critical`` si el nivel es I-II o hay escalamiento; si no, ``info``."""
    if escalated or (n_final is not None and n_final <= 2):
        return "critical"
    return "info"


def build_card(response: dict) -> dict:
    """Construye la tarjeta a partir de la respuesta de ``/api/consulta``
    (mismos campos: suggestion, n_final, escalated, alarms, record_hash...)."""
    suggestion = response["suggestion"]
    n_final = response["n_final"]
    escalated = response["escalated"]

    if n_final is not None:
        summary = f"Triage sugerido: nivel {LEVEL_NAMES[n_final]} — cita {suggestion['citation']}"
    else:
        summary = "Sin sugerencia del copiloto: escalar a revisión prioritaria"

    lines: list[str] = []
    if n_final is not None:
        lines.append(f"**Cita:** `{suggestion['citation']}`")
        if response.get("n_reglas") is not None and response.get("n_modelo") is not None \
                and response["n_reglas"] < response["n_modelo"]:
            lines.append(
                f"**Elevado por regla de alarma** (el modelo sugería nivel {response['n_modelo']}, "
                f"cita `{response.get('model_citation')}`)."
            )
    else:
        lines.append(f"El sistema se abstuvo (`{suggestion['reason']}`).")
        reference = next((a for a in response["alarms"] if a["level"] == response.get("n_reglas")), None)
        if reference:
            lines.append(
                f"Referencia de una regla de alarma, no del modelo: nivel {reference['level']} "
                f"(`{reference['citation']}`)."
            )
    if response.get("cited_fragment_text"):
        lines.append("")
        lines.append("> " + response["cited_fragment_text"].strip().replace("\n", "\n> "))
    if response["alarms"]:
        lines.append("")
        lines.append("**Alarmas detectadas:** " + "; ".join(
            f"{a['rule_id']} “{a['matched_pattern']}” → nivel {a['level']}" for a in response["alarms"]
        ))
    if response.get("degraded"):
        lines.append("")
        lines.append("⚠️ **Modo degradado:** respaldo determinista (Gemini no disponible).")
    lines.append("")
    lines.append(f"_{DISCLAIMER}_")

    return {
        # UUID estable derivado del hash del evento de auditoría.
        "uuid": str(uuid.UUID(response["record_hash"][:32])),
        "summary": summary[:140],
        "detail": "\n".join(lines),
        "indicator": indicator_for(n_final, escalated),
        "source": {"label": SOURCE_LABEL},
        "extension": {
            "record_hash": response["record_hash"],
            "backend_used": response["backend_used"],
            "degraded": response["degraded"],
            "escalated": escalated,
            "n_final": n_final,
        },
    }
