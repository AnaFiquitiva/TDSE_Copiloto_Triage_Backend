"""Escritura OPCIONAL de la decisión del profesional en un servidor FHIR R4
(p. ej. HAPI FHIR de ``docker-compose.yml``). Solo biblioteca estándar.

Desactivado por defecto (``FHIR_ENABLED=false``). Cuando está activo, cada
decisión registrada se envía como un Bundle de tipo ``transaction`` con:

- ``Encounter``: ``priority`` = nivel final de triage;
- ``Provenance``: apunta al Encounter y referencia la cita normativa y el hash
  del evento de auditoría;
- ``AuditEvent``: la creación, con los hashes de sugerencia y decisión.

La escritura NUNCA interrumpe el flujo: si el servidor no responde o devuelve
un error, se informa en el resultado (``ok: False``) y la decisión sigue
registrada en la auditoría local, que es la fuente de verdad del prototipo.
Los sistemas de códigos ``urn:tdse:*`` son locales de este prototipo; no son
terminologías oficiales.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

DEFAULT_FHIR_BASE_URL = "http://localhost:8080/fhir"
TIMEOUT_SECONDS = 5
TRIAGE_SYSTEM = "urn:tdse:triage-res5596"
CITATION_SYSTEM = "urn:tdse:corpus-cita"
AUDIT_SYSTEM = "urn:tdse:audit-record"
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
LEVEL_NAMES = {1: "Resucitación", 2: "Emergencia", 3: "Urgencia", 4: "Urgencia menor", 5: "No urgencia"}


def fhir_enabled() -> bool:
    return os.environ.get("FHIR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "si", "sí"}


def fhir_base_url() -> str:
    return os.environ.get("FHIR_BASE_URL", DEFAULT_FHIR_BASE_URL).rstrip("/")


def build_decision_bundle(suggestion: dict, decision: dict) -> dict:
    """Bundle transaccional FHIR R4 para una decisión. ``suggestion`` es el
    evento de sugerencia de la auditoría y ``decision`` el evento de decisión."""
    level = decision["final_level"]
    recorded = decision.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()
    user = decision.get("user") or "profesional (sin identificar)"
    enc_ref = f"urn:uuid:{uuid.uuid4()}"

    encounter = {
        "resourceType": "Encounter",
        "status": "in-progress",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "EMER",
            "display": "emergency",
        },
        "priority": {
            "coding": [
                {"system": TRIAGE_SYSTEM, "code": ROMAN[level], "display": f"Nivel {ROMAN[level]} - {LEVEL_NAMES[level]}"}
            ],
            "text": f"Triage nivel {ROMAN[level]} (Res. 5596/2015), decisión: {decision['action']}",
        },
        "identifier": [{"system": AUDIT_SYSTEM, "value": suggestion["hash"]}],
    }

    provenance: dict = {
        "resourceType": "Provenance",
        "target": [{"reference": enc_ref}],
        "recorded": recorded,
        "agent": [
            {
                "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type", "code": "author"}]},
                "who": {"display": user},
            },
            {
                "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type", "code": "assembler"}]},
                "who": {"display": f"TDSE copiloto ({suggestion.get('generation_model')}, {suggestion.get('backend_used')})"},
            },
        ],
        "entity": [
            {
                "role": "source",
                "what": {"identifier": {"system": AUDIT_SYSTEM, "value": suggestion["hash"]},
                         "display": "Evento de sugerencia en la auditoría del copiloto"},
            }
        ],
    }
    if suggestion.get("citation"):
        provenance["entity"].insert(0, {
            "role": "source",
            "what": {"identifier": {"system": CITATION_SYSTEM, "value": suggestion["citation"]},
                     "display": f"Fragmento normativo {suggestion['citation']}"},
        })
    if decision.get("reason"):
        provenance["reason"] = [{"text": decision["reason"]}]

    audit_event = {
        "resourceType": "AuditEvent",
        "type": {"system": "http://terminology.hl7.org/CodeSystem/audit-event-type", "code": "rest", "display": "RESTful Operation"},
        "subtype": [{"system": "http://hl7.org/fhir/restful-interaction", "code": "create", "display": "create"}],
        "action": "C",
        "recorded": recorded,
        "outcome": "0",
        "agent": [{"who": {"display": user}, "requestor": True}],
        "source": {"observer": {"display": "TDSE Copiloto de Triage RAG"}},
        "entity": [
            {
                "what": {"reference": enc_ref},
                "detail": [
                    {"type": "record_hash", "valueString": suggestion["hash"]},
                    {"type": "decision_hash", "valueString": decision["hash"]},
                    {"type": "action", "valueString": decision["action"]},
                    {"type": "review_seconds", "valueString": str(decision.get("review_seconds"))},
                ],
            }
        ],
    }

    def entry(resource: dict, full_url: str | None = None) -> dict:
        return {
            "fullUrl": full_url or f"urn:uuid:{uuid.uuid4()}",
            "resource": resource,
            "request": {"method": "POST", "url": resource["resourceType"]},
        }

    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [entry(encounter, enc_ref), entry(provenance), entry(audit_event)],
    }


def _post_bundle(base_url: str, bundle: dict) -> dict:
    request = urllib.request.Request(
        base_url,
        data=json.dumps(bundle).encode("utf-8"),
        headers={"Content-Type": "application/fhir+json", "Accept": "application/fhir+json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def write_decision(suggestion: dict, decision: dict) -> dict:
    """Envía la decisión al servidor FHIR si está activo. Nunca lanza."""
    if not fhir_enabled():
        return {"enabled": False}
    base = fhir_base_url()
    try:
        response = _post_bundle(base, build_decision_bundle(suggestion, decision))
        locations = [e.get("response", {}).get("location") for e in response.get("entry", [])]
        return {"enabled": True, "ok": True, "server": base, "locations": locations}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {"enabled": True, "ok": False, "server": base, "error": f"{type(exc).__name__}: {exc}"[:300]}
