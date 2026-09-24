"""Pruebas de CDS Hooks y de la escritura opcional en FHIR, sin red: la API
de Gemini no se llama (sin clave, caché apagada) y el servidor FHIR se simula."""
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import cds  # noqa: E402
import fhir_client  # noqa: E402

client = TestClient(app_module.app)


class _Offline(unittest.TestCase):
    extra_env: dict = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._patches = [
            patch.object(app_module, "DEFAULT_AUDIT_PATH", Path(self.tmp.name) / "audit.jsonl"),
            patch.dict("os.environ", {"GEMINI_CACHE_MODE": "off", **self.extra_env}, clear=True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.tmp.cleanup()

    def _consulta(self, texto):
        resp = client.post("/api/consulta", json={"texto": texto, "backend": "deterministic"})
        self.assertEqual(resp.status_code, 200)
        return resp.json()


class TestCDSHooks(_Offline):
    def test_discovery_lists_triage_service(self):
        body = client.get("/cds-services").json()
        service = body["services"][0]
        self.assertEqual(service["id"], "triage-copiloto")
        self.assertEqual(service["hook"], "encounter-start")

    def test_service_returns_critical_card_for_level1_alarm(self):
        resp = client.post(
            "/cds-services/triage-copiloto",
            json={"hook": "encounter-start", "hookInstance": "h1",
                  "context": {"relato": "La senora llego sin pulso, no respira y no reacciona."}},
        )
        self.assertEqual(resp.status_code, 200)
        card = resp.json()["cards"][0]
        self.assertEqual(card["indicator"], "critical")
        self.assertLessEqual(len(card["summary"]), 140)
        self.assertIn("responsabilidad del profesional de salud", card["detail"])
        self.assertEqual(card["source"]["label"], cds.SOURCE_LABEL)
        # Sin clave, la ruta principal cae al respaldo y la tarjeta lo dice.
        self.assertTrue(card["extension"]["degraded"])
        self.assertIn("Modo degradado", card["detail"])

    def test_missing_relato_is_400(self):
        resp = client.post("/cds-services/triage-copiloto", json={"hook": "encounter-start", "context": {}})
        self.assertEqual(resp.status_code, 400)


class TestCardBuilder(unittest.TestCase):
    def _response(self, **overrides):
        base = {
            "suggestion": {"citation": "RES5596-N4", "reason": "ok"},
            "n_final": 4, "n_modelo": 4, "n_reglas": None, "model_citation": "RES5596-N4",
            "escalated": False, "degraded": False, "alarms": [], "backend_used": "gemini",
            "record_hash": "a" * 64, "cited_fragment_text": "## [CITA: RES5596-N4] Nivel IV",
        }
        base.update(overrides)
        return base

    def test_indicator_rules(self):
        self.assertEqual(cds.indicator_for(1, False), "critical")
        self.assertEqual(cds.indicator_for(2, False), "critical")
        self.assertEqual(cds.indicator_for(3, False), "info")
        self.assertEqual(cds.indicator_for(None, True), "critical")

    def test_level4_card_is_info_and_cites(self):
        card = cds.build_card(self._response())
        self.assertEqual(card["indicator"], "info")
        self.assertIn("RES5596-N4", card["summary"])
        self.assertIn("`RES5596-N4`", card["detail"])

    def test_abstention_card_escalates_and_shows_rule_as_reference(self):
        card = cds.build_card(self._response(
            suggestion={"citation": None, "reason": "abstencion_del_modelo"}, n_final=None, n_modelo=None,
            n_reglas=2, escalated=True,
            alarms=[{"rule_id": "ALR-022", "level": 2, "citation": "ALARMA-SUICIDA-GRAVE", "matched_pattern": "x"}],
        ))
        self.assertEqual(card["indicator"], "critical")
        self.assertIn("escalar", card["summary"])
        self.assertIn("no del modelo", card["detail"])

    def test_raised_by_rule_is_explained(self):
        card = cds.build_card(self._response(
            suggestion={"citation": "ALARMA-SUICIDA-GRAVE", "reason": "elevado_por_regla_de_alarma"},
            n_final=2, n_modelo=4, n_reglas=2,
        ))
        self.assertIn("Elevado por regla de alarma", card["detail"])


class TestFHIRDisabledByDefault(_Offline):
    def test_decision_does_not_touch_fhir(self):
        record_hash = self._consulta("vengo por un certificado medico")["record_hash"]
        with patch("urllib.request.urlopen", side_effect=AssertionError("no debe llamar a FHIR")):
            resp = client.post(
                "/api/decision",
                json={"record_hash": record_hash, "action": "corrige", "final_level": 4, "reason": "x",
                      "review_seconds": 2},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fhir"], {"enabled": False})


class _FakeFHIRResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestFHIREnabled(_Offline):
    extra_env = {"FHIR_ENABLED": "true", "FHIR_BASE_URL": "http://fhir.local/fhir"}

    def test_decision_writes_encounter_provenance_and_auditevent(self):
        consulta = self._consulta("El paciente no respira y no reacciona.")
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["bundle"] = json.loads(request.data.decode("utf-8"))
            return _FakeFHIRResponse({"entry": [{"response": {"location": "Encounter/7/_history/1"}}]})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            resp = client.post(
                "/api/decision",
                json={"record_hash": consulta["record_hash"], "action": "corrige", "final_level": 1,
                      "reason": "paro cardiorrespiratorio", "review_seconds": 5, "user": "enf-1"},
            )
        self.assertEqual(resp.status_code, 200)
        fhir = resp.json()["fhir"]
        self.assertTrue(fhir["ok"])
        self.assertEqual(fhir["locations"], ["Encounter/7/_history/1"])
        self.assertEqual(captured["url"], "http://fhir.local/fhir")

        bundle = captured["bundle"]
        self.assertEqual(bundle["type"], "transaction")
        by_type = {e["resource"]["resourceType"]: e for e in bundle["entry"]}
        self.assertEqual(set(by_type), {"Encounter", "Provenance", "AuditEvent"})
        enc_url = by_type["Encounter"]["fullUrl"]
        self.assertEqual(by_type["Encounter"]["resource"]["priority"]["coding"][0]["code"], "I")
        prov = by_type["Provenance"]["resource"]
        self.assertEqual(prov["target"][0]["reference"], enc_url)
        values = [e["what"]["identifier"]["value"] for e in prov["entity"]]
        self.assertIn(consulta["record_hash"], values)
        if consulta["suggestion"]["citation"]:
            self.assertIn(consulta["suggestion"]["citation"], values)
        self.assertEqual(prov["reason"][0]["text"], "paro cardiorrespiratorio")
        audit = by_type["AuditEvent"]["resource"]
        self.assertEqual(audit["entity"][0]["what"]["reference"], enc_url)

    def test_fhir_down_does_not_break_the_decision(self):
        record_hash = self._consulta("vengo por un certificado medico")["record_hash"]
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
            resp = client.post(
                "/api/decision",
                json={"record_hash": record_hash, "action": "corrige", "final_level": 5, "reason": "x",
                      "review_seconds": 1},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fhir"]["ok"], False)
        self.assertIn("URLError", resp.json()["fhir"]["error"])
        # La auditoría local quedó registrada igual.
        audit = client.get("/api/auditoria").json()
        self.assertTrue(audit["cadena_valida"])
        self.assertEqual(audit["events"][-1]["event_type"], "decision")


class TestFHIRFlags(unittest.TestCase):
    def test_disabled_values(self):
        for value in ("false", "0", "", "no"):
            with patch.dict("os.environ", {"FHIR_ENABLED": value}):
                self.assertFalse(fhir_client.fhir_enabled())
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(fhir_client.fhir_enabled())


if __name__ == "__main__":
    unittest.main()
