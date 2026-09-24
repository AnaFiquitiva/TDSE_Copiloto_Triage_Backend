"""Pruebas de humo de la API web (requiere fastapi/httpx, ver
web/backend/requirements.txt). No forman parte de la suite principal de
`tests/` porque esta capa es opcional y tiene dependencias propias."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

client = TestClient(app)


class TestHealth(unittest.TestCase):
    def test_health_ok(self):
        resp = client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})


class TestConsulta(unittest.TestCase):
    def test_valid_case_returns_suggestion_and_baseline(self):
        resp = client.post(
            "/api/consulta",
            json={
                "texto": "El paciente no respira y no reacciona.",
                "corpus_format": "reformatted",
                "embedding_mode": "clinical_es",
                "backend": "deterministic",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("suggestion", body)
        self.assertIn("baseline", body)
        self.assertIn("retrieved", body)
        self.assertEqual(body["backend_used"], "deterministic")

    def test_empty_text_returns_400(self):
        resp = client.post("/api/consulta", json={"texto": "   "})
        self.assertEqual(resp.status_code, 400)


class _TempAuditMixin:
    """Cada prueba escribe en un registro temporal y sin red (sin API key)."""

    def setUp(self):
        import tempfile
        from unittest.mock import patch

        import app as app_module

        self.tmp = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.tmp.name) / "audit.jsonl"
        self._patches = [
            patch.object(app_module, "DEFAULT_AUDIT_PATH", self.audit_path),
            patch.dict("os.environ", {"GEMINI_CACHE_MODE": "off"}, clear=True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.tmp.cleanup()

    def _consulta(self, texto, backend="deterministic"):
        resp = client.post("/api/consulta", json={"texto": texto, "backend": backend})
        self.assertEqual(resp.status_code, 200)
        return resp.json()


class TestConsultaRevision(_TempAuditMixin, unittest.TestCase):
    def test_consulta_returns_review_fields(self):
        body = self._consulta("El paciente no respira y no reacciona.")
        for key in ("record_hash", "backend_used", "degraded", "escalated", "n_modelo", "n_reglas",
                    "n_final", "alarms", "cited_fragment_text"):
            self.assertIn(key, body)
        self.assertEqual(len(body["record_hash"]), 64)
        self.assertEqual(body["n_reglas"], 1)
        self.assertTrue(body["alarms"][0]["text"])
        if not body["suggestion"]["abstained"]:
            self.assertIn(body["suggestion"]["citation"], body["cited_fragment_text"])

    def test_default_backend_is_gemini_and_degrades_without_key(self):
        body = self._consulta("vengo por un certificado medico", backend="gemini")
        self.assertEqual(body["backend_used"], "deterministic_fallback")
        self.assertTrue(body["degraded"])


class TestDecision(_TempAuditMixin, unittest.TestCase):
    def test_corrige_without_level_or_reason_is_422(self):
        record_hash = self._consulta("vengo por un certificado medico")["record_hash"]
        for payload in (
            {"record_hash": record_hash, "action": "corrige", "reason": "x", "review_seconds": 3},
            {"record_hash": record_hash, "action": "corrige", "final_level": 2, "review_seconds": 3},
            {"record_hash": record_hash, "action": "corrige", "final_level": 2, "reason": "  ", "review_seconds": 3},
            {"record_hash": record_hash, "action": "corrige", "final_level": 7, "reason": "x", "review_seconds": 3},
            {"record_hash": record_hash, "action": "aprueba", "review_seconds": 3},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(client.post("/api/decision", json=payload).status_code, 422)

    def test_unknown_record_hash_is_404(self):
        resp = client.post(
            "/api/decision", json={"record_hash": "f" * 64, "action": "confirma", "review_seconds": 1}
        )
        self.assertEqual(resp.status_code, 404)

    def test_full_flow_consulta_decision_auditoria(self):
        first = self._consulta("vengo a pedir un certificado medico para el trabajo")
        second = self._consulta("El paciente no respira y no reacciona.")

        correction = client.post(
            "/api/decision",
            json={
                "record_hash": first["record_hash"],
                "action": "corrige",
                "final_level": 4,
                "reason": "refiere dolor asociado no documentado en el relato",
                "review_seconds": 21.5,
                "user": "enfermeria-1",
            },
        )
        self.assertEqual(correction.status_code, 200, correction.text)
        self.assertEqual(correction.json()["final_level"], 4)
        self.assertEqual(correction.json()["ref_hash"], first["record_hash"])

        if second["n_final"] is not None:
            confirm = client.post(
                "/api/decision",
                json={"record_hash": second["record_hash"], "action": "confirma", "review_seconds": 4.0},
            )
            self.assertEqual(confirm.status_code, 200, confirm.text)
            self.assertEqual(confirm.json()["final_level"], second["n_final"])

        audit = client.get("/api/auditoria", params={"limit": 10}).json()
        self.assertTrue(audit["cadena_valida"])
        self.assertEqual(audit["completitud"]["completitud"], 1.0)
        types = [e["event_type"] for e in audit["events"]]
        self.assertEqual(types[:3], ["sugerencia", "sugerencia", "decision"])
        # El evento original no se reescribe: sigue sin decisión humana.
        self.assertIsNone(audit["events"][0]["human_decision"])

    def test_confirming_an_abstention_is_422(self):
        from unittest.mock import patch

        import app as app_module
        from src.audit import AuditLog

        body = self._consulta("vengo por un certificado medico")
        record = AuditLog(app_module.DEFAULT_AUDIT_PATH).find(body["record_hash"])
        if record["suggested_level"] is not None:
            # Forzar una abstención para probar la regla sin depender del corpus.
            with patch.object(AuditLog, "find", return_value={**record, "suggested_level": None}):
                resp = client.post(
                    "/api/decision",
                    json={"record_hash": body["record_hash"], "action": "confirma", "review_seconds": 1},
                )
        else:
            resp = client.post(
                "/api/decision",
                json={"record_hash": body["record_hash"], "action": "confirma", "review_seconds": 1},
            )
        self.assertEqual(resp.status_code, 422)


class TestExperimento(unittest.TestCase):
    def test_returns_kappa_baseline_and_six_cells(self):
        resp = client.get("/api/experimento")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("kappa_ponderado_cuadratico", body["kappa"])
        self.assertEqual([c["cell"] for c in body["cells"]], ["E1", "E2", "E3", "E4", "E5", "E6"])
        self.assertGreaterEqual(body["n_total_cases"], 60)


class TestDataset(unittest.TestCase):
    def test_returns_distribution_and_chi2(self):
        resp = client.get("/api/dataset")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("distribution", body)
        self.assertIn("chi2_independence_test", body)
        self.assertGreater(body["total"], 0)


class TestCasosYCorpus(unittest.TestCase):
    def test_casos_returns_seventy_plus(self):
        resp = client.get("/api/casos")
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.json()["total"], 60)

    def test_corpus_reformatted_has_no_missing_levels_among_alarm_chunks(self):
        resp = client.get("/api/corpus", params={"corpus_format": "reformatted"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertGreater(body["total"], 0)


if __name__ == "__main__":
    unittest.main()
