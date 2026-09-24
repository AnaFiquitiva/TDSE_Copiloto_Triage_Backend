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
