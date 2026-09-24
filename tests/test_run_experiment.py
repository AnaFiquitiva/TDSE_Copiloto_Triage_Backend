"""Pruebas del arnés de E5/E6 sin red: sin clave ni caché deben quedar "no
evaluadas" (nunca calculadas con el respaldo); con un modelo inyectado se
verifican las repeticiones y la comparación pareada E6 frente a E5."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import run_experiment
from src.llm_generator import LLMBackedGenerator
from src.pipeline import CopilotoPipeline
from src.retrieval import BM25Retriever
from src.pipeline import _cached_corpus

CASES = run_experiment.load_cases()[:12]


class TestGeminiCellsNotEvaluated(unittest.TestCase):
    def test_without_key_or_cache_cells_are_marked_not_evaluated(self):
        with patch.object(run_experiment, "gemini_resources", return_value={"api_key": False, "cache_llm": False}):
            cells, comparisons = run_experiment.run_gemini_cells(CASES, run_experiment.baseline_levels(CASES))
        self.assertEqual([c["cell"] for c in cells], ["E5", "E6"])
        for cell in cells:
            self.assertIn("sin API ni caché", cell["no_evaluada"])
        self.assertIsNone(comparisons)

    def test_readonly_without_cache_is_not_evaluated(self):
        with patch.object(run_experiment, "gemini_resources", return_value={"api_key": True, "cache_llm": False}):
            cells, _ = run_experiment.run_gemini_cells(
                CASES, run_experiment.baseline_levels(CASES), first_mode="readonly"
            )
        self.assertTrue(all(c.get("no_evaluada") for c in cells))

    def test_report_marks_not_evaluated_cells(self):
        cells = [run_experiment._not_evaluated("E5", "no evaluada (sin API ni caché)")]
        md = run_experiment.format_markdown(
            {"n_pares": 1, "kappa_ponderado_cuadratico": 1.0},
            run_experiment.run_baseline(CASES),
            cells,
            n_total_cases=len(CASES),
        )
        self.assertIn("no evaluada (sin API ni caché)", md)
        self.assertIn("H5", md)


def _fake_pipeline_factory(*args, **kwargs):
    """CopilotoPipeline con la ruta principal simulada: BM25 como recuperador
    y un 'LLM' que siempre elige el primer candidato."""
    chunks = list(_cached_corpus("reformatted"))
    llm = LLMBackedGenerator(
        BM25Retriever(chunks),
        generate_fn=lambda prompt: {
            "citation": prompt.split('id="')[1].split('"')[0],
            "abstain": False,
            "reasoning": "primer candidato",
        },
    )
    kwargs["llm_generator"] = llm
    return CopilotoPipeline(*args, **kwargs)


class TestGeminiCellsWithInjectedModel(unittest.TestCase):
    def test_three_repetitions_and_paired_h5_comparison(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_experiment, "RESULTS_DIR", Path(tmp)
        ), patch.object(run_experiment, "CopilotoPipeline", _fake_pipeline_factory), patch.object(
            run_experiment, "gemini_resources", return_value={"api_key": False, "cache_llm": True}
        ):
            cells, comparisons = run_experiment.run_gemini_cells(CASES, run_experiment.baseline_levels(CASES))
            self.assertTrue((Path(tmp) / "audit_E5.jsonl").exists())

        e5, e6 = cells
        self.assertFalse(e5.get("no_evaluada"))
        self.assertEqual(e5["tasa_modo_degradado"], 0.0)
        self.assertEqual(e5["repeticiones"]["n"], 3)
        self.assertEqual(e5["repeticiones"]["modos_cache"], ["readonly"] * 3)
        s = e5["repeticiones"]["S_sub_triage"]
        self.assertEqual(s["min"], s["max"])  # modelo determinista: sin variación
        self.assertEqual(e5["completitud_registro"], 1.0)
        self.assertTrue(e5["cadena_auditoria_valida"])

        h5 = comparisons["H5_E6_vs_E5"]
        self.assertGreater(h5["n_pareados"], 0)
        # El guardarraíl solo puede subir la urgencia: E6 nunca pierde casos I-II frente a E5.
        self.assertEqual(h5["mcnemar_sens_I_II"]["solo_b_acierta"], 0)
        self.assertLessEqual(h5["S_a"], h5["S_b"])


if __name__ == "__main__":
    unittest.main()
