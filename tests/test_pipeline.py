"""Pruebas del pipeline. Ninguna llama a la red: la ruta Gemini se prueba sin
`GEMINI_API_KEY` (para forzar el respaldo) o con un generador inyectado."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.audit import AuditLog, completeness
from src.ingest import Chunk
from src.llm_client import LLMUnavailableError
from src.llm_generator import LLMBackedGenerator
from src.pipeline import CopilotoPipeline
from src.retrieval import RankedChunk

# Entorno sin clave y con la caché del LLM apagada: garantiza que ninguna
# prueba llame a la API aunque la máquina tenga GEMINI_API_KEY configurada.
_OFFLINE_ENV = {"GEMINI_CACHE_MODE": "off"}


def _offline():
    return patch.dict("os.environ", _OFFLINE_ENV, clear=True)


class TestCopilotoPipeline(unittest.TestCase):
    def test_default_backend_is_gemini(self):
        with tempfile.TemporaryDirectory() as tmp, _offline():
            pipeline = CopilotoPipeline(corpus_format="reformatted", audit_path=Path(tmp) / "a.jsonl")
            self.assertEqual(pipeline.backend, "gemini")

    def test_run_case_writes_audit_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            pipeline = CopilotoPipeline(
                corpus_format="reformatted",
                embedding_mode="clinical_es",
                audit_path=audit_path,
                backend="deterministic",
            )
            pipeline.run_case("T001", "El paciente no respira y no reacciona.")

            records = AuditLog(audit_path).read_all()
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["case_id"], "T001")
            self.assertEqual(record["backend_used"], "deterministic")
            self.assertFalse(record["degraded"])
            self.assertIn("retrieved_fragments", record)
            self.assertIn("suggested_level", record)
            self.assertIn("abstained", record)
            self.assertEqual(len(record["corpus_hash"]), 64)
            self.assertEqual(record["thresholds"]["k"], 5)
            self.assertEqual(record["retrieved_fragments"][0]["rank"], 1)

    def test_new_run_has_valid_chain_and_full_completeness(self):
        with tempfile.TemporaryDirectory() as tmp, _offline():
            audit_path = Path(tmp) / "audit.jsonl"
            pipeline = CopilotoPipeline(corpus_format="reformatted", audit_path=audit_path)
            for i, text in enumerate(
                ["El paciente no respira.", "vengo por un certificado medico", "algo me pasa"]
            ):
                pipeline.run_case(f"T{i}", text)
            log = AuditLog(audit_path)
            self.assertEqual(log.verify_chain(), (True, None))
            self.assertEqual(completeness(log.read_all())["completitud"], 1.0)

    def test_abstention_never_returns_a_level_without_citation(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            pipeline = CopilotoPipeline(
                corpus_format="raw", embedding_mode="generic", audit_path=audit_path, backend="deterministic"
            )
            suggestion = pipeline.run_case("T002", "algo me duele no se muy bien que es")
            if suggestion.abstained:
                self.assertIsNone(suggestion.level)
                self.assertIsNone(suggestion.citation)
            else:
                self.assertIsNotNone(suggestion.citation)


class TestGeminiBackendFallback(unittest.TestCase):
    def test_missing_api_key_falls_back_and_is_marked_degraded(self):
        with tempfile.TemporaryDirectory() as tmp, _offline():
            audit_path = Path(tmp) / "audit.jsonl"
            pipeline = CopilotoPipeline(corpus_format="reformatted", backend="gemini", audit_path=audit_path)
            suggestion = pipeline.run_case("T003", "El paciente no respira y no reacciona.")

            # Sin API key, el pipeline nunca debe fallar: usa el respaldo, marcado.
            self.assertEqual(suggestion.backend_used, "deterministic_fallback")
            self.assertTrue(suggestion.degraded)
            record = AuditLog(audit_path).read_all()[0]
            self.assertEqual(record["backend_used"], "deterministic_fallback")
            self.assertTrue(record["degraded"])

    def test_injected_generate_fn_raising_falls_back_per_case(self):
        """La función de generación inyectada lanza LLMUnavailableError: el
        caso cae al respaldo con backend_used=deterministic_fallback y
        degraded=True."""

        def failing_generate(prompt):
            raise LLMUnavailableError("fallo simulado de red")

        chunk = Chunk("RES5596-N5", "certificado medico", "niveles", "reformatted", level=5)
        llm = LLMBackedGenerator(_FixedRetriever([RankedChunk(chunk, 0.9)]), generate_fn=failing_generate)
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            pipeline = CopilotoPipeline(
                corpus_format="reformatted", backend="gemini", audit_path=audit_path, llm_generator=llm
            )
            suggestion = pipeline.run_case("T004", "vengo a pedir un certificado medico")
            self.assertEqual(suggestion.backend_used, "deterministic_fallback")
            self.assertTrue(suggestion.degraded)
            self.assertEqual(AuditLog(audit_path).read_all()[0]["backend_used"], "deterministic_fallback")

    def test_injected_generate_fn_success_is_not_degraded(self):
        chunk = Chunk("RES5596-N5", "certificado medico", "niveles", "reformatted", level=5)
        llm = LLMBackedGenerator(
            _FixedRetriever([RankedChunk(chunk, 0.9)]),
            generate_fn=lambda prompt: {"citation": "RES5596-N5", "abstain": False, "reasoning": "x"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            pipeline = CopilotoPipeline(
                corpus_format="reformatted", backend="gemini", audit_path=audit_path, llm_generator=llm
            )
            suggestion = pipeline.run_case("T005", "vengo a pedir un certificado medico")
            self.assertEqual(suggestion.backend_used, "gemini")
            self.assertFalse(suggestion.degraded)
            record = AuditLog(audit_path).read_all()[0]
            self.assertEqual(record["prompt_version"], "triage-selector-v1")
            self.assertEqual(completeness([record])["completitud"], 1.0)


class _FixedRetriever:
    def __init__(self, ranked):
        self._ranked = ranked

    def retrieve(self, query, k=5):
        return self._ranked[:k]


if __name__ == "__main__":
    unittest.main()
