"""Pruebas de GeminiRetriever con un `embed_fn` falso (sin red ni API key):
verifica el cálculo de similitud coseno y el cacheo en disco de los
embeddings del corpus."""
import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from src.ingest import Chunk
from src.llm_client import LLMUnavailableError
from src.retrieval import GeminiRetriever


def _fake_embed(text: str, task_type: str | None = None) -> list[float]:
    """Embedding determinista y falso: vector one-hot según la primera
    palabra del texto, suficiente para probar el ranking por similitud.
    Ignora `task_type` (a diferencia del real, ver test_llm_client.py para
    la prueba de que sí se envía a la API)."""
    vocab = ["dolor", "fiebre", "certificado", "otro"]
    first_word = text.strip().split()[0].lower() if text.strip() else "otro"
    return [1.0 if first_word == v else 0.0 for v in vocab] or [0.0] * len(vocab)


class TestGeminiRetriever(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"GEMINI_CACHE_MODE": "readwrite"})
        env.start()
        self.addCleanup(env.stop)
        self.chunks = [
            Chunk("A", "dolor de pecho", "src", "reformatted", level=2),
            Chunk("B", "fiebre alta", "src", "reformatted", level=1),
            Chunk("C", "certificado medico", "src", "reformatted", level=5),
        ]

    def test_retrieve_ranks_by_cosine_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            retriever = GeminiRetriever(self.chunks, cache_path=cache_path, embed_fn=_fake_embed)
            results = retriever.retrieve("dolor en la espalda", k=1)
            self.assertEqual(results[0].chunk.chunk_id, "A")

    def test_corpus_embeddings_are_cached_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            call_count = {"n": 0}

            def counting_embed(text, task_type=None):
                call_count["n"] += 1
                return _fake_embed(text, task_type=task_type)

            GeminiRetriever(self.chunks, cache_path=cache_path, embed_fn=counting_embed)
            self.assertEqual(call_count["n"], 3)  # una llamada por chunk del corpus
            self.assertTrue(cache_path.exists())

            # Una segunda instancia con el mismo cache no debe volver a llamar
            # a embed_fn para los mismos textos.
            GeminiRetriever(self.chunks, cache_path=cache_path, embed_fn=counting_embed)
            self.assertEqual(call_count["n"], 3)

    def test_cache_persists_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            GeminiRetriever(self.chunks, cache_path=cache_path, embed_fn=_fake_embed)
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 3)

    def test_query_embeddings_are_cached_and_replayable_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            retriever = GeminiRetriever(self.chunks, cache_path=cache_path, embed_fn=_fake_embed)
            first = retriever.retrieve("fiebre de tres dias", k=1)

            def no_network(text, task_type=None):
                raise AssertionError("readonly no debe llamar a embed_fn")

            with patch.dict("os.environ", {"GEMINI_CACHE_MODE": "readonly"}):
                replay = GeminiRetriever(self.chunks, cache_path=cache_path, embed_fn=no_network)
                again = replay.retrieve("fiebre de tres dias", k=1)
                self.assertEqual(first[0].chunk.chunk_id, again[0].chunk.chunk_id)
                with self.assertRaises(LLMUnavailableError):
                    replay.retrieve("una consulta que nunca se cacheo", k=1)

    def test_legacy_cache_keys_without_model_are_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            GeminiRetriever(self.chunks, cache_path=cache_path, embed_fn=_fake_embed)
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            legacy = {k.split(":", 1)[1]: v for k, v in data.items()}  # formato v1 "TASK:hash"
            cache_path.write_text(json.dumps(legacy), encoding="utf-8")

            def no_network(text, task_type=None):
                raise AssertionError("las claves v1 deben reutilizarse")

            GeminiRetriever(self.chunks, cache_path=cache_path, embed_fn=no_network)


if __name__ == "__main__":
    unittest.main()
