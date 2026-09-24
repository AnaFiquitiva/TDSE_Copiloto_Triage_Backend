"""Pruebas del cliente de Gemini SIN llamadas de red reales: se simula
`urllib.request.urlopen` para mantener la suite de pruebas rápida,
determinista y sin necesitar una API key."""
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from src import llm_client


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestGetApiKey(unittest.TestCase):
    def test_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(llm_client.LLMUnavailableError):
                llm_client.get_api_key()

    def test_present_key_is_returned(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            self.assertEqual(llm_client.get_api_key(), "fake-key")


class TestEmbedText(unittest.TestCase):
    def test_parses_embedding_values(self):
        fake_payload = {"embedding": {"values": [0.1, 0.2, 0.3]}}
        with patch("urllib.request.urlopen", return_value=_FakeResponse(fake_payload)):
            result = llm_client.embed_text("dolor de cabeza", api_key="fake-key")
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_malformed_response_raises(self):
        with patch("urllib.request.urlopen", return_value=_FakeResponse({"unexpected": True})):
            with self.assertRaises(llm_client.LLMUnavailableError):
                llm_client.embed_text("texto", api_key="fake-key")

    def test_task_type_is_sent_in_request_payload(self):
        """GeminiRetriever depende de que task_type efectivamente viaje en el
        payload para lograr la codificación asimétrica consulta/documento."""
        fake_payload = {"embedding": {"values": [0.1]}}
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(fake_payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm_client.embed_text("texto", api_key="fake-key", task_type="RETRIEVAL_DOCUMENT")
        self.assertEqual(captured["body"].get("taskType"), "RETRIEVAL_DOCUMENT")

    def test_no_task_type_omits_field_from_payload(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse({"embedding": {"values": [0.1]}})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm_client.embed_text("texto", api_key="fake-key")
        self.assertNotIn("taskType", captured["body"])


def _generation_payload(result: dict) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(result)}]}}]}


class TestGenerateStructured(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.tmp.name) / "llm_cache.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_json_from_candidate_text(self):
        fake_payload = _generation_payload({"citation": "N2", "abstain": False, "reasoning": "porque si"})
        with patch.dict("os.environ", {"GEMINI_CACHE_MODE": "off"}), patch(
            "urllib.request.urlopen", return_value=_FakeResponse(fake_payload)
        ):
            result = llm_client.generate_structured("un prompt", api_key="fake-key")
        self.assertEqual(result["citation"], "N2")
        self.assertFalse(result["abstain"])

    def test_api_key_travels_in_header_not_in_url(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = {k.lower(): v for k, v in request.header_items()}
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(_generation_payload({"citation": None, "abstain": True, "reasoning": ""}))

        with patch.dict("os.environ", {"GEMINI_CACHE_MODE": "off"}), patch(
            "urllib.request.urlopen", side_effect=fake_urlopen
        ):
            llm_client.generate_structured("un prompt", api_key="clave-secreta-123")
        self.assertNotIn("clave-secreta-123", captured["url"])
        self.assertNotIn("key=", captured["url"])
        self.assertEqual(captured["headers"]["x-goog-api-key"], "clave-secreta-123")
        self.assertEqual(captured["body"]["generationConfig"]["temperature"], 0.0)

    def test_embedding_key_travels_in_header_not_in_url(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = {k.lower(): v for k, v in request.header_items()}
            return _FakeResponse({"embedding": {"values": [0.1]}})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm_client.embed_text("texto", api_key="clave-secreta-123")
        self.assertNotIn("clave-secreta-123", captured["url"])
        self.assertEqual(captured["headers"]["x-goog-api-key"], "clave-secreta-123")

    def test_model_is_pinned_without_latest_alias(self):
        self.assertNotIn("latest", llm_client.DEFAULT_GENERATION_MODEL)
        self.assertNotIn("latest", llm_client.DEFAULT_EMBEDDING_MODEL)

    def test_readwrite_caches_and_second_call_does_not_hit_network(self):
        payload = _generation_payload({"citation": "N2", "abstain": False, "reasoning": "x"})
        with patch.dict("os.environ", {"GEMINI_CACHE_MODE": "readwrite"}):
            with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)) as mock_open:
                llm_client.generate_structured("p", api_key="k", prompt_version="v1", cache_path=self.cache_path)
                self.assertEqual(mock_open.call_count, 1)
            with patch("urllib.request.urlopen", side_effect=AssertionError("no debe llamar a la red")):
                result = llm_client.generate_structured(
                    "p", api_key="k", prompt_version="v1", cache_path=self.cache_path
                )
        self.assertEqual(result["citation"], "N2")
        cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.assertIn(llm_client.llm_cache_key(llm_client.GENERATION_MODEL, "v1", "p"), cache)

    def test_readonly_hit_works_without_api_key(self):
        key = llm_client.llm_cache_key(llm_client.GENERATION_MODEL, "v1", "p")
        self.cache_path.write_text(json.dumps({key: {"citation": "N3", "abstain": False}}), encoding="utf-8")
        with patch.dict("os.environ", {"GEMINI_CACHE_MODE": "readonly"}, clear=True):
            result = llm_client.generate_structured("p", prompt_version="v1", cache_path=self.cache_path)
        self.assertEqual(result["citation"], "N3")

    def test_readonly_without_entry_raises_unavailable(self):
        with patch.dict("os.environ", {"GEMINI_CACHE_MODE": "readonly", "GEMINI_API_KEY": "k"}), patch(
            "urllib.request.urlopen", side_effect=AssertionError("no debe llamar a la red")
        ):
            with self.assertRaises(llm_client.LLMUnavailableError):
                llm_client.generate_structured("otro prompt", prompt_version="v1", cache_path=self.cache_path)

    def test_prompt_version_changes_cache_key(self):
        self.assertNotEqual(
            llm_client.llm_cache_key("m", "v1", "p"), llm_client.llm_cache_key("m", "v2", "p")
        )

    def test_invalid_cache_mode_raises(self):
        with patch.dict("os.environ", {"GEMINI_CACHE_MODE": "siempre"}):
            with self.assertRaises(ValueError):
                llm_client.get_cache_mode()


class TestRetryBehavior(unittest.TestCase):
    def test_retries_on_retryable_status_then_succeeds(self):
        fake_payload = {"embedding": {"values": [1.0]}}
        error_503 = urllib.error.HTTPError(
            url="http://x", code=503, msg="unavailable", hdrs=None, fp=MagicMock(read=lambda: b"{}")
        )
        with patch(
            "urllib.request.urlopen", side_effect=[error_503, _FakeResponse(fake_payload)]
        ), patch("time.sleep"):
            result = llm_client.embed_text("texto", api_key="fake-key")
        self.assertEqual(result, [1.0])

    def test_non_retryable_status_raises_immediately(self):
        error_404 = urllib.error.HTTPError(
            url="http://x", code=404, msg="not found", hdrs=None, fp=MagicMock(read=lambda: b"{}")
        )
        with patch("urllib.request.urlopen", side_effect=error_404):
            with self.assertRaises(llm_client.LLMUnavailableError):
                llm_client.embed_text("texto", api_key="fake-key")


if __name__ == "__main__":
    unittest.main()
