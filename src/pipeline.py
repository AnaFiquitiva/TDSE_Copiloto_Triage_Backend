"""Orquesta un caso individual a través de ingesta -> recuperación ->
generación -> registro de auditoría (Sección 4 del documento del proyecto).

Dos rutas (ver README.md, sección "Arquitectura"):

- ``backend="gemini"`` (**por defecto**, ruta principal): *embeddings* y
  generación vía la API de Gemini (`GeminiRetriever` + `LLMBackedGenerator`).
  Si falta la API key o la API falla tras los reintentos, ese caso cae a la
  ruta de respaldo determinista y queda marcado como modo degradado
  (``backend_used="deterministic_fallback"``, ``degraded=True``) tanto en la
  sugerencia como en el registro de auditoría.
- ``backend="deterministic"``: TF-IDF (`Retriever`) + vecino-más-cercano
  (`MinimalGenerator`), pedido de forma explícita (p. ej. en E1-E4, para que
  el estudio de recuperación sea reproducible sin red). Queda registrado como
  ``backend_used="deterministic"`` y NO es modo degradado.

Flujo de ``run_case`` (Sección 4): extracción de alarmas (componente 1) ->
ruta principal o respaldo -> guardarraíl monótono ``min(n_modelo, n_reglas)``
si ``use_guardrail`` (también sobre el respaldo) -> registro de auditoría.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import llm_client
from .audit import DETERMINISTIC_GENERATOR_VERSION, AuditLog, build_audit_record
from .extraction import AlarmDetector
from .generator import MIN_RELATIVE_MARGIN, MIN_SCORE, MinimalGenerator, Suggestion
from .guardrail import apply_guardrail, finalize_without_guardrail
from .ingest import Chunk, corpus_hash, load_corpus
from .llm_client import LLMUnavailableError
from .llm_generator import PROMPT_VERSION, LLMBackedGenerator
from .retrieval import GeminiRetriever, Retriever

DEFAULT_AUDIT_PATH = Path(__file__).resolve().parent.parent / "results" / "audit_log.jsonl"

BACKENDS = ("gemini", "deterministic")
BACKEND_GEMINI = "gemini"
BACKEND_FALLBACK = "deterministic_fallback"
BACKEND_DETERMINISTIC = "deterministic"


@lru_cache(maxsize=None)
def _cached_corpus(corpus_format: str) -> tuple[Chunk, ...]:
    return tuple(load_corpus(corpus_format))


@lru_cache(maxsize=None)
def _cached_corpus_hash(corpus_format: str) -> str:
    return corpus_hash(corpus_format)


def build_generator(corpus_format: str, embedding_mode: str, k: int = 5) -> MinimalGenerator:
    chunks = list(_cached_corpus(corpus_format))
    retriever = Retriever(chunks, embedding_mode=embedding_mode)
    return MinimalGenerator(retriever, k=k)


def build_llm_generator(corpus_format: str, k: int = 5) -> LLMBackedGenerator:
    chunks = list(_cached_corpus(corpus_format))
    retriever = GeminiRetriever(chunks)
    return LLMBackedGenerator(retriever, k=k)


class CopilotoPipeline:
    """Pipeline reproducible para una configuración (corpus_format x
    embedding_mode x backend). Ver la tabla de celdas en la Sección 5."""

    def __init__(
        self,
        corpus_format: str,
        embedding_mode: str = "clinical_es",
        k: int = 5,
        audit_path: Path = DEFAULT_AUDIT_PATH,
        backend: str = "gemini",
        llm_generator=None,
        use_guardrail: bool = True,
        detector: AlarmDetector | None = None,
    ):
        if backend not in BACKENDS:
            raise ValueError("backend debe ser 'gemini' o 'deterministic'")
        self.corpus_format = corpus_format
        self.embedding_mode = embedding_mode
        self.k = k
        self.backend = backend
        self.use_guardrail = use_guardrail
        self.detector = detector or AlarmDetector()
        self.corpus_hash = _cached_corpus_hash(corpus_format)
        self.generator = build_generator(corpus_format, embedding_mode, k=k)
        self._llm_generator = llm_generator
        if backend == BACKEND_GEMINI and llm_generator is None:
            # Construir el retriever de Gemini de una vez (dispara los
            # embeddings del corpus, cacheados en disco); si falla aquí
            # mismo (p. ej. sin API key ni caché), cada llamada a run_case()
            # usará la ruta de respaldo, marcada como degradada.
            try:
                self._llm_generator = build_llm_generator(corpus_format, k=k)
            except LLMUnavailableError:
                self._llm_generator = None
        self.audit_log = AuditLog(audit_path)
        self.last_record: dict | None = None

    def _thresholds(self) -> dict:
        return {"MIN_SCORE": MIN_SCORE, "MIN_RELATIVE_MARGIN": MIN_RELATIVE_MARGIN, "k": self.k}

    def _suggest(self, patient_text: str) -> Suggestion:
        """Ruta principal o respaldo; nunca deja el caso sin respuesta."""
        if self.backend == BACKEND_GEMINI and self._llm_generator is not None:
            try:
                suggestion = self._llm_generator.suggest(patient_text)
                suggestion.backend_used = BACKEND_GEMINI
                suggestion.degraded = False
                return suggestion
            except LLMUnavailableError:
                pass
        suggestion = self.generator.suggest(patient_text)
        if self.backend == BACKEND_GEMINI:
            suggestion.backend_used = BACKEND_FALLBACK
            suggestion.degraded = True
        else:
            suggestion.backend_used = BACKEND_DETERMINISTIC
            suggestion.degraded = False
        return suggestion

    def run_case(self, case_id: str, patient_text: str) -> Suggestion:
        extraction = self.detector.extract(patient_text)
        suggestion = self._suggest(patient_text)
        if self.use_guardrail:
            suggestion = apply_guardrail(suggestion, extraction)
        else:
            suggestion = finalize_without_guardrail(suggestion)

        if suggestion.backend_used == BACKEND_GEMINI:
            generation_model = llm_client.GENERATION_MODEL
            embedding_model = llm_client.EMBEDDING_MODEL
            prompt_version = PROMPT_VERSION
        else:
            generation_model = DETERMINISTIC_GENERATOR_VERSION
            embedding_model = f"tfidf-{self.embedding_mode}"
            prompt_version = None

        record = build_audit_record(
            case_id=case_id,
            patient_text=patient_text,
            corpus_format=self.corpus_format,
            embedding_mode=self.embedding_mode,
            suggestion=suggestion,
            corpus_hash=self.corpus_hash,
            generation_model=generation_model,
            embedding_model=embedding_model,
            prompt_version=prompt_version,
            thresholds=self._thresholds(),
            use_guardrail=self.use_guardrail,
            alarm_rules_version=self.detector.version if self.use_guardrail else None,
        )
        self.last_record = self.audit_log.append(record)
        return suggestion
