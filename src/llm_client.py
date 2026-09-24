"""Cliente mínimo para la API de Gemini (embeddings + generación), usado por la
ruta principal del copiloto (ver README.md, sección "Arquitectura").

Deliberadamente NO usa el SDK oficial `google-generativeai` para no agregar
una dependencia externa al proyecto: solo usa `urllib` (biblioteca estándar).

**Manejo de la API key**: se lee EXCLUSIVAMENTE de la variable de entorno
`GEMINI_API_KEY` y viaja en el encabezado `x-goog-api-key`, nunca en la URL
(las URL suelen terminar en logs de proxies y en trazas de error). Este
módulo nunca la recibe como literal en el código ni la escribe a ningún
archivo.

**Reproducibilidad**:

- Los modelos están fijados a un ID estable (sin alias móviles tipo
  ``-latest``) y se pueden cambiar con las variables de entorno
  ``GEMINI_GENERATION_MODEL`` y ``GEMINI_EMBEDDING_MODEL``.
- La generación usa ``temperature = 0.0``.
- Las respuestas del LLM se cachean en disco
  (``data/.llm_response_cache.json``, excluido de git) con clave
  ``sha256(modelo + PROMPT_VERSION + prompt)``. El modo de la caché se
  controla con ``GEMINI_CACHE_MODE``:

  * ``readwrite`` (por defecto): usa la caché y guarda las respuestas nuevas;
  * ``readonly``: repetición sin red; si falta una entrada, lanza
    ``LLMUnavailableError`` en lugar de llamar a la API;
  * ``off``: ni lee ni escribe la caché.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# ID estable (sin sufijo -latest) para que dos corridas separadas en el
# tiempo usen el mismo modelo. Ver README.md, sección "Variables de entorno".
DEFAULT_GENERATION_MODEL = "gemini-2.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = os.environ.get("GEMINI_GENERATION_MODEL", DEFAULT_GENERATION_MODEL)
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
GENERATION_TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30
# Códigos de error transitorios sobre los que vale la pena reintentar.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

CACHE_MODES = ("readwrite", "readonly", "off")
DEFAULT_LLM_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / ".llm_response_cache.json"
)


class LLMUnavailableError(Exception):
    """Se lanza cuando el backend Gemini no puede usarse: falta la API key,
    hay un error de red, la respuesta no tiene el formato esperado, o la
    caché está en modo ``readonly`` y no tiene la entrada pedida. El llamador
    (CopilotoPipeline) usa esto como señal para caer a la ruta de respaldo
    determinista, marcada como modo degradado."""


def get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMUnavailableError(
            "No se encontró la variable de entorno GEMINI_API_KEY. Configúrala "
            "con tu propia API key (ver README.md, sección 'Variables de "
            "entorno'); sin ella, el copiloto usa la ruta de respaldo "
            "determinista (modo degradado)."
        )
    return api_key


def get_cache_mode() -> str:
    """Modo de caché vigente, leído en cada llamada (no al importar) para que
    un experimento pueda alternar entre ``readwrite`` y ``readonly`` entre
    repeticiones sin reiniciar el proceso."""
    mode = os.environ.get("GEMINI_CACHE_MODE", "readwrite").strip().lower()
    if mode not in CACHE_MODES:
        raise ValueError(
            f"GEMINI_CACHE_MODE debe ser uno de {CACHE_MODES}, no {mode!r}"
        )
    return mode


MAX_RETRY_DELAY_SECONDS = 60.0
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def _retry_delay_seconds(body: str, default: float) -> float:
    """Espera sugerida por la API en un 429 (campo ``RetryInfo.retryDelay``,
    p. ej. ``"38s"``), acotada a ``MAX_RETRY_DELAY_SECONDS``. Respetarla evita
    gastar los reintentos antes de que se libere la cuota por minuto."""
    match = _RETRY_DELAY_RE.search(body)
    if not match:
        return default
    return min(float(match.group(1)) + 1.0, MAX_RETRY_DELAY_SECONDS)


def _post_json(url: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        wait = RETRY_BACKOFF_SECONDS * attempt
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = LLMUnavailableError(
                f"Gemini API devolvió HTTP {exc.code}: {body[:300]}"
            )
            if exc.code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                raise last_error from exc
            if exc.code == 429:
                wait = _retry_delay_seconds(body, default=wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = LLMUnavailableError(f"Error de red llamando a Gemini: {exc}")
            if attempt == MAX_RETRIES:
                raise last_error from exc
        time.sleep(wait)

    raise last_error or LLMUnavailableError("Fallo desconocido llamando a Gemini")


def embed_text(text: str, api_key: str | None = None, task_type: str | None = None) -> list[float]:
    """Obtiene el vector de embedding real de Gemini para `text`.

    `task_type` activa la codificación asimétrica de `gemini-embedding-001`:
    un mismo texto produce un vector distinto según si se declara como
    "RETRIEVAL_DOCUMENT" (fragmentos del corpus) o "RETRIEVAL_QUERY" (el
    relato del paciente). Usar el `task_type` correcto en cada lado mejora
    la calidad de la recuperación frente a no declarar ninguno (equivalente
    a "RETRIEVAL_QUERY" para ambos lados); ver `GeminiRetriever`, que es
    quien cachea estos vectores en disco.
    """
    api_key = api_key or get_api_key()
    url = f"{API_BASE}/{EMBEDDING_MODEL}:embedContent"
    payload: dict = {"content": {"parts": [{"text": text}]}}
    if task_type:
        payload["taskType"] = task_type
    try:
        response = _post_json(url, payload, api_key)
        return response["embedding"]["values"]
    except (KeyError, TypeError) as exc:
        raise LLMUnavailableError(f"Respuesta de embeddings con formato inesperado: {exc}") from exc


# Esquema de salida estructurada: fuerza a Gemini a responder SOLO con estos
# tres campos (ver README.md: la cita se valida siempre contra la lista de
# candidatos que nosotros mismos recuperamos; el nivel nunca se toma de un
# campo de texto libre del modelo).
_GENERATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "citation": {
            "type": "STRING",
            "nullable": True,
            "description": "El id exacto del candidato elegido, o null si no hay evidencia suficiente.",
        },
        "abstain": {
            "type": "BOOLEAN",
            "description": "true si ningún candidato describe claramente el caso del relato.",
        },
        "reasoning": {"type": "STRING", "description": "Justificación breve, en español."},
    },
    "required": ["abstain", "reasoning"],
}


def llm_cache_key(model: str, prompt_version: str, prompt: str) -> str:
    """Clave de la caché de respuestas: sha256(modelo + PROMPT_VERSION + prompt)."""
    return hashlib.sha256((model + prompt_version + prompt).encode("utf-8")).hexdigest()


def _load_llm_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_llm_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")


def generate_structured(
    prompt: str,
    api_key: str | None = None,
    prompt_version: str = "",
    cache_path: Path | None = None,
) -> dict:
    """Llama a Gemini con salida JSON forzada al esquema `_GENERATION_SCHEMA`,
    con temperatura 0 y caché de respuestas en disco (ver docstring del
    módulo para los modos de la caché).

    La caché se consulta ANTES de buscar la API key, para que una repetición
    en modo ``readonly`` funcione sin clave y sin red.
    """
    mode = get_cache_mode()
    cache_path = cache_path or DEFAULT_LLM_CACHE_PATH
    key = llm_cache_key(GENERATION_MODEL, prompt_version, prompt)

    cache: dict = {}
    if mode != "off":
        cache = _load_llm_cache(cache_path)
        if key in cache:
            return cache[key]
    if mode == "readonly":
        raise LLMUnavailableError(
            "GEMINI_CACHE_MODE=readonly y la caché de respuestas no tiene esta "
            "entrada; no se llama a la API en modo de repetición."
        )

    api_key = api_key or get_api_key()
    url = f"{API_BASE}/{GENERATION_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": GENERATION_TEMPERATURE,
            "responseMimeType": "application/json",
            "responseSchema": _GENERATION_SCHEMA,
        },
    }
    try:
        response = _post_json(url, payload, api_key)
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMUnavailableError(f"Respuesta de generación con formato inesperado: {exc}") from exc

    if mode == "readwrite":
        cache[key] = result
        _save_llm_cache(cache_path, cache)
    return result
