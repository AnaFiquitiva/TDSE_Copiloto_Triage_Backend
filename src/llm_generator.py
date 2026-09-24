"""Componente 3, ruta principal: usa Gemini para decidir, entre los
fragmentos recuperados, cuál describe mejor el relato del paciente — en vez
de la regla de vecino-más-cercano de `MinimalGenerator`.

**Principio de seguridad (no negociable):** el modelo de lenguaje solo puede
elegir un `citation` de la lista de candidatos que NOSOTROS le dimos (los
fragmentos recuperados por el `Retriever`/`GeminiRetriever`); nunca se le
pide ni se le permite inventar un nivel directamente. Si el modelo devuelve
un id que no está en la lista de candidatos, o pide abstenerse, el resultado
es una abstención — nunca se "corrige" ni se acepta un nivel que no venga
acompañado de una cita verificable contra el propio corpus. Esto preserva,
incluso con un LLM real en el loop, el principio de "evidencia y
trazabilidad" del documento del proyecto.

Implementa la misma interfaz que `MinimalGenerator` (`.suggest(texto) ->
Suggestion`), así que `CopilotoPipeline` puede intercambiarlos sin cambios en
el resto del código.
"""
from __future__ import annotations

from .generator import Suggestion
from .llm_client import generate_structured
from .retrieval import RankedChunk

# Versión del prompt: forma parte de la clave de la caché de respuestas y del
# registro de auditoría. SÚBELA (p. ej. "triage-selector-v2") cada vez que
# cambie `_PROMPT_TEMPLATE` o `_format_candidates`, para que las respuestas
# cacheadas de un prompt anterior no se reutilicen por error.
PROMPT_VERSION = "triage-selector-v1"


def _default_generate(prompt: str) -> dict:
    return generate_structured(prompt, prompt_version=PROMPT_VERSION)


_PROMPT_TEMPLATE = """Eres un asistente de apoyo al triage de urgencias en Colombia. \
NUNCA diagnostiques ni inventes información clínica. Tu única tarea es decidir, \
entre los fragmentos normativos recuperados abajo, cuál describe mejor el relato \
del paciente, y devolver su id EXACTO tal como aparece en la lista.

Si ningún fragmento describe claramente el caso, o el relato no da información \
suficiente para decidir con confianza, responde con abstain=true y deja \
citation en null. Nunca inventes un id que no esté en la lista de candidatos.

Relato del paciente:
\"\"\"{patient_text}\"\"\"

Fragmentos candidatos:
{candidate_lines}
"""


def _format_candidates(candidates: list[RankedChunk]) -> str:
    return "\n".join(
        f'- id="{rc.chunk.chunk_id}" (nivel {rc.chunk.level}): '
        f'{rc.chunk.text.strip()[:300].replace(chr(10), " ")}'
        for rc in candidates
    )


class LLMBackedGenerator:
    """Generador de la ruta principal, respaldado por Gemini (requiere
    `GEMINI_API_KEY` o una caché de respuestas). Ver el aviso de seguridad en
    el docstring del módulo."""

    def __init__(self, retriever, k: int = 5, generate_fn=_default_generate):
        self.retriever = retriever
        self.k = k
        self._generate_fn = generate_fn

    def suggest(self, patient_text: str) -> Suggestion:
        retrieved = self.retriever.retrieve(patient_text, k=self.k)
        candidates = [rc for rc in retrieved if rc.chunk.level is not None]

        if not candidates:
            return Suggestion(
                level=None,
                citation=None,
                confidence=0.0,
                abstained=True,
                reason="evidencia_insuficiente",
                retrieved=retrieved,
            )

        prompt = _PROMPT_TEMPLATE.format(
            patient_text=patient_text, candidate_lines=_format_candidates(candidates)
        )

        # LLMUnavailableError se propaga a propósito: CopilotoPipeline decide
        # si cae de vuelta al backend determinista ante un fallo de red/API.
        result = self._generate_fn(prompt)

        abstain = bool(result.get("abstain", True))
        citation = result.get("citation")
        matching = next((rc for rc in candidates if rc.chunk.chunk_id == citation), None)

        if abstain or matching is None:
            reason = "abstencion_del_modelo" if abstain else "cita_invalida_del_modelo"
            return Suggestion(
                level=None,
                citation=None,
                confidence=0.0,
                abstained=True,
                reason=reason,
                retrieved=retrieved,
            )

        return Suggestion(
            level=matching.chunk.level,
            citation=matching.chunk.chunk_id,
            # La confianza reportada es el score de RECUPERACIÓN del fragmento
            # elegido, no una confianza autorreportada por el LLM (que no es
            # confiable ni se le pide en el esquema de salida).
            confidence=matching.score,
            abstained=False,
            reason="ok_llm",
            retrieved=retrieved,
        )
