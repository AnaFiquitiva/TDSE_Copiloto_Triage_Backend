"""Componente 3 (versión mínima): produce una sugerencia de nivel de triage
con cita obligatoria, o se abstiene si la evidencia recuperada es
insuficiente o contradictoria.

Tal como se describe en la Sección 5 del documento del proyecto, esta versión
mínima no reemplaza la capa de generación conceptual completa (Sección 4):
es, deliberadamente, un clasificador simple de vecino-más-cercano sobre los
fragmentos recuperados por el ``Retriever``, suficiente para hacer
observables H1, H2 y H4 sin comprometer el foco del semestre en el
componente de recuperación.

Regla de decisión: el nivel sugerido es el del fragmento con mayor similitud
que tenga un nivel asociado. Se exige, además, que ese fragmento supere un
umbral mínimo de similitud absoluta (evidencia suficiente) y que su
similitud aventaje con un margen mínimo al mejor fragmento recuperado que
proponga un nivel *distinto* (evidencia no contradictoria); de lo contrario
el sistema se abstiene en lugar de forzar una elección entre dos criterios
igualmente respaldados por la evidencia recuperada.

El diseño deja un punto de extensión explícito para un despliegue real que
use un modelo de lenguaje vía API en lugar de esta regla determinista; se
mantiene deshabilitado por defecto para que la evaluación sea 100% offline y
reproducible sin credenciales externas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .retrieval import RankedChunk, Retriever

# Similitud mínima que debe tener el mejor fragmento con nivel asociado para
# considerar que hay evidencia suficiente.
MIN_SCORE = 0.08

# Margen relativo mínimo que el mejor fragmento debe aventajar al mejor
# fragmento que proponga un nivel distinto, para no considerar la evidencia
# contradictoria. Un margen de 0.0 aceptaría cualquier empate; con 0.20 se
# exige que el ganador supere al segundo lugar en al menos un 20% de su score.
MIN_RELATIVE_MARGIN = 0.20


@dataclass
class Suggestion:
    level: int | None
    citation: str | None
    confidence: float
    abstained: bool
    reason: str
    retrieved: list[RankedChunk] = field(default_factory=list)
    # Ruta que produjo la sugerencia: "gemini" (ruta principal),
    # "deterministic_fallback" (se pidió Gemini pero falló y se usó el
    # respaldo) o "deterministic" (ruta determinista pedida explícitamente).
    backend_used: str = "deterministic"
    # True solo cuando se usó la ruta de respaldo en lugar de la principal.
    degraded: bool = False


class MinimalGenerator:
    """Clasificador determinista de nivel de triage con cita obligatoria."""

    def __init__(self, retriever: Retriever, k: int = 5):
        self.retriever = retriever
        self.k = k

    def suggest(self, patient_text: str) -> Suggestion:
        retrieved = self.retriever.retrieve(patient_text, k=self.k)
        candidates = [rc for rc in retrieved if rc.score > 0 and rc.chunk.level is not None]

        if not candidates:
            return Suggestion(
                level=None,
                citation=None,
                confidence=0.0,
                abstained=True,
                reason="evidencia_insuficiente",
                retrieved=retrieved,
            )

        best = candidates[0]
        if best.score < MIN_SCORE:
            return Suggestion(
                level=None,
                citation=None,
                confidence=0.0,
                abstained=True,
                reason="evidencia_insuficiente",
                retrieved=retrieved,
            )

        best_other = next(
            (rc for rc in candidates[1:] if rc.chunk.level != best.chunk.level), None
        )
        runner_up_score = best_other.score if best_other is not None else 0.0
        margin = (best.score - runner_up_score) / best.score
        confidence = best.score / (best.score + runner_up_score) if runner_up_score else 1.0

        if margin < MIN_RELATIVE_MARGIN:
            return Suggestion(
                level=None,
                citation=None,
                confidence=confidence,
                abstained=True,
                reason="evidencia_contradictoria",
                retrieved=retrieved,
            )

        return Suggestion(
            level=best.chunk.level,
            citation=best.chunk.chunk_id,
            confidence=confidence,
            abstained=False,
            reason="ok",
            retrieved=retrieved,
        )
