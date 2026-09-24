"""Guardarraíl monótono: combina la sugerencia del modelo con las alarmas del
componente 1 (``src/extraction.py``).

    n_final = min(n_modelo, n_reglas)

Por construcción SOLO puede aumentar la urgencia (número menor), nunca
disminuirla. Casos:

======================  ==========  ==============================================
Modelo                  Reglas      Resultado
======================  ==========  ==============================================
nivel n_m con cita      None        n_final = n_m, cita del modelo
nivel n_m con cita      n_r         n_final = min(n_m, n_r); si gana la regla, la
                                    cita final es la de la regla y la del modelo
                                    se conserva en ``model_citation``
abstención              n_r         escalated = True; n_r queda como referencia
                                    (con la cita de su alarma en ``alarms``),
                                    NO como sugerencia del modelo
abstención              None        escalated = True, sin nivel
======================  ==========  ==============================================

Invariantes (cubiertos por pruebas): ``n_final <= n_modelo`` y
``n_final <= n_reglas`` cuando existen; nunca hay un nivel sin cita; toda
abstención termina con ``escalated = True``.
"""
from __future__ import annotations

from dataclasses import replace

from .extraction import ExtractionResult
from .generator import Suggestion

REASON_RAISED_BY_RULE = "elevado_por_regla_de_alarma"


def finalize_without_guardrail(suggestion: Suggestion) -> Suggestion:
    """Completa los campos del guardarraíl cuando está desactivado (celda
    E5, E1-E4): n_final = n_modelo. La abstención igual se escala, porque
    ninguna abstención puede ser silenciosa."""
    level = None if suggestion.abstained else suggestion.level
    return replace(
        suggestion,
        n_modelo=level,
        n_reglas=None,
        n_final=level,
        escalated=suggestion.abstained,
        model_citation=None if suggestion.abstained else suggestion.citation,
        alarms=[],
    )


def apply_guardrail(suggestion: Suggestion, extraction: ExtractionResult) -> Suggestion:
    n_r = extraction.n_reglas
    alarms = [a.to_dict() for a in extraction.alarms]

    if suggestion.abstained or suggestion.level is None:
        # El modelo se abstuvo: se escala SIEMPRE. Si hay una alarma, su nivel
        # queda como referencia visible (n_reglas + alarms), pero no se
        # presenta como sugerencia del modelo: level y n_final quedan en None.
        return replace(
            suggestion,
            level=None,
            citation=None,
            abstained=True,
            n_modelo=None,
            n_reglas=n_r,
            n_final=None,
            escalated=True,
            model_citation=None,
            alarms=alarms,
        )

    n_m = suggestion.level
    if n_r is None or n_m <= n_r:
        return replace(
            suggestion,
            n_modelo=n_m,
            n_reglas=n_r,
            n_final=n_m,
            escalated=False,
            model_citation=suggestion.citation,
            alarms=alarms,
        )

    # La regla es más urgente: gana la regla, con su propia cita.
    winner = extraction.most_urgent()
    return replace(
        suggestion,
        level=n_r,
        citation=winner.citation,
        reason=REASON_RAISED_BY_RULE,
        n_modelo=n_m,
        n_reglas=n_r,
        n_final=n_r,
        escalated=False,
        model_citation=suggestion.citation,
        alarms=alarms,
    )
