"""Métricas formales definidas en la Sección 5 del documento del proyecto.

Todas las funciones son deterministas y no dependen de librerías externas de
ciencia de datos, para que la evaluación sea reproducible con solo la
biblioteca estándar de Python.
"""
from __future__ import annotations

import math
import random
from collections import Counter


def sub_triage_rate(pairs: list[tuple[int, int]]) -> float:
    """Calcula S: tasa de sub-triage ponderada por distancia (Sección 5).

    ``pairs`` es una lista de tuplas ``(nivel_referencia, nivel_sugerido)``.
    Solo cuenta como sub-triage cuando el nivel sugerido es MENOS urgente
    (número mayor) que el de referencia.
    """
    if not pairs:
        return 0.0
    total = sum(max(0, sug - ref) for ref, sug in pairs)
    return total / len(pairs)


def sub_triage_by_level(pairs_with_ref_level: list[tuple[int, int]]) -> dict[int, float]:
    """Desglosa S por cada nivel de referencia (I-V)."""
    by_level: dict[int, list[tuple[int, int]]] = {}
    for ref, sug in pairs_with_ref_level:
        by_level.setdefault(ref, []).append((ref, sug))
    return {level: sub_triage_rate(p) for level, p in sorted(by_level.items())}


def sensitivity_high_risk(pairs: list[tuple[int, int]], high_risk_levels=(1, 2)) -> float:
    """Sensibilidad en los niveles de mayor riesgo (I-II): de los casos cuyo
    nivel de referencia es I o II, qué fracción fue sugerida también como I o
    II (es decir, no fue subestimada fuera del rango de alto riesgo)."""
    relevant = [(ref, sug) for ref, sug in pairs if ref in high_risk_levels]
    if not relevant:
        return float("nan")
    correct = sum(1 for ref, sug in relevant if sug in high_risk_levels)
    return correct / len(relevant)


def weighted_kappa(pairs: list[tuple[int, int]], levels=(1, 2, 3, 4, 5)) -> float:
    """Kappa de Cohen ponderado cuadráticamente para escalas ordinales.

    ``pairs`` es una lista de tuplas ``(nivel_evaluador_1, nivel_evaluador_2)``.
    """
    n = len(pairs)
    if n == 0:
        return float("nan")

    levels = list(levels)
    idx = {lvl: i for i, lvl in enumerate(levels)}
    k = len(levels)

    observed = [[0] * k for _ in range(k)]
    for a, b in pairs:
        observed[idx[a]][idx[b]] += 1

    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[r][c] for r in range(k)) for c in range(k)]

    weights = [[(i - j) ** 2 for j in range(k)] for i in range(k)]

    observed_sum = sum(
        weights[i][j] * observed[i][j] for i in range(k) for j in range(k)
    )
    expected_sum = sum(
        weights[i][j] * (row_totals[i] * col_totals[j] / n)
        for i in range(k)
        for j in range(k)
    )

    if expected_sum == 0:
        return 1.0
    return 1 - (observed_sum / expected_sum)


def recall_at_k(hits: list[bool]) -> float:
    """Fracción de casos en los que el fragmento normativo correcto apareció
    entre los top-k recuperados. ``hits`` ya trae, por caso, si hubo acierto."""
    if not hits:
        return float("nan")
    return sum(hits) / len(hits)


def abstention_metrics(
    should_abstain_flags: list[bool], actually_abstained_flags: list[bool]
) -> dict[str, float]:
    """H4: tasa de abstención correcta y tasa de abstención indebida.

    - ``tasa_abstencion_correcta``: de los casos que deberían llevar a
      abstención (evidencia insuficiente), qué fracción el sistema realmente
      se abstuvo.
    - ``tasa_abstencion_indebida``: de los casos con evidencia suficiente,
      qué fracción el sistema se abstuvo de todas formas (costo de
      seguridad excesivo).
    """
    should = [
        actual
        for should_flag, actual in zip(should_abstain_flags, actually_abstained_flags)
        if should_flag
    ]
    should_not = [
        actual
        for should_flag, actual in zip(should_abstain_flags, actually_abstained_flags)
        if not should_flag
    ]
    return {
        "tasa_abstencion_correcta": (sum(should) / len(should)) if should else float("nan"),
        "tasa_abstencion_indebida": (sum(should_not) / len(should_not)) if should_not else float("nan"),
    }


def citation_faithfulness(citation_matches: list[bool]) -> float:
    """Fidelidad de citación (Sección 3.4, criterio de RAGAS adaptado):
    fracción de sugerencias (no abstenidas) cuya cita efectivamente
    corresponde al nivel sugerido según el corpus."""
    if not citation_matches:
        return float("nan")
    return sum(citation_matches) / len(citation_matches)


def confusion_counts(pairs: list[tuple[int, int]]) -> Counter:
    """Matriz de confusión (referencia, sugerido) como conteos, útil para
    inspección manual en el reporte de resultados."""
    return Counter(pairs)


# ---------------------------------------------------------------------------
# Métricas agregadas en el incremento I1 (paper v3). Convención: un nivel
# sugerido ``None`` representa una abstención.
# ---------------------------------------------------------------------------


def coverage(suggested_levels: list[int | None]) -> float:
    """Fracción de casos en los que el sistema respondió (no se abstuvo)."""
    if not suggested_levels:
        return float("nan")
    return sum(1 for lvl in suggested_levels if lvl is not None) / len(suggested_levels)


def paired_sub_triage(
    reference: list[int], sugg_a: list[int | None], sugg_b: list[int | None]
) -> dict:
    """S de A y de B calculado SOLO sobre el subconjunto de casos en que
    ambos responden, para que la comparación no premie a quien se abstiene
    en los casos difíciles. Devuelve ``{"n", "S_a", "S_b"}``."""
    pairs_a: list[tuple[int, int]] = []
    pairs_b: list[tuple[int, int]] = []
    for ref, a, b in zip(reference, sugg_a, sugg_b):
        if a is None or b is None:
            continue
        pairs_a.append((ref, a))
        pairs_b.append((ref, b))
    if not pairs_a:
        return {"n": 0, "S_a": float("nan"), "S_b": float("nan")}
    return {"n": len(pairs_a), "S_a": sub_triage_rate(pairs_a), "S_b": sub_triage_rate(pairs_b)}


def over_triage_rate(pairs: list[tuple[int, int]]) -> float:
    """Tasa de sobre-triage ponderada por distancia, simétrica a S: solo
    cuenta cuando el nivel sugerido es MÁS urgente (número menor) que el de
    referencia. ``pairs`` es ``(nivel_referencia, nivel_sugerido)``."""
    if not pairs:
        return 0.0
    return sum(max(0, ref - sug) for ref, sug in pairs) / len(pairs)


def exact_agreement(pairs: list[tuple[int, int]]) -> float:
    """Fracción de casos cuyo nivel sugerido coincide exactamente con el de
    referencia."""
    if not pairs:
        return float("nan")
    return sum(1 for ref, sug in pairs if ref == sug) / len(pairs)


def reciprocal_rank(ranked_ids: list[str], relevant_id: str) -> float:
    """1/posición del fragmento relevante en el ranking (1-indexado), o 0 si
    no aparece."""
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id == relevant_id:
            return 1.0 / position
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: list[float]) -> float:
    """MRR: media de los rangos recíprocos por caso."""
    if not reciprocal_ranks:
        return float("nan")
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def recall_at_1(ranked_ids_per_case: list[list[str]], relevant_ids: list[str]) -> float:
    """Fracción de casos cuyo primer fragmento recuperado es el relevante."""
    hits = [bool(ids) and ids[0] == rel for ids, rel in zip(ranked_ids_per_case, relevant_ids)]
    return recall_at_k(hits)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Percentil con interpolación lineal (equivalente a numpy 'linear')."""
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(values_or_pairs: list, stat_fn, n: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Intervalo de confianza bootstrap por percentiles, remuestreando los
    CASOS (elementos de ``values_or_pairs``) con reemplazo.

    Es determinista: la semilla fija el generador. Las réplicas en que la
    estadística no está definida (NaN, p. ej. sin casos de nivel I-II en la
    muestra) se descartan. Devuelve ``{"estimate", "ci_low", "ci_high",
    "n_boot"}``.
    """
    estimate = stat_fn(values_or_pairs) if values_or_pairs else float("nan")
    if not values_or_pairs:
        return {"estimate": estimate, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    rng = random.Random(seed)
    size = len(values_or_pairs)
    stats: list[float] = []
    for _ in range(n):
        sample = [values_or_pairs[rng.randrange(size)] for _ in range(size)]
        value = stat_fn(sample)
        if value == value:  # descarta NaN
            stats.append(value)
    stats.sort()
    return {
        "estimate": estimate,
        "ci_low": _percentile(stats, alpha / 2),
        "ci_high": _percentile(stats, 1 - alpha / 2),
        "n_boot": len(stats),
    }


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> float:
    """Valor p exacto (binomial, bilateral) de la prueba de McNemar a partir
    de los pares discordantes: casos que solo A acierta y casos que solo B
    acierta. Con 0 y 5 da p = 2 * 0.5**5 = 0.0625."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)
