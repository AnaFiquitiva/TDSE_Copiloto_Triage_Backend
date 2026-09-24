"""Compara la ruta determinista (TF-IDF + vecino-más-cercano, sin red) contra
la ruta principal Gemini (embeddings + generación reales) sobre las mismas
viñetas de `data/cases.json`, usando siempre el corpus reformateado (para
aislar el efecto de la ruta del efecto del formato del corpus, que ya se
evalúa por separado en `run_experiment.py`).

Requiere `GEMINI_API_KEY` o una caché de respuestas (ver README.md, sección
"Variables de entorno"). Los casos en que la ruta Gemini cae al respaldo
(``backend_used="deterministic_fallback"``) se reportan APARTE y no se
mezclan con los que sí respondió Gemini: mezclarlos haría que la columna
"Gemini" midiera en parte al determinista.

Uso:
    python -m experiments.compare_backends
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from experiments.run_experiment import (  # noqa: E402
    _f,
    _nan_to_none,
    baseline_levels,
    collect_predictions,
    evaluate_predictions,
    load_cases,
)
from src.pipeline import CopilotoPipeline  # noqa: E402

RESULTS_DIR = ROOT / "results"
CORPUS_FORMAT = "reformatted"


def _run(backend: str, cases: list[dict]) -> list[dict]:
    audit_path = RESULTS_DIR / f"audit_compare_{backend}.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    pipeline = CopilotoPipeline(corpus_format=CORPUS_FORMAT, backend=backend, audit_path=audit_path)
    return collect_predictions(pipeline, cases)


def _evaluate_subset(name: str, predictions: list[dict], cases: list[dict], c0: dict) -> dict | None:
    ids = {p["case_id"] for p in predictions}
    subset = [c for c in cases if c["case_id"] in ids]
    if not subset:
        return None
    result = evaluate_predictions(name, predictions, subset, c0)
    result["n_casos"] = len(subset)
    return result


def compare(cases: list[dict]) -> dict:
    c0 = baseline_levels(cases)
    det_preds = _run("deterministic", cases)
    gem_preds = _run("gemini", cases)

    usage = Counter(p["backend_used"] for p in gem_preds)
    served_ids = {p["case_id"] for p in gem_preds if p["backend_used"] == "gemini"}

    return {
        "backend_realmente_usado": dict(usage),
        # Comparación justa: ambas rutas sobre los MISMOS casos que respondió Gemini.
        "gemini": _evaluate_subset(
            "gemini", [p for p in gem_preds if p["case_id"] in served_ids], cases, c0
        ),
        "deterministic_mismos_casos": _evaluate_subset(
            "deterministic", [p for p in det_preds if p["case_id"] in served_ids], cases, c0
        ),
        # Referencia: determinista sobre todos los casos.
        "deterministic_todos": _evaluate_subset("deterministic", det_preds, cases, c0),
        # Aparte: casos en que la ruta Gemini cayó al respaldo.
        "respaldo": _evaluate_subset(
            "deterministic_fallback", [p for p in gem_preds if p["case_id"] not in served_ids], cases, c0
        ),
    }


_ROWS = [
    ("Casos", "n_casos", 0),
    ("Casos elegibles", "n_eligible_cases", 0),
    ("Cobertura", "cobertura", 2),
    ("S (sub-triage ponderado)", "S_sub_triage", 3),
    ("Sensibilidad I-II", "sensibilidad_I_II", 3),
    ("Recall@1", "recall_at_1", 3),
    ("Recall@k", "recall_at_k", 3),
    ("MRR", "mrr", 3),
    ("Sobre-triage", "sobre_triage", 3),
    ("Abstención correcta", "tasa_abstencion_correcta", 3),
    ("Abstención indebida", "tasa_abstencion_indebida", 3),
]


def _cell(result: dict | None, key: str, digits: int) -> str:
    if result is None:
        return "—"
    value = result.get(key)
    if isinstance(value, int) and digits == 0:
        return str(value)
    return _f(value, digits)


def format_markdown(comparison: dict, n_total_cases: int) -> str:
    gem = comparison["gemini"]
    det_same = comparison["deterministic_mismos_casos"]
    det_all = comparison["deterministic_todos"]
    fallback = comparison["respaldo"]
    lines = [
        "# Comparación de rutas: determinista vs. Gemini",
        "",
        f"Generado por `experiments/compare_backends.py` sobre las {n_total_cases} viñetas de "
        f"`data/cases.json`, con corpus `{CORPUS_FORMAT}`.",
        "",
        f"Ruta realmente usada al pedir Gemini, por caso: {comparison['backend_realmente_usado']}. "
        "Los casos que cayeron al respaldo (`deterministic_fallback`) se reportan en su propia "
        "columna y NO se mezclan con los de Gemini.",
        "",
        "| Métrica | Gemini (casos que respondió) | Determinista (mismos casos) | "
        "Determinista (todos) | Respaldo (casos degradados) |",
        "|---|---|---|---|---|",
    ]
    for label, key, digits in _ROWS:
        lines.append(
            f"| {label} | {_cell(gem, key, digits)} | {_cell(det_same, key, digits)} | "
            f"{_cell(det_all, key, digits)} | {_cell(fallback, key, digits)} |"
        )
    if gem is None:
        lines += [
            "",
            "**Gemini no respondió ningún caso** (sin `GEMINI_API_KEY` ni caché de respuestas): "
            "no hay comparación que reportar.",
        ]
    lines += [
        "",
        f"Con {n_total_cases} casos esta comparación es ilustrativa, no concluyente estadísticamente.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    cases = load_cases()
    print("Evaluando ruta determinista y ruta Gemini (esta última puede llamar a la API)...")
    comparison = compare(cases)
    summary = format_markdown(comparison, n_total_cases=len(cases))
    (RESULTS_DIR / "backend_comparison.md").write_text(summary, encoding="utf-8")
    (RESULTS_DIR / "backend_comparison.json").write_text(
        json.dumps(_nan_to_none(comparison), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
