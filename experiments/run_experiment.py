"""Ejecuta el diseño experimental 2x2 (E1-E4) más la línea base C0 descrito
en la Sección 5 del documento del proyecto, sobre el conjunto de viñetas
de ``data/cases.json``, y escribe los resultados en ``results/``.

E1-E4 usan SIEMPRE la ruta determinista (``backend="deterministic"``) de forma
explícita: el estudio de recuperación debe ser reproducible sin red ni clave.

Uso:
    python -m experiments.run_experiment
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from src import llm_client, metrics  # noqa: E402
from src.audit import completeness, verify_chain  # noqa: E402
from src.baseline import RuleBasedBaseline  # noqa: E402
from src.extraction import AlarmDetector  # noqa: E402
from src.ingest import canonical_citation  # noqa: E402
from src.llm_generator import PROMPT_VERSION  # noqa: E402
from src.pipeline import CopilotoPipeline  # noqa: E402

CASES_PATH = ROOT / "data" / "cases.json"
RESULTS_DIR = ROOT / "results"
HIGH_RISK = (1, 2)

CELLS = [
    ("E1", "raw", "generic"),
    ("E2", "reformatted", "generic"),
    ("E3", "raw", "clinical_es"),
    ("E4", "reformatted", "clinical_es"),
]


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cases"]


def is_eligible(case: dict) -> bool:
    """Caso con nivel de referencia duro (entra en S, sensibilidad, recall)."""
    return not case["should_abstain"] and not case["ambiguous"] and case["final_level"] is not None


def intra_team_kappa(cases: list[dict]) -> dict:
    pairs = [
        (c["evaluator1"]["level"], c["evaluator2"]["level"])
        for c in cases
        if c["evaluator1"]["level"] is not None and c["evaluator2"]["level"] is not None
    ]
    return {
        "n_pares": len(pairs),
        "kappa_ponderado_cuadratico": metrics.weighted_kappa(pairs),
    }


def baseline_levels(cases: list[dict]) -> dict[str, int]:
    """Nivel que C0 asigna a cada caso (C0 nunca se abstiene)."""
    baseline = RuleBasedBaseline()
    return {c["case_id"]: baseline.classify(c["text"]).level for c in cases}


def collect_predictions(pipeline: CopilotoPipeline, cases: list[dict]) -> list[dict]:
    """Corre el pipeline sobre todos los casos y guarda, por caso, lo mínimo
    necesario para calcular métricas y comparaciones pareadas."""
    predictions = []
    for case in cases:
        s = pipeline.run_case(case["case_id"], case["text"])
        predictions.append(
            {
                "case_id": case["case_id"],
                "level": None if s.abstained else s.level,
                "citation": None if s.abstained else s.citation,
                "abstained": s.abstained,
                "retrieved_ids": [canonical_citation(rc.chunk.chunk_id) for rc in s.retrieved],
                "backend_used": s.backend_used,
                "degraded": s.degraded,
                "escalated": s.escalated,
                "n_modelo": s.n_modelo,
                "n_reglas": s.n_reglas,
            }
        )
    return predictions


# --- Estadísticas sobre listas de casos (para bootstrap) ---------------------

def _stat_sub_triage(pairs: list[tuple[int, int]]) -> float:
    return metrics.sub_triage_rate(pairs) if pairs else float("nan")


def _stat_mean(values: list) -> float:
    return sum(values) / len(values) if values else float("nan")


def _nan_to_none(obj):
    """JSON estricto: NaN no es JSON válido (rompe navegadores y FastAPI)."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def evaluate_predictions(
    cell_name: str,
    predictions: list[dict],
    cases: list[dict],
    c0_levels: dict[str, int],
    audit_records: list[dict] | None = None,
) -> dict:
    """Todas las métricas del paper para una celda, a partir de sus
    predicciones por caso."""
    by_id = {p["case_id"]: p for p in predictions}

    should_abstain_flags: list[bool] = []
    actually_abstained_flags: list[bool] = []
    answered_pairs: list[tuple[int, int]] = []
    sensitivity_pairs: list[tuple[int, int]] = []
    sens_correct_flags: list[bool] = []
    recall_hits: list[bool] = []
    rr_values: list[float] = []
    top1_ids: list[list[str]] = []
    relevant_ids: list[str] = []
    citation_matches: list[bool] = []
    refs: list[int] = []
    cell_levels: list[int | None] = []
    c0_list: list[int | None] = []
    only_cell_hr = only_c0_hr = 0

    for case in cases:
        pred = by_id[case["case_id"]]
        if case["should_abstain"]:
            should_abstain_flags.append(True)
            actually_abstained_flags.append(pred["abstained"])
            continue
        if not is_eligible(case):
            continue

        should_abstain_flags.append(False)
        actually_abstained_flags.append(pred["abstained"])

        ref = case["final_level"]
        refs.append(ref)
        cell_levels.append(pred["level"])
        c0_list.append(c0_levels[case["case_id"]])

        relevant = case["final_citation"]
        recall_hits.append(relevant in pred["retrieved_ids"])
        rr_values.append(metrics.reciprocal_rank(pred["retrieved_ids"], relevant))
        top1_ids.append(pred["retrieved_ids"])
        relevant_ids.append(relevant)

        # La abstención cuenta como fallo en la sensibilidad I-II.
        sug_for_sens = pred["level"] if pred["level"] is not None else 99
        sensitivity_pairs.append((ref, sug_for_sens))
        if ref in HIGH_RISK:
            cell_ok = sug_for_sens in HIGH_RISK
            c0_ok = c0_levels[case["case_id"]] in HIGH_RISK
            sens_correct_flags.append(cell_ok)
            only_cell_hr += int(cell_ok and not c0_ok)
            only_c0_hr += int(c0_ok and not cell_ok)

        if pred["level"] is not None:
            answered_pairs.append((ref, pred["level"]))
            citation_matches.append(canonical_citation(pred["citation"]) == relevant)

    abst = metrics.abstention_metrics(should_abstain_flags, actually_abstained_flags)
    paired_c0 = metrics.paired_sub_triage(refs, cell_levels, c0_list)
    n_eligible = len(refs)
    n_degraded = sum(1 for p in predictions if p["degraded"])

    result = {
        "cell": cell_name,
        "n_eligible_cases": n_eligible,
        "n_abstained_among_eligible": sum(1 for lvl in cell_levels if lvl is None),
        "cobertura": metrics.coverage(cell_levels),
        "S_sub_triage": metrics.sub_triage_rate(answered_pairs),
        "S_sub_triage_ci": metrics.bootstrap_ci(answered_pairs, _stat_sub_triage),
        "S_por_nivel": metrics.sub_triage_by_level(answered_pairs),
        "S_pareado_vs_C0": {
            "n": paired_c0["n"],
            "S_celda": paired_c0["S_a"],
            "S_C0": paired_c0["S_b"],
        },
        "sensibilidad_I_II": metrics.sensitivity_high_risk(sensitivity_pairs),
        "sensibilidad_I_II_ci": metrics.bootstrap_ci(sens_correct_flags, _stat_mean),
        "mcnemar_sens_I_II_vs_C0": {
            "solo_celda_acierta": only_cell_hr,
            "solo_C0_acierta": only_c0_hr,
            "p_value": metrics.mcnemar_exact(only_cell_hr, only_c0_hr),
        },
        "recall_at_1": metrics.recall_at_1(top1_ids, relevant_ids),
        "recall_at_k": metrics.recall_at_k(recall_hits),
        "recall_at_k_ci": metrics.bootstrap_ci(recall_hits, _stat_mean),
        "mrr": metrics.mean_reciprocal_rank(rr_values),
        "fidelidad_citacion": metrics.citation_faithfulness(citation_matches),
        "tasa_abstencion_correcta": abst["tasa_abstencion_correcta"],
        "tasa_abstencion_indebida": abst["tasa_abstencion_indebida"],
        "sobre_triage": metrics.over_triage_rate(answered_pairs),
        "acuerdo_exacto": metrics.exact_agreement(answered_pairs),
        "tasa_modo_degradado": n_degraded / len(predictions) if predictions else float("nan"),
        "n_escalados": sum(1 for p in predictions if p["escalated"]),
    }
    if audit_records is not None:
        chain_ok, _first_bad = verify_chain(audit_records)
        result["completitud_registro"] = completeness(audit_records)["completitud"]
        result["cadena_auditoria_valida"] = chain_ok
    return result


def run_cell(
    cell_name: str,
    corpus_format: str,
    embedding_mode: str,
    cases: list[dict],
    c0_levels: dict[str, int] | None = None,
) -> dict:
    """Celda del estudio de recuperación E1-E4: ruta determinista explícita y
    sin guardarraíl, para aislar el efecto de corpus y recuperación (el efecto
    del guardarraíl se mide aparte, E6 frente a E5)."""
    audit_path = RESULTS_DIR / f"audit_{cell_name}.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    pipeline = CopilotoPipeline(
        corpus_format,
        embedding_mode,
        k=5,
        audit_path=audit_path,
        backend="deterministic",
        use_guardrail=False,
    )
    predictions = collect_predictions(pipeline, cases)
    c0_levels = c0_levels if c0_levels is not None else baseline_levels(cases)
    result = evaluate_predictions(
        cell_name, predictions, cases, c0_levels, pipeline.audit_log.read_all()
    )
    result.update({"corpus_format": corpus_format, "embedding_mode": embedding_mode})
    result["predicciones"] = predictions
    return result


def run_baseline(cases: list[dict]) -> dict:
    baseline = RuleBasedBaseline()
    eligible_pairs: list[tuple[int, int]] = []
    citation_matches: list[bool] = []
    sens_flags: list[bool] = []

    for case in cases:
        if not is_eligible(case):
            continue
        result = baseline.classify(case["text"])
        eligible_pairs.append((case["final_level"], result.level))
        citation_matches.append(result.citation == case["final_citation"])
        if case["final_level"] in HIGH_RISK:
            sens_flags.append(result.level in HIGH_RISK)

    return {
        "cell": "C0",
        "descripcion": "Linea base de reglas (arbol de decision congelado, sin RAG)",
        "n_eligible_cases": len(eligible_pairs),
        "cobertura": 1.0,
        "S_sub_triage": metrics.sub_triage_rate(eligible_pairs),
        "S_sub_triage_ci": metrics.bootstrap_ci(eligible_pairs, _stat_sub_triage),
        "S_por_nivel": metrics.sub_triage_by_level(eligible_pairs),
        "sensibilidad_I_II": metrics.sensitivity_high_risk(eligible_pairs),
        "sensibilidad_I_II_ci": metrics.bootstrap_ci(sens_flags, _stat_mean),
        "fidelidad_citacion": metrics.citation_faithfulness(citation_matches),
        "sobre_triage": metrics.over_triage_rate(eligible_pairs),
        "acuerdo_exacto": metrics.exact_agreement(eligible_pairs),
    }


# --- Reporte ----------------------------------------------------------------

def _f(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/d"
    return f"{value:.{digits}f}"


def _ci(ci: dict | None) -> str:
    if not ci:
        return "n/d"
    return f"{_f(ci['estimate'])} [{_f(ci['ci_low'])}–{_f(ci['ci_high'])}]"


def format_markdown(
    kappa: dict,
    baseline: dict,
    cells: list[dict],
    n_total_cases: int,
    comparisons: dict | None = None,
    models: dict | None = None,
) -> str:
    lines = [
        "# Resultados del diseño experimental: 2x2 (E1-E4), ruta principal (E5-E6) y línea base (C0)",
        "",
        "Generado automáticamente por `experiments/run_experiment.py`. Estas cifras son "
        f"el resultado de ejecutar el pipeline sobre las {n_total_cases} viñetas sintéticas "
        "de `data/cases.json` (dentro del rango de 60-100 casos del protocolo del "
        "documento del proyecto, pero con doble ciego intra-equipo, no evaluadores "
        "clínicos externos); no constituyen una validación clínica externa (ver "
        "`README.md`, sección de limitaciones).",
        "",
        "Convenciones: S y sobre-triage se calculan solo sobre los casos en que la celda "
        "responde; la sensibilidad I-II cuenta la abstención como fallo. Los intervalos son "
        "IC bootstrap del 95 % (2000 réplicas por casos, semilla 42). 'S par. (celda / C0)' "
        "compara S sobre el subconjunto en que ambos responden.",
        "",
        "## Acuerdo intra-equipo (gold standard)",
        "",
        f"- Pares evaluados: {kappa['n_pares']}",
        f"- Kappa ponderado cuadrático (κ_w): {kappa['kappa_ponderado_cuadratico']:.3f}",
        "",
        "## Línea base C0 (árbol de reglas, sin RAG)",
        "",
        f"- Casos elegibles: {baseline['n_eligible_cases']}",
        f"- S (sub-triage ponderado): {_ci(baseline.get('S_sub_triage_ci'))}",
        f"- Sensibilidad niveles I-II: {_ci(baseline.get('sensibilidad_I_II_ci'))}",
        f"- Sobre-triage: {_f(baseline.get('sobre_triage'))}",
        f"- Acuerdo exacto: {_f(baseline.get('acuerdo_exacto'))}",
        f"- Fidelidad de citación: {_f(baseline['fidelidad_citacion'])}",
        f"- S por nivel de referencia: {baseline['S_por_nivel']}",
        "",
        "## Celdas del copiloto",
        "",
        "| Celda | Corpus | Recuperación | N eleg. | Cobertura | S [IC95] | S par. (celda / C0, n) | "
        "Sensib. I-II [IC95] | McNemar vs C0 (p) | Recall@1 | Recall@k [IC95] | MRR | "
        "Fidelidad | Abst. correcta | Abst. indebida | Sobre-triage | Completitud | Degradado |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        if cell.get("no_evaluada"):
            lines.append(
                f"| {cell['cell']} | {cell.get('corpus_format', '')} | {cell.get('embedding_mode', '')} | "
                f"{cell['no_evaluada']} |" + " |" * 14
            )
            continue
        par = cell["S_pareado_vs_C0"]
        mc = cell["mcnemar_sens_I_II_vs_C0"]
        lines.append(
            "| {cell} | {cf} | {em} | {n} | {cov} | {s} | {sa} / {sb} (n={pn}) | {sens} | "
            "{mcp} ({a}:{b}) | {r1} | {rk} | {mrr} | {fid} | {ac} | {ai} | {ot} | {comp} | {deg} |".format(
                cell=cell["cell"],
                cf=cell["corpus_format"],
                em=cell["embedding_mode"],
                n=cell["n_eligible_cases"],
                cov=_f(cell["cobertura"], 2),
                s=_ci(cell["S_sub_triage_ci"]),
                sa=_f(par["S_celda"]),
                sb=_f(par["S_C0"]),
                pn=par["n"],
                sens=_ci(cell["sensibilidad_I_II_ci"]),
                mcp=_f(mc["p_value"]),
                a=mc["solo_celda_acierta"],
                b=mc["solo_C0_acierta"],
                r1=_f(cell["recall_at_1"]),
                rk=_ci(cell["recall_at_k_ci"]),
                mrr=_f(cell["mrr"]),
                fid=_f(cell["fidelidad_citacion"]),
                ac=_f(cell["tasa_abstencion_correcta"]),
                ai=_f(cell["tasa_abstencion_indebida"]),
                ot=_f(cell["sobre_triage"]),
                comp=_f(cell.get("completitud_registro"), 2),
                deg=_f(cell["tasa_modo_degradado"], 2),
            )
        )
    lines.append("")
    lines.append("McNemar vs C0 (H1): valor p exacto sobre los casos de referencia I-II; entre paréntesis, "
                 "casos que solo la celda clasifica en I-II : casos que solo C0 clasifica en I-II.")
    lines.append("E1-E4: ruta determinista explícita, sin guardarraíl (estudio de recuperación, sin red). "
                 "E5-E6: ruta principal (BM25 + Gemini fusionados por RRF, y LLM que elige la cita), "
                 "sin y con guardarraíl; sus métricas se calculan solo sobre los casos que respondió "
                 "Gemini ('Degradado' = fracción que cayó al respaldo).")

    evaluated_gemini = [c for c in cells if c.get("repeticiones")]
    if evaluated_gemini:
        lines += ["", "### Repeticiones de E5 y E6 (media [mín–máx])", ""]
        reps0 = evaluated_gemini[0]["repeticiones"]
        lines.append(f"{reps0['n']} repeticiones; modos de caché: {', '.join(reps0['modos_cache'])}.")
        lines.append("")
        lines.append("| Celda | " + " | ".join(REPETITION_METRICS) + " |")
        lines.append("|---|" + "---|" * len(REPETITION_METRICS))
        for cell in evaluated_gemini:
            reps = cell["repeticiones"]
            row = []
            for key in REPETITION_METRICS:
                stat = reps.get(key)
                row.append(f"{_f(stat['media'])} [{_f(stat['min'])}–{_f(stat['max'])}]" if stat else "n/d")
            lines.append(f"| {cell['cell']} | " + " | ".join(row) + " |")

    h5 = (comparisons or {}).get("H5_E6_vs_E5")
    lines += ["", "### H5: efecto del guardarraíl (E6 frente a E5, pareado)", ""]
    if h5:
        mc = h5["mcnemar_sens_I_II"]
        lines += [
            f"- Casos elegibles pareados: {h5['n_pareados']} (ambas responden: {h5['n_ambos_responden']})",
            f"- S pareado: E6 {_f(h5['S_a'])} / E5 {_f(h5['S_b'])}",
            f"- Sobre-triage pareado: E6 {_f(h5['sobre_triage_a'])} / E5 {_f(h5['sobre_triage_b'])}",
            f"- Sensibilidad I-II, McNemar exacto: p = {_f(mc['p_value'])} "
            f"(solo E6 acierta: {mc['solo_a_acierta']}; solo E5 acierta: {mc['solo_b_acierta']})",
        ]
    else:
        lines.append("No disponible: E5 y E6 no se evaluaron (se necesita `GEMINI_API_KEY` o la caché de respuestas).")

    if models:
        lines += [
            "",
            "### Versiones",
            "",
            f"- Modelo de generación: `{models['generation_model']}`; embeddings: `{models['embedding_model']}`",
            f"- Prompt: `{models['prompt_version']}`; reglas de alarma: `{models['alarm_rules_version']}`",
        ]

    lines.append("")
    lines.append("### S desglosado por nivel de referencia, por celda")
    lines.append("")
    for cell in cells:
        if not cell.get("no_evaluada"):
            lines.append(f"- **{cell['cell']}**: {cell['S_por_nivel']}")
    return "\n".join(lines) + "\n"


# --- Celdas de la ruta principal (Gemini): E5 y E6 ---------------------------

GEMINI_CELLS = [
    # (celda, use_guardrail)
    ("E5", False),
    ("E6", True),
]
N_REPETITIONS = 3
REPETITION_METRICS = [
    "cobertura",
    "S_sub_triage",
    "sensibilidad_I_II",
    "recall_at_1",
    "recall_at_k",
    "mrr",
    "sobre_triage",
    "tasa_modo_degradado",
]
GEMINI_RETRIEVAL_LABEL = "rrf(bm25+gemini) + LLM"


def gemini_resources() -> dict:
    """Qué hay disponible para evaluar la ruta principal sin inventar
    resultados: una API key y/o una caché de respuestas del LLM."""
    cache = llm_client.DEFAULT_LLM_CACHE_PATH
    return {
        "api_key": bool(os.environ.get("GEMINI_API_KEY")),
        "cache_llm": cache.exists() and cache.stat().st_size > 2,
    }


@contextmanager
def _cache_mode(mode: str):
    previous = os.environ.get("GEMINI_CACHE_MODE")
    os.environ["GEMINI_CACHE_MODE"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GEMINI_CACHE_MODE", None)
        else:
            os.environ["GEMINI_CACHE_MODE"] = previous


def _not_evaluated(cell_name: str, reason: str) -> dict:
    return {
        "cell": cell_name,
        "corpus_format": "reformatted",
        "embedding_mode": GEMINI_RETRIEVAL_LABEL,
        "no_evaluada": reason,
    }


def run_gemini_cell(
    cell_name: str, use_guardrail: bool, cases: list[dict], c0_levels: dict[str, int], audit_path: Path
) -> dict:
    """Una repetición de E5/E6 con la ruta principal. Las métricas se
    calculan SOLO sobre los casos que respondió Gemini; los que cayeron al
    respaldo se cuentan en ``tasa_modo_degradado`` y no se mezclan."""
    if audit_path.exists():
        audit_path.unlink()
    pipeline = CopilotoPipeline(
        "reformatted", "clinical_es", k=5, audit_path=audit_path, backend="gemini", use_guardrail=use_guardrail
    )
    predictions = collect_predictions(pipeline, cases)
    served = [p for p in predictions if not p["degraded"]]
    if not served:
        return _not_evaluated(cell_name, "no evaluada (Gemini no respondió ningún caso)")
    served_ids = {p["case_id"] for p in served}
    served_cases = [c for c in cases if c["case_id"] in served_ids]
    result = evaluate_predictions(
        cell_name, served, served_cases, c0_levels, pipeline.audit_log.read_all()
    )
    result.update(
        {
            "corpus_format": "reformatted",
            "embedding_mode": GEMINI_RETRIEVAL_LABEL + (" + guardarraíl" if use_guardrail else ""),
            "use_guardrail": use_guardrail,
            "tasa_modo_degradado": 1 - len(served) / len(predictions),
            "n_casos_respaldo": len(predictions) - len(served),
            "predicciones": predictions,
        }
    )
    return result


def compare_paired(preds_a: list[dict], preds_b: list[dict], cases: list[dict]) -> dict:
    """Comparación pareada A vs. B sobre los casos elegibles que ambas
    evaluaron con la ruta principal: S y sobre-triage donde ambas responden,
    y McNemar exacto de la sensibilidad I-II (abstención = fallo)."""
    by_a = {p["case_id"]: p for p in preds_a if not p["degraded"]}
    by_b = {p["case_id"]: p for p in preds_b if not p["degraded"]}
    refs: list[int] = []
    la: list[int | None] = []
    lb: list[int | None] = []
    only_a = only_b = 0
    for case in cases:
        cid = case["case_id"]
        if not is_eligible(case) or cid not in by_a or cid not in by_b:
            continue
        a, b = by_a[cid]["level"], by_b[cid]["level"]
        refs.append(case["final_level"])
        la.append(a)
        lb.append(b)
        if case["final_level"] in HIGH_RISK:
            a_ok, b_ok = a in HIGH_RISK, b in HIGH_RISK
            only_a += int(a_ok and not b_ok)
            only_b += int(b_ok and not a_ok)
    both = [(r, x, y) for r, x, y in zip(refs, la, lb) if x is not None and y is not None]
    paired = metrics.paired_sub_triage(refs, la, lb)
    return {
        "n_pareados": len(refs),
        "n_ambos_responden": paired["n"],
        "S_a": paired["S_a"],
        "S_b": paired["S_b"],
        "sobre_triage_a": metrics.over_triage_rate([(r, x) for r, x, _ in both]),
        "sobre_triage_b": metrics.over_triage_rate([(r, y) for r, _, y in both]),
        "mcnemar_sens_I_II": {
            "solo_a_acierta": only_a,
            "solo_b_acierta": only_b,
            "p_value": metrics.mcnemar_exact(only_a, only_b),
        },
    }


def _repetition_summary(results: list[dict], modes: list[str]) -> dict:
    summary: dict = {"n": len(results), "modos_cache": modes}
    for key in REPETITION_METRICS:
        values = [r[key] for r in results if not r.get("no_evaluada") and r[key] == r[key]]
        if values:
            summary[key] = {"media": sum(values) / len(values), "min": min(values), "max": max(values)}
    return summary


def run_gemini_cells(
    cases: list[dict],
    c0_levels: dict[str, int],
    first_mode: str | None = None,
    repetitions: int = N_REPETITIONS,
    cells_to_run: tuple[str, ...] = ("E5", "E6"),
) -> tuple[list[dict], dict | None]:
    """E5 y E6 con ``repetitions`` repeticiones: la primera en ``readwrite``
    si hay API key (si no, ``readonly``), las demás en ``readonly``. Si no
    hay ni clave ni caché, NO se calculan con el respaldo: quedan como "no
    evaluadas"."""
    selected = [(name, guard) for name, guard in GEMINI_CELLS if name in cells_to_run]
    resources = gemini_resources()
    if not resources["api_key"] and not resources["cache_llm"]:
        return [_not_evaluated(name, "no evaluada (sin API ni caché)") for name, _ in selected], None
    if first_mode == "readonly" and not resources["cache_llm"]:
        return [_not_evaluated(name, "no evaluada (sin caché; modo solo lectura)") for name, _ in selected], None

    first = first_mode or ("readwrite" if resources["api_key"] else "readonly")
    modes = [first] + ["readonly"] * (repetitions - 1)
    reps: dict[str, list[dict]] = {name: [] for name, _ in selected}
    with tempfile.TemporaryDirectory() as tmp:
        for r, mode in enumerate(modes, start=1):
            with _cache_mode(mode):
                for name, guard in selected:
                    # Se conserva el registro de auditoría de la primera
                    # repetición; las demás se escriben en un directorio temporal.
                    audit_path = RESULTS_DIR / f"audit_{name}.jsonl" if r == 1 else Path(tmp) / f"{name}_{r}.jsonl"
                    reps[name].append(run_gemini_cell(name, guard, cases, c0_levels, audit_path))

    cells = []
    for name, _ in selected:
        main_result = reps[name][0]
        if not main_result.get("no_evaluada"):
            main_result["repeticiones"] = _repetition_summary(reps[name], modes)
        cells.append(main_result)

    by_name = {c["cell"]: c for c in cells}
    comparisons = None
    e5, e6 = by_name.get("E5"), by_name.get("E6")
    if e5 and e6 and not e5.get("no_evaluada") and not e6.get("no_evaluada"):
        comparisons = {"H5_E6_vs_E5": compare_paired(e6["predicciones"], e5["predicciones"], cases)}
    return cells, comparisons


def run_all(cases: list[dict], include_gemini: bool = True, gemini_first_mode: str | None = None) -> dict:
    """Todas las celdas. ``gemini_first_mode="readonly"`` evalúa E5/E6 solo
    desde la caché (lo usan la GUI y la API web para no gastar cuota)."""
    kappa = intra_team_kappa(cases)
    baseline = run_baseline(cases)
    c0 = baseline_levels(cases)
    cells = [run_cell(name, fmt, mode, cases, c0_levels=c0) for name, fmt, mode in CELLS]
    comparisons = None
    if include_gemini:
        gemini_cells, comparisons = run_gemini_cells(cases, c0, first_mode=gemini_first_mode)
        cells += gemini_cells
    return {
        "kappa_intra_equipo": kappa,
        "baseline_C0": baseline,
        "celdas": cells,
        "comparaciones": comparisons,
        "modelos": models_info(),
    }


def models_info() -> dict:
    return {
        "generation_model": llm_client.GENERATION_MODEL,
        "embedding_model": llm_client.EMBEDDING_MODEL,
        "prompt_version": PROMPT_VERSION,
        "alarm_rules_version": AlarmDetector().version,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    cases = load_cases()
    output = run_all(cases)

    (RESULTS_DIR / "raw_results.json").write_text(
        json.dumps(_nan_to_none(output), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_md = format_markdown(
        output["kappa_intra_equipo"],
        output["baseline_C0"],
        output["celdas"],
        n_total_cases=len(cases),
        comparisons=output["comparaciones"],
        models=output["modelos"],
    )
    (RESULTS_DIR / "summary.md").write_text(summary_md, encoding="utf-8")
    print(summary_md)


if __name__ == "__main__":
    main()
