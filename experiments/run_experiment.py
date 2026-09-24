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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from src import metrics  # noqa: E402
from src.audit import completeness, verify_chain  # noqa: E402
from src.baseline import RuleBasedBaseline  # noqa: E402
from src.ingest import canonical_citation  # noqa: E402
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
                "escalated": getattr(s, "escalated", s.abstained),
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
    """Celda del estudio de recuperación E1-E4 (ruta determinista explícita)."""
    audit_path = RESULTS_DIR / f"audit_{cell_name}.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    pipeline = CopilotoPipeline(
        corpus_format, embedding_mode, k=5, audit_path=audit_path, backend="deterministic"
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


def format_markdown(kappa: dict, baseline: dict, cells: list[dict], n_total_cases: int) -> str:
    lines = [
        "# Resultados del diseño experimental 2x2 + línea base (C0)",
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
    lines.append("McNemar vs C0: valor p exacto sobre los casos de referencia I-II; entre paréntesis, "
                 "casos que solo la celda clasifica en I-II : casos que solo C0 clasifica en I-II.")
    lines.append("")
    lines.append("### S desglosado por nivel de referencia, por celda")
    lines.append("")
    for cell in cells:
        if not cell.get("no_evaluada"):
            lines.append(f"- **{cell['cell']}**: {cell['S_por_nivel']}")
    return "\n".join(lines) + "\n"


def run_all(cases: list[dict]) -> dict:
    kappa = intra_team_kappa(cases)
    baseline = run_baseline(cases)
    c0 = baseline_levels(cases)
    cells = [run_cell(name, fmt, mode, cases, c0_levels=c0) for name, fmt, mode in CELLS]
    return {"kappa_intra_equipo": kappa, "baseline_C0": baseline, "celdas": cells}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    cases = load_cases()
    output = run_all(cases)

    (RESULTS_DIR / "raw_results.json").write_text(
        json.dumps(_nan_to_none(output), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_md = format_markdown(
        output["kappa_intra_equipo"], output["baseline_C0"], output["celdas"], n_total_cases=len(cases)
    )
    (RESULTS_DIR / "summary.md").write_text(summary_md, encoding="utf-8")
    print(summary_md)


if __name__ == "__main__":
    main()
