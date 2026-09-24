"""Valida un archivo de casos (``data/cases.json`` o ``data/cases_nuevos.json``)
contra el esquema mínimo de documentación por caso del paper.

Comprueba:

- campos obligatorios y tipos;
- niveles del 1 al 5 (o null solo en casos ambiguos o de abstención);
- que toda cita exista en ``LEVEL_BY_CITATION`` y que la cita final
  corresponda al nivel final;
- consistencia entre ``resolution_method`` y el acuerdo de los evaluadores;
- que haya al menos ``--min-por-nivel`` casos (15 por defecto) en cada uno de
  los niveles I y II;
- ``case_id`` únicos; y reporta el número de ambiguos y de abstenciones.

NO corrige ni genera niveles: solo informa. El etiquetado es del equipo.

Uso:
    python -m experiments.validate_cases
    python -m experiments.validate_cases data/cases_nuevos.json --min-por-nivel 0
"""
from __future__ import annotations

import argparse
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

from src.ingest import LEVEL_BY_CITATION  # noqa: E402

DEFAULT_CASES_PATH = ROOT / "data" / "cases.json"
REQUIRED_FIELDS = (
    "case_id",
    "text",
    "evaluator1",
    "evaluator2",
    "resolution_method",
    "final_level",
    "final_citation",
    "ambiguous",
    "should_abstain",
)
RESOLUTION_METHODS = (
    "acuerdo_inicial",
    "consenso_tras_discusion",
    "desempate_tercer_integrante",
    "ambiguo",
    "no_aplica_caso_de_abstencion",
)
LEVELS = (1, 2, 3, 4, 5)


def _check_evaluator(case_id: str, name: str, ev, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(ev, dict) or "level" not in ev or "citation" not in ev:
        errors.append(f"{case_id}: {name} debe tener 'level' y 'citation'")
        return
    if ev["level"] is not None and ev["level"] not in LEVELS:
        errors.append(f"{case_id}: {name}.level fuera de 1-5 ({ev['level']})")
    if ev["citation"] is not None and ev["citation"] not in LEVEL_BY_CITATION:
        errors.append(f"{case_id}: {name}.citation inexistente ({ev['citation']})")
    elif ev["citation"] is not None and ev["level"] is not None and LEVEL_BY_CITATION[ev["citation"]] != ev["level"]:
        # Un evaluador puede citar el criterio correcto y aun así discrepar en
        # el nivel (es justo lo que se resuelve después): solo se advierte.
        warnings.append(
            f"{case_id}: {name} asigna nivel {ev['level']} con una cita de nivel "
            f"{LEVEL_BY_CITATION[ev['citation']]} ({ev['citation']})"
        )


def validate_cases(cases: list[dict], min_per_high_level: int = 15) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    ids = Counter(c.get("case_id") for c in cases)
    for case_id, n in ids.items():
        if n > 1:
            errors.append(f"case_id duplicado: {case_id} ({n} veces)")

    for c in cases:
        cid = c.get("case_id", "<sin case_id>")
        missing = [f for f in REQUIRED_FIELDS if f not in c]
        if missing:
            errors.append(f"{cid}: faltan campos {missing}")
            continue
        if not isinstance(c["text"], str) or not c["text"].strip():
            errors.append(f"{cid}: 'text' vacío")
        for flag in ("ambiguous", "should_abstain"):
            if not isinstance(c[flag], bool):
                errors.append(f"{cid}: '{flag}' debe ser booleano")

        _check_evaluator(cid, "evaluator1", c["evaluator1"], errors, warnings)
        _check_evaluator(cid, "evaluator2", c["evaluator2"], errors, warnings)
        if "evaluator3" in c:
            _check_evaluator(cid, "evaluator3", c["evaluator3"], errors, warnings)

        final_level, final_citation = c["final_level"], c["final_citation"]
        no_level_expected = c["ambiguous"] or c["should_abstain"]
        if no_level_expected:
            if final_level is not None or final_citation is not None:
                errors.append(f"{cid}: un caso ambiguo o de abstención no debe tener nivel/cita final")
        else:
            if final_level not in LEVELS:
                errors.append(f"{cid}: final_level debe estar entre 1 y 5 (es {final_level})")
            if final_citation not in LEVEL_BY_CITATION:
                errors.append(f"{cid}: final_citation inexistente ({final_citation})")
            elif final_level in LEVELS and LEVEL_BY_CITATION[final_citation] != final_level:
                errors.append(
                    f"{cid}: final_citation {final_citation} es de nivel "
                    f"{LEVEL_BY_CITATION[final_citation]}, no {final_level}"
                )
        if c["ambiguous"] and c["should_abstain"]:
            errors.append(f"{cid}: no puede ser ambiguo y de abstención a la vez")

        method = c["resolution_method"]
        e1, e2 = c["evaluator1"], c["evaluator2"]
        if not isinstance(e1, dict) or not isinstance(e2, dict):
            continue
        agree = e1.get("level") == e2.get("level") and e1.get("citation") == e2.get("citation")
        if method not in RESOLUTION_METHODS:
            errors.append(f"{cid}: resolution_method desconocido ({method})")
        elif method == "acuerdo_inicial":
            # El acuerdo se mide sobre el NIVEL (es lo que entra en el kappa);
            # una cita distinta con el mismo nivel solo se advierte.
            if e1.get("level") != e2.get("level"):
                errors.append(f"{cid}: 'acuerdo_inicial' pero los evaluadores asignan niveles distintos")
            elif e1.get("citation") != e2.get("citation"):
                warnings.append(
                    f"{cid}: 'acuerdo_inicial' en nivel, pero con citas distintas "
                    f"({e1.get('citation')} / {e2.get('citation')}); final: {final_citation}"
                )
            if final_level != e1.get("level"):
                errors.append(f"{cid}: 'acuerdo_inicial' pero final_level difiere del nivel acordado")
        elif method == "consenso_tras_discusion":
            if agree:
                errors.append(f"{cid}: 'consenso_tras_discusion' pero los evaluadores ya coincidían")
            if not c.get("nota_resolucion"):
                warnings.append(f"{cid}: consenso sin 'nota_resolucion' que documente la discusión")
        elif method == "desempate_tercer_integrante":
            if agree:
                errors.append(f"{cid}: desempate pero los evaluadores ya coincidían")
            e3 = c.get("evaluator3")
            if not isinstance(e3, dict):
                errors.append(f"{cid}: desempate sin 'evaluator3'")
            elif e3.get("level") != final_level:
                errors.append(f"{cid}: el nivel final no es el del tercer integrante")
        elif method == "ambiguo":
            if not c["ambiguous"]:
                errors.append(f"{cid}: resolution_method 'ambiguo' pero ambiguous=false")
        elif method == "no_aplica_caso_de_abstencion":
            if not c["should_abstain"]:
                errors.append(f"{cid}: caso de abstención pero should_abstain=false")
        if c["ambiguous"] and method != "ambiguo":
            errors.append(f"{cid}: ambiguous=true exige resolution_method 'ambiguo'")
        if c["should_abstain"] and method != "no_aplica_caso_de_abstencion":
            errors.append(f"{cid}: should_abstain=true exige resolution_method 'no_aplica_caso_de_abstencion'")

    by_level = Counter(c.get("final_level") for c in cases if c.get("final_level") in LEVELS)
    for level in (1, 2):
        if by_level[level] < min_per_high_level:
            errors.append(
                f"Nivel {level}: {by_level[level]} casos, se exigen al menos {min_per_high_level}"
            )

    return {
        "n_casos": len(cases),
        "por_nivel": {lvl: by_level[lvl] for lvl in LEVELS},
        "n_ambiguos": sum(1 for c in cases if c.get("ambiguous") is True),
        "n_abstencion": sum(1 for c in cases if c.get("should_abstain") is True),
        "metodos_resolucion": dict(Counter(c.get("resolution_method") for c in cases)),
        "errores": errors,
        "advertencias": warnings,
        "valido": not errors,
    }


def load_case_file(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cases"] if isinstance(data, dict) else data


def format_report(path: Path, report: dict) -> str:
    lines = [
        f"Validación de {path}",
        f"- Casos: {report['n_casos']}  | por nivel final: {report['por_nivel']}",
        f"- Ambiguos: {report['n_ambiguos']}  | de abstención: {report['n_abstencion']}",
        f"- Métodos de resolución: {report['metodos_resolucion']}",
        f"- Resultado: {'VÁLIDO' if report['valido'] else 'CON ERRORES'}",
    ]
    for title, items in (("Errores", report["errores"]), ("Advertencias", report["advertencias"])):
        if items:
            lines.append(f"{title} ({len(items)}):")
            lines += [f"  - {item}" for item in items]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--min-por-nivel", type=int, default=15,
                        help="Mínimo de casos exigido en los niveles I y II (0 para desactivar).")
    args = parser.parse_args()
    path = Path(args.path)
    report = validate_cases(load_case_file(path), min_per_high_level=args.min_por_nivel)
    print(format_report(path, report))
    return 0 if report["valido"] else 1


if __name__ == "__main__":
    sys.exit(main())
