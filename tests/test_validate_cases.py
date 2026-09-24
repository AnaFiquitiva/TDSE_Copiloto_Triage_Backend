"""Pruebas de la validación del gold standard (sin modificar ni generar
niveles: solo se verifica que el validador detecte inconsistencias)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import run_experiment
from experiments.validate_cases import DEFAULT_CASES_PATH, load_case_file, validate_cases

ROOT = Path(__file__).resolve().parent.parent


def _case(**overrides):
    base = {
        "case_id": "T1",
        "text": "relato",
        "evaluator1": {"level": 2, "citation": "RES5596-N2"},
        "evaluator2": {"level": 2, "citation": "RES5596-N2"},
        "resolution_method": "acuerdo_inicial",
        "final_level": 2,
        "final_citation": "RES5596-N2",
        "ambiguous": False,
        "should_abstain": False,
    }
    base.update(overrides)
    return base


class TestGoldStandard(unittest.TestCase):
    def test_current_gold_standard_is_valid(self):
        report = validate_cases(load_case_file(DEFAULT_CASES_PATH))
        self.assertTrue(report["valido"], report["errores"])
        self.assertGreaterEqual(report["por_nivel"][1], 15)
        self.assertGreaterEqual(report["por_nivel"][2], 15)
        self.assertEqual(report["n_ambiguos"], 2)
        self.assertEqual(report["n_abstencion"], 6)

    def test_template_is_structurally_valid_and_empty(self):
        cases = load_case_file(ROOT / "data" / "cases_nuevos.template.json")
        self.assertEqual(cases, [])
        self.assertTrue(validate_cases(cases, min_per_high_level=0)["valido"])


class TestValidatorDetectsErrors(unittest.TestCase):
    def _errors(self, cases, min_per=0):
        return validate_cases(cases, min_per_high_level=min_per)["errores"]

    def test_valid_case_has_no_errors(self):
        self.assertEqual(self._errors([_case()]), [])

    def test_missing_field(self):
        c = _case()
        del c["final_citation"]
        self.assertTrue(any("faltan campos" in e for e in self._errors([c])))

    def test_level_out_of_range(self):
        self.assertTrue(self._errors([_case(final_level=6)]))

    def test_unknown_citation(self):
        self.assertTrue(any("inexistente" in e for e in self._errors([_case(final_citation="INVENTADA")])))

    def test_final_citation_level_mismatch(self):
        self.assertTrue(any("es de nivel" in e for e in self._errors([_case(final_citation="RES5596-N4")])))

    def test_initial_agreement_with_different_levels(self):
        c = _case(evaluator2={"level": 3, "citation": "RES5596-N3"})
        self.assertTrue(any("niveles distintos" in e for e in self._errors([c])))

    def test_tiebreak_requires_third_evaluator_matching_final(self):
        c = _case(resolution_method="desempate_tercer_integrante", evaluator2={"level": 3, "citation": "RES5596-N3"})
        self.assertTrue(any("evaluator3" in e for e in self._errors([c])))
        c["evaluator3"] = {"level": 3, "citation": "RES5596-N3"}
        self.assertTrue(any("tercer integrante" in e for e in self._errors([c])))

    def test_abstention_case_must_not_have_level(self):
        c = _case(should_abstain=True, resolution_method="no_aplica_caso_de_abstencion")
        self.assertTrue(any("no debe tener nivel" in e for e in self._errors([c])))

    def test_minimum_high_risk_cases(self):
        errors = self._errors([_case()], min_per=15)
        self.assertTrue(any("Nivel 1" in e for e in errors))
        self.assertTrue(any("Nivel 2" in e for e in errors))

    def test_duplicate_ids(self):
        self.assertTrue(any("duplicado" in e for e in self._errors([_case(), copy.deepcopy(_case())])))


class TestRunNewCases(unittest.TestCase):
    def test_new_cases_file_without_gemini_marks_e6_not_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases_prueba.json"
            path.write_text(json.dumps({"cases": [_case()]}), encoding="utf-8")
            with patch.object(run_experiment, "RESULTS_DIR", Path(tmp)), patch.object(
                run_experiment, "gemini_resources", return_value={"api_key": False, "cache_llm": False}
            ):
                md = run_experiment.run_new_cases(path)
            self.assertIn("casos nuevos", md)
            self.assertIn("E6", md)
            self.assertIn("no evaluada", md)
            self.assertTrue((Path(tmp) / "summary_cases_prueba.md").exists())

    def test_invalid_new_cases_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "malo.json"
            path.write_text(json.dumps({"cases": [_case(final_level=9)]}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                run_experiment.run_new_cases(path)


if __name__ == "__main__":
    unittest.main()
