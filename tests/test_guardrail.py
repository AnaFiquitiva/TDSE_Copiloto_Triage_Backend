"""Pruebas de propiedades del guardarraíl: se recorren TODAS las
combinaciones de nivel del modelo (1-5 o abstención) y nivel de reglas (1-5
o ninguna alarma) y se verifican los invariantes del diseño."""
import itertools
import unittest

from src.extraction import Alarm, ExtractionResult
from src.generator import Suggestion
from src.guardrail import REASON_RAISED_BY_RULE, apply_guardrail, finalize_without_guardrail
from src.ingest import LEVEL_BY_CITATION

LEVELS_OR_NONE = [1, 2, 3, 4, 5, None]
# Una cita real por nivel, para que las sugerencias y alarmas sean coherentes.
CITATION_BY_LEVEL = {1: "RES5596-N1", 2: "RES5596-N2", 3: "RES5596-N3", 4: "RES5596-N4", 5: "RES5596-N5"}


def _suggestion(n_m):
    if n_m is None:
        return Suggestion(None, None, 0.0, True, "evidencia_insuficiente")
    return Suggestion(n_m, CITATION_BY_LEVEL[n_m], 0.8, False, "ok")


def _extraction(n_r):
    if n_r is None:
        return ExtractionResult("texto", [], None)
    alarms = [Alarm("ALR-X", n_r, CITATION_BY_LEVEL[n_r], "patron")]
    if n_r < 5:  # una alarma adicional menos urgente no debe cambiar n_reglas
        alarms.append(Alarm("ALR-Y", 5, CITATION_BY_LEVEL[5], "otro"))
    return ExtractionResult("texto", alarms, n_r)


class TestGuardrailInvariants(unittest.TestCase):
    def _all_results(self):
        for n_m, n_r in itertools.product(LEVELS_OR_NONE, LEVELS_OR_NONE):
            yield n_m, n_r, apply_guardrail(_suggestion(n_m), _extraction(n_r))

    def test_final_level_never_less_urgent_than_model_or_rules(self):
        for n_m, n_r, out in self._all_results():
            with self.subTest(n_modelo=n_m, n_reglas=n_r):
                if out.n_final is not None and n_m is not None:
                    self.assertLessEqual(out.n_final, n_m)
                if out.n_final is not None and n_r is not None:
                    self.assertLessEqual(out.n_final, n_r)

    def test_final_is_min_when_model_answers(self):
        for n_m, n_r, out in self._all_results():
            if n_m is None:
                continue
            with self.subTest(n_modelo=n_m, n_reglas=n_r):
                expected = n_m if n_r is None else min(n_m, n_r)
                self.assertEqual(out.n_final, expected)
                self.assertEqual(out.level, expected)

    def test_never_a_level_without_a_citation(self):
        for n_m, n_r, out in self._all_results():
            with self.subTest(n_modelo=n_m, n_reglas=n_r):
                if out.level is not None or out.n_final is not None:
                    self.assertIsNotNone(out.citation)
                    self.assertEqual(LEVEL_BY_CITATION[out.citation], out.level)

    def test_every_abstention_is_escalated(self):
        for n_m, n_r, out in self._all_results():
            with self.subTest(n_modelo=n_m, n_reglas=n_r):
                if out.abstained:
                    self.assertTrue(out.escalated)

    def test_abstention_with_alarm_keeps_rule_as_reference_not_as_suggestion(self):
        out = apply_guardrail(_suggestion(None), _extraction(1))
        self.assertTrue(out.abstained)
        self.assertTrue(out.escalated)
        self.assertIsNone(out.level)
        self.assertIsNone(out.n_final)
        self.assertEqual(out.n_reglas, 1)
        self.assertEqual(out.alarms[0]["citation"], "RES5596-N1")

    def test_rule_wins_keeps_model_citation(self):
        out = apply_guardrail(_suggestion(4), _extraction(2))
        self.assertEqual(out.level, 2)
        self.assertEqual(out.citation, "RES5596-N2")
        self.assertEqual(out.model_citation, "RES5596-N4")
        self.assertEqual(out.n_modelo, 4)
        self.assertEqual(out.reason, REASON_RAISED_BY_RULE)

    def test_rule_less_urgent_than_model_does_not_change_anything(self):
        out = apply_guardrail(_suggestion(1), _extraction(2))
        self.assertEqual(out.level, 1)
        self.assertEqual(out.citation, "RES5596-N1")
        self.assertEqual(out.reason, "ok")


class TestWithoutGuardrail(unittest.TestCase):
    def test_abstention_is_still_escalated(self):
        out = finalize_without_guardrail(_suggestion(None))
        self.assertTrue(out.escalated)
        self.assertIsNone(out.n_final)

    def test_answer_passes_through(self):
        out = finalize_without_guardrail(_suggestion(3))
        self.assertEqual((out.n_modelo, out.n_final, out.level), (3, 3, 3))
        self.assertFalse(out.escalated)
        self.assertIsNone(out.n_reglas)


if __name__ == "__main__":
    unittest.main()
