"""Pruebas del componente 1 (detector de alarmas) y de la separación
respecto a la línea base C0."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.extraction import DEFAULT_ALARM_RULES_PATH, AlarmDetector, normalize
from src.ingest import LEVEL_BY_CITATION

ROOT = Path(__file__).resolve().parent.parent

# SHA-256 de los artefactos congelados de C0 en el commit base 6195112
# (saltos de línea normalizados a LF para que la prueba no dependa de
# core.autocrlf). Si esta prueba falla, alguien modificó la línea base.
C0_FROZEN_HASHES = {
    "data/keyword_rules.json": "8ba60952859bfb078ba12484a2930c78d7328b33107295372cd0973d63b26b47",
    "src/baseline.py": "13f9930566f30f8fc53ae71a5edf5ad9c4ad0c6e3f54b179954f807e1559f410",
}


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class TestC0IsFrozen(unittest.TestCase):
    def test_keyword_rules_and_baseline_are_unchanged(self):
        for rel_path, expected in C0_FROZEN_HASHES.items():
            with self.subTest(archivo=rel_path):
                self.assertEqual(_sha256_lf(ROOT / rel_path), expected)

    def test_alarm_module_does_not_use_c0_rules(self):
        source = (ROOT / "src" / "extraction.py").read_text(encoding="utf-8")
        self.assertNotIn("keyword_rules.json", source)
        self.assertNotIn("from .baseline", source)
        self.assertNotIn("import baseline", source)


class TestAlarmRulesFile(unittest.TestCase):
    def test_every_citation_exists_with_the_same_level(self):
        config = json.loads(DEFAULT_ALARM_RULES_PATH.read_text(encoding="utf-8"))
        for rule in config["rules"]:
            with self.subTest(regla=rule["id"]):
                self.assertIn(rule["citation"], LEVEL_BY_CITATION)
                self.assertEqual(LEVEL_BY_CITATION[rule["citation"]], rule["level"])

    def test_rule_ids_are_unique_and_have_origin(self):
        config = json.loads(DEFAULT_ALARM_RULES_PATH.read_text(encoding="utf-8"))
        ids = [r["id"] for r in config["rules"]]
        self.assertEqual(len(ids), len(set(ids)))
        for rule in config["rules"]:
            self.assertIn(rule["origen"], {"linea_base_C0", "criterio_normativo", "analisis_error_avance"})

    def test_invalid_citation_level_is_rejected(self):
        bad = {
            "version": "x",
            "rules": [{"id": "R", "level": 3, "citation": "RES5596-N1", "patterns": ["a"], "origen": "x"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                AlarmDetector(path)


class TestAlarmDetector(unittest.TestCase):
    def setUp(self):
        self.detector = AlarmDetector()

    def test_detects_level1_alarm(self):
        result = self.detector.extract("La señora llegó sin pulso y no respira.")
        self.assertEqual(result.n_reglas, 1)
        self.assertIn("ALR-001", [a.rule_id for a in result.alarms])

    def test_no_alarm_gives_none(self):
        result = self.detector.extract("Vengo a pedir un certificado médico para el trabajo.")
        self.assertEqual(result.alarms, [])
        self.assertIsNone(result.n_reglas)

    def test_n_reglas_is_the_most_urgent_alarm(self):
        result = self.detector.extract("Se desmayó y ahora no respira.")
        self.assertEqual({a.level for a in result.alarms}, {1, 2})
        self.assertEqual(result.n_reglas, 1)
        self.assertEqual(result.most_urgent().level, 1)

    def test_negated_respiratory_alarm_does_not_fire(self):
        result = self.detector.extract("No tiene dificultad respiratoria severa ni cianosis.")
        self.assertEqual(result.alarms, [])
        affirmed = self.detector.extract("Tiene dificultad respiratoria severa.")
        self.assertEqual(affirmed.n_reglas, 2)

    def test_negation_window_does_not_cross_clauses(self):
        """El 'sin' de 'sin pulso' no debe negar el 'no respira' siguiente."""
        result = self.detector.extract("llego sin pulso, no respira")
        patterns = {a.matched_pattern for a in result.alarms}
        self.assertEqual(result.n_reglas, 1)
        self.assertTrue({"sin pulso", "no respira"} & patterns)

    def test_lexicon_synonyms_do_not_trigger_level1_consciousness_rule(self):
        """'le cuesta mantenerse despierto' se normaliza igual que 'no
        reacciona'; la regla de nivel I no debe dispararse por eso."""
        result = self.detector.extract("le cuesta mantenerse despierto")
        self.assertNotIn("ALR-002", [a.rule_id for a in result.alarms])

    def test_lexicon_normalized_surface_is_also_searched(self):
        # "esta convulsionando" -> "convulsion activa" (léxico clínico).
        result = self.detector.extract("El niño está convulsionando.")
        self.assertEqual(result.n_reglas, 2)
        self.assertIn("convulsion activa", result.normalized_text)


class TestNegationMechanism(unittest.TestCase):
    """Prueba del mecanismo con un archivo de reglas mínimo (independiente de
    la redacción de alarm_rules.json)."""

    def setUp(self):
        config = {
            "version": "prueba",
            "negation_words": ["no", "sin", "niega", "nunca"],
            "negation_window": 3,
            "rules": [
                {
                    "id": "R-RESP",
                    "level": 2,
                    "citation": "ALARMA-DIFRESP-GRAVE",
                    "patterns": ["dificultad para respirar"],
                    "origen": "criterio_normativo",
                }
            ],
        }
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "rules.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        self.detector = AlarmDetector(path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_tiene_dificultad_para_respirar_is_negated(self):
        self.assertEqual(self.detector.extract("no tiene dificultad para respirar").alarms, [])

    def test_affirmed_phrase_fires(self):
        self.assertEqual(self.detector.extract("tiene dificultad para respirar").n_reglas, 2)

    def test_negation_outside_window_does_not_apply(self):
        result = self.detector.extract("no vino ayer pero hoy tiene mucha dificultad para respirar")
        self.assertEqual(result.n_reglas, 2)

    def test_negated_first_occurrence_affirmed_second_fires(self):
        text = "niega dificultad para respirar en reposo; al caminar si presenta dificultad para respirar"
        self.assertEqual(self.detector.extract(text).n_reglas, 2)


class TestNormalize(unittest.TestCase):
    def test_lowercase_accents_and_lexicon(self):
        self.assertEqual(normalize("Me DUELE el pecho"), "dolor toracico")


if __name__ == "__main__":
    unittest.main()
