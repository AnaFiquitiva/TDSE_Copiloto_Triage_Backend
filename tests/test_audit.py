"""Pruebas del registro de auditoría v2: hash encadenado, eventos de
decisión de solo anexado y métrica de completitud."""
import json
import tempfile
import unittest
from pathlib import Path

from src.audit import (
    GENESIS_HASH,
    AuditLog,
    build_audit_record,
    completeness,
    verify_chain,
)
from src.generator import Suggestion
from src.ingest import Chunk
from src.retrieval import RankedChunk


def _suggestion(level=2, abstained=False):
    chunk = Chunk("RES5596-N2", "texto", "niveles", "reformatted", level=2)
    return Suggestion(
        level=None if abstained else level,
        citation=None if abstained else "RES5596-N2",
        confidence=0.7,
        abstained=abstained,
        reason="evidencia_insuficiente" if abstained else "ok",
        retrieved=[RankedChunk(chunk, 0.5)],
    )


def _record(case_id="C1", abstained=False):
    return build_audit_record(
        case_id=case_id,
        patient_text="relato",
        corpus_format="reformatted",
        embedding_mode="clinical_es",
        suggestion=_suggestion(abstained=abstained),
        corpus_hash="a" * 64,
        generation_model="minimal-generator-v1",
        embedding_model="tfidf-clinical_es",
        prompt_version=None,
        thresholds={"MIN_SCORE": 0.08, "MIN_RELATIVE_MARGIN": 0.2, "k": 5},
    )


class TestHashChain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.log = AuditLog(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_event_chains_to_genesis(self):
        event = self.log.append(_record())
        self.assertEqual(event["prev_hash"], GENESIS_HASH)
        self.assertEqual(len(event["hash"]), 64)

    def test_chain_is_valid_after_appends(self):
        for i in range(3):
            self.log.append(_record(f"C{i}"))
        self.assertEqual(self.log.verify_chain(), (True, None))

    def test_modifying_a_record_breaks_the_chain(self):
        for i in range(3):
            self.log.append(_record(f"C{i}"))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[1])
        tampered["suggested_level"] = 5  # alguien "maquilla" el nivel sugerido
        lines[1] = json.dumps(tampered, ensure_ascii=False)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(self.log.verify_chain(), (False, 1))

    def test_deleting_a_record_breaks_the_chain(self):
        for i in range(3):
            self.log.append(_record(f"C{i}"))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        del lines[0]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok, first_bad = verify_chain(self.log.read_all())
        self.assertFalse(ok)
        self.assertEqual(first_bad, 0)


class TestDecisions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = AuditLog(Path(self.tmp.name) / "audit.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_decision_is_a_new_event_and_original_is_untouched(self):
        original = self.log.append(_record())
        decision = self.log.append_decision(
            original["hash"], "corrige", final_level=1, reason="signos de shock", user="enf1", review_seconds=12.5
        )
        records = self.log.read_all()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0], original)
        self.assertIsNone(records[0]["human_decision"])
        self.assertEqual(decision["ref_hash"], original["hash"])
        self.assertEqual(decision["final_level"], 1)
        self.assertEqual(self.log.verify_chain(), (True, None))
        self.assertEqual(self.log.find(decision["hash"]), decision)

    def test_confirm_uses_suggested_level(self):
        original = self.log.append(_record())
        decision = self.log.append_decision(original["hash"], "confirma", review_seconds=3)
        self.assertEqual(decision["final_level"], 2)

    def test_correct_requires_level_and_reason(self):
        original = self.log.append(_record())
        with self.assertRaises(ValueError):
            self.log.append_decision(original["hash"], "corrige", final_level=1, reason="  ")
        with self.assertRaises(ValueError):
            self.log.append_decision(original["hash"], "corrige", final_level=None, reason="x")

    def test_cannot_confirm_an_abstention(self):
        original = self.log.append(_record(abstained=True))
        with self.assertRaises(ValueError):
            self.log.append_decision(original["hash"], "confirma")

    def test_unknown_hash_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.log.append_decision("f" * 64, "confirma")


class TestCompleteness(unittest.TestCase):
    def test_new_records_are_100_percent_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AuditLog(Path(tmp) / "audit.jsonl")
            first = log.append(_record())
            log.append(_record("C2", abstained=True))
            log.append_decision(first["hash"], "confirma", review_seconds=1)
            result = completeness(log.read_all())
            self.assertEqual(result["completitud"], 1.0)
            self.assertEqual(result["n_registros"], 3)

    def test_missing_field_lowers_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AuditLog(Path(tmp) / "audit.jsonl")
            good = log.append(_record())
            bad = dict(good)
            del bad["corpus_hash"]
            result = completeness([good, bad])
            self.assertEqual(result["completitud"], 0.5)
            self.assertEqual(result["faltantes"], {"corpus_hash": 1})

    def test_level_without_citation_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = AuditLog(Path(tmp) / "audit.jsonl").append(_record())
            record["citation"] = None
            self.assertEqual(completeness([record])["completitud"], 0.0)


if __name__ == "__main__":
    unittest.main()
