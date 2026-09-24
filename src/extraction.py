"""Componente 1: extracción de señales de alarma del relato libre.

Detecta, con reglas explícitas y auditables (``data/alarm_rules.json``), las
señales de alarma que el guardarraíl (``src/guardrail.py``) usa para elevar la
urgencia de la sugerencia del modelo cuando este la subestima. Es un
componente separado de la línea base C0: no comparte código ni archivo de
reglas con ``src/baseline.py``.

Cada patrón se busca en dos superficies del mismo relato:

1. el texto en minúsculas y sin tildes, y
2. el texto normalizado con el léxico coloquial -> clínico de
   ``src/retrieval.py`` (``normalize``).

Buscar solo en (2) no basta: el léxico colapsa expresiones de gravedad
distinta en un mismo término (p. ej. "no reacciona" y "le cuesta mantenerse
despierto" se vuelven ambas "alteracion del estado de conciencia"), y una
regla de nivel I escrita con la expresión coloquial terminaría disparándose
con cualquiera de sus sinónimos.

**Negación simple**: una coincidencia se ignora si alguna palabra de
``negation_words`` aparece entre las ``negation_window`` palabras previas.
La ventana no cruza signos de puntuación (fin de cláusula, como en NegEx):
sin ese límite, "llegó sin pulso, no respira" descartaría "no respira" por
el "sin" de la cláusula anterior.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import LEVEL_BY_CITATION
from .retrieval import _normalize_lexicon, _strip_accents

DEFAULT_ALARM_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "alarm_rules.json"

_WORD_RE = re.compile(r"[a-z_]+")
_CLAUSE_BREAK_RE = re.compile(r"[.,;:!?()\n]")


def normalize(text: str) -> str:
    """Minúsculas, sin tildes y con el léxico coloquial -> clínico que ya usa
    el modo ``clinical_es`` de la recuperación (se reutiliza, no se duplica)."""
    return _normalize_lexicon(text)


def _plain(text: str) -> str:
    return _strip_accents(text.lower())


@dataclass(frozen=True)
class Alarm:
    rule_id: str
    level: int
    citation: str
    matched_pattern: str

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "level": self.level,
            "citation": self.citation,
            "matched_pattern": self.matched_pattern,
        }


@dataclass
class ExtractionResult:
    normalized_text: str
    alarms: list[Alarm] = field(default_factory=list)
    # Nivel más urgente (mínimo) entre las alarmas, o None si no hay ninguna.
    n_reglas: int | None = None

    def most_urgent(self) -> Alarm | None:
        """La alarma que fija ``n_reglas`` (la primera, en el orden del
        archivo de reglas, entre las de nivel mínimo)."""
        if not self.alarms:
            return None
        return min(self.alarms, key=lambda a: a.level)


class AlarmDetector:
    """Detector de alarmas basado en ``data/alarm_rules.json``."""

    def __init__(self, rules_path: Path = DEFAULT_ALARM_RULES_PATH):
        config = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        self.version: str = config.get("version", "sin_version")
        self.negation_words: set[str] = {_plain(w) for w in config.get("negation_words", [])}
        self.negation_window: int = int(config.get("negation_window", 3))
        self.rules: list[dict] = config["rules"]
        self._validate()

    def _validate(self) -> None:
        """Toda regla debe citar un fragmento existente y su nivel debe ser
        el de esa cita: una alarma nunca puede imponer un nivel sin cita."""
        for rule in self.rules:
            citation = rule["citation"]
            if citation not in LEVEL_BY_CITATION:
                raise ValueError(f"Regla {rule['id']}: la cita {citation} no existe en LEVEL_BY_CITATION")
            if LEVEL_BY_CITATION[citation] != rule["level"]:
                raise ValueError(
                    f"Regla {rule['id']}: nivel {rule['level']} no coincide con el de {citation} "
                    f"({LEVEL_BY_CITATION[citation]})"
                )

    def _is_negated(self, surface: str, start: int) -> bool:
        clause = _CLAUSE_BREAK_RE.split(surface[:start])[-1]
        previous = _WORD_RE.findall(clause)[-self.negation_window:] if self.negation_window else []
        return any(word in self.negation_words for word in previous)

    def _affirmed_match(self, surfaces: tuple[str, ...], pattern: str) -> bool:
        needle = _plain(pattern)
        for surface in surfaces:
            start = surface.find(needle)
            while start != -1:
                if not self._is_negated(surface, start):
                    return True
                start = surface.find(needle, start + 1)
        return False

    def extract(self, text: str) -> ExtractionResult:
        normalized = normalize(text)
        surfaces = (_plain(text), normalized)
        alarms: list[Alarm] = []
        for rule in self.rules:
            for pattern in rule["patterns"]:
                if self._affirmed_match(surfaces, pattern):
                    alarms.append(Alarm(rule["id"], rule["level"], rule["citation"], pattern))
                    break  # una alarma por regla
        n_reglas = min((a.level for a in alarms), default=None)
        return ExtractionResult(normalized_text=normalized, alarms=alarms, n_reglas=n_reglas)
