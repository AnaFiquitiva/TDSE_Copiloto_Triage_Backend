"""Ingesta de corpus normativo: convierte los documentos crudos y reformateados
en una lista de fragmentos (chunks) indexables, cada uno con un identificador de
cita explícito y, siempre que sea posible, un único nivel de triage asociado.

La dificultad de parsear el formato "crudo" (líneas de flujograma sin
encabezados semánticos y tabla con separadores `|`) es intencional: reproduce,
en miniatura, el reto descrito en la Sección 3.3 del documento del proyecto
(guías con layout complejo).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

# Nivel asociado a cada cita del corpus. Cada fragmento se diseñó, a
# propósito, para corresponder a un único nivel (incluso los que documentan
# una rama de un flujograma de señales de alarma), de modo que la sugerencia
# del sistema siempre pueda anclarse a una cita trazable y no dependa de
# resolver una ambigüedad interna del propio fragmento.
LEVEL_BY_CITATION = {
    "RES5596-N1": 1,
    "RES5596-N2": 2,
    "RES5596-N3": 3,
    "RES5596-N4": 4,
    "RES5596-N5": 5,
    "ALARMA-DOLOR-TORACICO-GRAVE": 2,
    "ALARMA-DOLOR-TORACICO-LEVE": 3,
    "ALARMA-DIFRESP-GRAVE": 2,
    "ALARMA-DIFRESP-LEVE": 3,
    "ALARMA-FIEBRE-GRAVE": 1,
    "ALARMA-FIEBRE-LEVE": 4,
    "ALARMA-TRAUMA-GRAVE": 1,
    "ALARMA-TRAUMA-LEVE": 4,
    "ALARMA-SUICIDA-GRAVE": 2,
    "ALARMA-SUICIDA-LEVE": 3,
    "ALARMA-SANGRADO-GRAVE": 1,
    "ALARMA-SANGRADO-LEVE": 4,
    "ALARMA-ABDOMEN-GRAVE": 1,
    "ALARMA-ABDOMEN-LEVE": 4,
    "ALARMA-CONVULSION-GRAVE": 1,
    "ALARMA-CONVULSION-LEVE": 3,
    "ALARMA-CEFALEA-GRAVE": 1,
    "ALARMA-CEFALEA-LEVE": 4,
    "ALARMA-QUEMADURA-GRAVE": 1,
    "ALARMA-QUEMADURA-LEVE": 4,
    "ALARMA-ALERGIA-GRAVE": 1,
    "ALARMA-ALERGIA-LEVE": 4,
    "ALARMA-INTOXICACION-GRAVE": 1,
    "ALARMA-INTOXICACION-LEVE": 3,
    "ALARMA-GESTANTE-GRAVE": 1,
    "ALARMA-GESTANTE-LEVE": 5,
}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    corpus_format: str  # "raw" | "reformatted"
    level: int | None = None


def _reformatted_chunks(path: Path) -> list[Chunk]:
    raw_text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?=^## )", raw_text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    for section in sections:
        m = re.search(r"\[CITA:\s*([A-Z0-9\-]+)\]", section)
        if not m:
            continue
        citation = m.group(1)
        text = section.strip()
        chunks.append(
            Chunk(
                chunk_id=citation,
                text=text,
                source=path.stem,
                corpus_format="reformatted",
                level=LEVEL_BY_CITATION.get(citation),
            )
        )
    return chunks


_RAW_ROW_RE = re.compile(r"^(N\d)\|([^|]+)\|([^|]+)\|(.+)$", re.MULTILINE)
_RAW_FLOW_RE = re.compile(r"^Flujograma ([^\->]+?)\s*->\s*(.+)$", re.MULTILINE)

# Orden en el que aparecen los flujogramas en el documento crudo (una línea
# por rama). Se usa EXCLUSIVAMENTE en la evaluación (experiments/) para poder
# comparar de forma justa el recall@k entre el corpus crudo y el reformateado,
# ya que el parser crudo pierde el identificador semántico de cada rama (ver
# `_raw_chunks`). El sistema de recuperación y generación NO tiene acceso a
# este mapeo: solo lo usa el evaluador externo para saber qué contó como acierto.
RAW_ALARM_CITATION_ORDER = [
    "ALARMA-DOLOR-TORACICO-GRAVE",
    "ALARMA-DOLOR-TORACICO-LEVE",
    "ALARMA-DIFRESP-GRAVE",
    "ALARMA-DIFRESP-LEVE",
    "ALARMA-FIEBRE-GRAVE",
    "ALARMA-FIEBRE-LEVE",
    "ALARMA-TRAUMA-GRAVE",
    "ALARMA-TRAUMA-LEVE",
    "ALARMA-SUICIDA-GRAVE",
    "ALARMA-SUICIDA-LEVE",
    "ALARMA-SANGRADO-GRAVE",
    "ALARMA-SANGRADO-LEVE",
    "ALARMA-ABDOMEN-GRAVE",
    "ALARMA-ABDOMEN-LEVE",
    "ALARMA-CONVULSION-GRAVE",
    "ALARMA-CONVULSION-LEVE",
    "ALARMA-CEFALEA-GRAVE",
    "ALARMA-CEFALEA-LEVE",
    "ALARMA-QUEMADURA-GRAVE",
    "ALARMA-QUEMADURA-LEVE",
    "ALARMA-ALERGIA-GRAVE",
    "ALARMA-ALERGIA-LEVE",
    "ALARMA-INTOXICACION-GRAVE",
    "ALARMA-INTOXICACION-LEVE",
    "ALARMA-GESTANTE-GRAVE",
    "ALARMA-GESTANTE-LEVE",
]


def _raw_chunks(path: Path) -> list[Chunk]:
    """Parsea el formato crudo (tabla con `|` o flujogramas en una línea).

    A propósito, este parser es más frágil que el de la versión reformateada:
    depende de una expresión regular ajustada al layout específico del
    documento crudo, tal como ocurriría con un extractor genérico de PDF a
    texto enfrentado a una tabla real. En particular, las líneas de
    flujograma no traen un identificador semántico de cita (a diferencia del
    `[CITA: ...]` explícito de la versión reformateada): solo se les puede
    asignar un id posicional, lo que es, en sí mismo, una manifestación
    concreta de la brecha G3 descrita en el documento del proyecto.
    """
    raw_text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []

    for match in _RAW_ROW_RE.finditer(raw_text):
        code, _desc, _criterios, _tiempo = match.groups()
        citation = f"RES5596-{code}"
        text = match.group(0).replace("|", " | ")
        chunks.append(
            Chunk(
                chunk_id=citation,
                text=text,
                source=path.stem,
                corpus_format="raw",
                level=LEVEL_BY_CITATION.get(citation),
            )
        )

    for i, match in enumerate(_RAW_FLOW_RE.finditer(raw_text)):
        topic, body = match.groups()
        citation = f"ALARMA-RAW-{i}"
        # El nivel se extrae directamente del propio texto crudo (sufijo
        # "=> N<d>"), tal como lo haría un extractor de información genérico;
        # lo que el formato crudo pierde no es el dato, sino la cita semántica
        # limpia que sí tiene la versión reformateada.
        level_match = re.search(r"N(\d)\s*$", body.strip())
        level = int(level_match.group(1)) if level_match else None
        chunks.append(
            Chunk(
                chunk_id=citation,
                text=f"{topic.strip()}: {body.strip()}",
                source=path.stem,
                corpus_format="raw",
                level=level,
            )
        )
    return chunks


def canonical_citation(chunk_id: str) -> str:
    """Traduce un id de chunk crudo (p.ej. ALARMA-RAW-2) a su cita semántica
    equivalente en la versión reformateada, para fines de evaluación."""
    if chunk_id.startswith("ALARMA-RAW-"):
        idx = int(chunk_id.rsplit("-", 1)[-1])
        return RAW_ALARM_CITATION_ORDER[idx]
    return chunk_id


def load_corpus(corpus_format: str) -> list[Chunk]:
    """Carga todos los chunks de un formato de corpus ('raw' o 'reformatted')."""
    if corpus_format not in {"raw", "reformatted"}:
        raise ValueError("corpus_format debe ser 'raw' o 'reformatted'")

    directory = CORPUS_DIR / corpus_format
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*")):
        if corpus_format == "reformatted":
            chunks.extend(_reformatted_chunks(path))
        else:
            chunks.extend(_raw_chunks(path))
    return chunks


def corpus_hash(corpus_format: str) -> str:
    """SHA-256 del contenido canónico de los fragmentos de un corpus.

    El contenido canónico es la lista de fragmentos ordenada por
    ``chunk_id``, cada uno con su texto y su nivel, serializada como JSON con
    llaves ordenadas. Así el hash cambia si cambia cualquier texto o nivel
    citable, pero no por el orden de los archivos en disco ni por los saltos
    de línea del sistema operativo (``read_text`` ya los normaliza a ``\n``).
    Se registra en cada evento de auditoría para saber exactamente contra qué
    versión del corpus se generó una sugerencia.
    """
    chunks = sorted(load_corpus(corpus_format), key=lambda c: c.chunk_id)
    canonical = json.dumps(
        [{"chunk_id": c.chunk_id, "text": c.text, "level": c.level} for c in chunks],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    for fmt in ("raw", "reformatted"):
        print(f"--- {fmt} ---")
        for c in load_corpus(fmt):
            print(c.chunk_id, "|", c.level, "|", c.text[:60].replace("\n", " "))
