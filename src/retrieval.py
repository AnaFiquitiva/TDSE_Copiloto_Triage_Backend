"""Recuperación sobre el corpus normativo.

Implementa dos variantes del Factor B del diseño 2x2 (Sección 5 del documento
del proyecto):

- ``generic``: vectorización TF-IDF simple, sin ningún ajuste al dominio
  clínico en español. Sirve de proxy determinista y reproducible de un modelo
  de *embeddings* multilingüe genérico.
- ``clinical_es``: la misma vectorización TF-IDF, mediante un léxico de
  normalización que mapea expresiones coloquiales del relato del paciente a
  su término clínico equivalente antes de vectorizar (p. ej. "me falta el
  aire" -> "dificultad_respiratoria"). Sirve de proxy determinista de un
  modelo de *embeddings* clínico en español (p. ej. Carrino et al., 2022).

Estas dos variantes son deterministas y no requieren red ni credenciales:
son las del estudio de recuperación E1-E4 y la ruta de respaldo. Además, este
módulo ofrece `GeminiRetriever` (más abajo), que usa *embeddings* reales de la
API de Gemini, y es la parte densa de la ruta principal (ver README.md,
sección "Arquitectura").
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import llm_client
from .ingest import Chunk
from .llm_client import LLMUnavailableError, embed_text

# Léxico de normalización coloquial -> clínico usado únicamente por la
# variante ``clinical_es``. Las claves están ya normalizadas (minúsculas,
# sin tildes) para que el emparejamiento sea robusto frente a variaciones
# ortográficas del relato libre.
_CLINICAL_LEXICON: dict[str, str] = {
    "dolor en el pecho": "dolor toracico",
    "dolor de pecho": "dolor toracico",
    "me duele el pecho": "dolor toracico",
    "me falta el aire": "dificultad respiratoria",
    "no puedo respirar bien": "dificultad respiratoria",
    "cuesta respirar": "dificultad respiratoria",
    "se esta ahogando": "dificultad respiratoria",
    "se desmayo": "alteracion del estado de conciencia",
    "perdio el conocimiento": "alteracion del estado de conciencia",
    "no reacciona": "alteracion del estado de conciencia",
    "esta muy confundido": "alteracion del estado de conciencia",
    "sangra mucho": "hemorragia",
    "no para de sangrar": "hemorragia",
    "esta sangrando bastante": "hemorragia",
    "tiene mucha fiebre": "fiebre alta",
    "esta muy caliente": "fiebre alta",
    "quiere hacerse dano": "ideacion suicida",
    "quiere quitarse la vida": "ideacion suicida",
    "piensa en suicidarse": "ideacion suicida",
    "se golpeo la cabeza": "trauma craneoencefalico",
    "se pego en la cabeza": "trauma craneoencefalico",
    "le dieron temblores fuertes": "convulsion",
    "empezo a convulsionar": "convulsion",
    "convulsionando sin parar": "convulsion activa",
    "esta convulsionando": "convulsion activa",
    "no ha recuperado el conocimiento entre los episodios": "no recupera el estado de conciencia estado epileptico",
    "ya paso la convulsion": "convulsion ya cedio",
    "vomito en chorro": "vomito en proyectil",
    # Agregados al ampliar el corpus con 7 motivos de consulta nuevos (ver
    # corpus/reformatted/signos_alarma.md): sin estas entradas, el modo
    # clinical_es no aportaba ninguna ventaja sobre 'generic' para estos
    # motivos, porque el relato coloquial no comparte token literal con el
    # texto de la guia.
    "le duele mucho la barriga": "dolor abdominal",
    "dolor en la barriga": "dolor abdominal",
    "dolor en el estomago": "dolor abdominal",
    "la barriga esta dura como una tabla": "abdomen en tabla rigidez generalizada",
    "duele mucho cuando le presionan y sueltan": "dolor a la descompresion signo de rebote",
    "dolor de cabeza": "cefalea",
    "el peor dolor de cabeza de su vida": "cefalea subita la peor de la vida",
    "cuello muy rigido": "rigidez de nuca",
    "el cuello muy duro": "rigidez de nuca",
    "dolor de cabeza parecido al que le da siempre": "cefalea habitual tipo tensional",
    "se quemo": "quemadura",
    "muy hinchados": "hinchazon",
    "se le cierra la garganta": "hinchazon de garganta",
    "no tiene la cara hinchada": "sin hinchazon facial",
    "se tomo varias pastillas": "intoxicacion sustancia",
    "tomo mas licor de lo normal": "intoxicacion sustancia de bajo riesgo",
    "le cuesta mantenerse despierto": "alteracion del estado de conciencia",
    "esta embarazada": "gestante embarazo",
    "empezo a sangrar bastante por la vagina": "sangrado vaginal abundante durante el embarazo",
}


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_lexicon(text: str) -> str:
    """Reemplaza expresiones coloquiales por su equivalente clínico."""
    normalized = _strip_accents(text.lower())
    for phrase, canonical in _CLINICAL_LEXICON.items():
        normalized = normalized.replace(phrase, canonical)
    return normalized


_TOKEN_RE = re.compile(r"[a-z_]+")

# Nota de una prueba descartada: se evaluó agregar bigramas de palabras
# adyacentes (p. ej. "dolor_toracico" además de "dolor" y "toracico") para
# que frases cortas pesaran como unidad. Medido sobre las 74 viñetas de
# data/cases.json, SÍ mejoró recall@k (0.606->0.667) y la fidelidad de
# citación, pero EMPEORÓ la sensibilidad en niveles I-II (0.394->0.333,
# celda E4) porque el vocabulario más denso diluye el margen de confianza
# entre las ramas GRAVE/LEVE de un mismo motivo de consulta. Como la
# sensibilidad en niveles I-II es la métrica de seguridad más importante del
# proyecto (H2), se descartó el cambio y se documenta aquí para no
# reintentarlo sin volver a medir el mismo trade-off.


def tokenize(text: str, embedding_mode: str) -> list[str]:
    if embedding_mode == "clinical_es":
        text = _normalize_lexicon(text)
    else:
        text = _strip_accents(text.lower())
    return _TOKEN_RE.findall(text)


@dataclass
class RankedChunk:
    chunk: Chunk
    score: float


class Retriever:
    """Índice TF-IDF minimalista, sin dependencias externas."""

    def __init__(self, chunks: list[Chunk], embedding_mode: str = "generic"):
        if embedding_mode not in {"generic", "clinical_es"}:
            raise ValueError("embedding_mode debe ser 'generic' o 'clinical_es'")
        self.chunks = chunks
        self.embedding_mode = embedding_mode
        self._doc_tokens = [tokenize(c.text, embedding_mode) for c in chunks]
        self._df = self._document_frequencies(self._doc_tokens)
        self._n_docs = len(chunks)
        self._doc_vectors = [
            self._tfidf_vector(tokens) for tokens in self._doc_tokens
        ]

    @staticmethod
    def _document_frequencies(doc_tokens: list[list[str]]) -> Counter:
        df: Counter = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] += 1
        return df

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # +1 en numerador y denominador (suavizado) para evitar división por
        # cero y para no descartar términos que solo aparecen en la consulta.
        return math.log((1 + self._n_docs) / (1 + df)) + 1.0

    def _tfidf_vector(self, tokens: list[str]) -> Counter:
        tf = Counter(tokens)
        vec: Counter = Counter()
        for term, count in tf.items():
            vec[term] = count * self._idf(term)
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return Counter({k: v / norm for k, v in vec.items()})

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        common = set(a) & set(b)
        return sum(a[t] * b[t] for t in common)

    def retrieve(self, query: str, k: int = 3) -> list[RankedChunk]:
        query_tokens = tokenize(query, self.embedding_mode)
        query_vec = self._tfidf_vector(query_tokens)
        scored = [
            RankedChunk(chunk=chunk, score=self._cosine(query_vec, doc_vec))
            for chunk, doc_vec in zip(self.chunks, self._doc_vectors)
        ]
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:k]


DEFAULT_EMBEDDING_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / ".gemini_embedding_cache.json"
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class GeminiRetriever:
    """Recuperación densa con *embeddings* reales de la API de Gemini
    (`gemini-embedding-001` por defecto), en vez de TF-IDF. Es la parte densa
    de la ruta principal; requiere `GEMINI_API_KEY` o una caché completa.

    Usa la codificación **asimétrica** que ofrece `gemini-embedding-001`: los
    fragmentos del corpus se embeben con `task_type="RETRIEVAL_DOCUMENT"` y el
    relato del paciente con `task_type="RETRIEVAL_QUERY"`. Estos dos vectores
    para el mismo texto NO son iguales (la similitud coseno entre ambos es de
    apenas ~0.84 en la práctica): usar el `task_type` correcto en cada lado,
    en vez del comportamiento por defecto de la API (equivalente a tratar
    todo como consulta), es lo que la propia documentación de Gemini
    recomienda para tareas de recuperación y mejora la calidad del ranking.

    Los embeddings (de los fragmentos Y de las consultas) se cachean en disco
    (`cache_path`, por defecto `data/.gemini_embedding_cache.json`, excluido
    de git). La clave combina el modelo, el `task_type` y un hash del texto
    exacto, así que un cambio en el corpus o de modelo invalida solo las
    entradas afectadas. La caché respeta `GEMINI_CACHE_MODE` (ver
    `llm_client`): en `readonly` una entrada faltante lanza
    `LLMUnavailableError` en lugar de llamar a la API, y en `off` no se lee
    ni se escribe.

    Implementa la misma interfaz que `Retriever` (`.retrieve(query, k)`).
    """

    def __init__(
        self,
        chunks: list[Chunk],
        cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
        embed_fn=embed_text,
        model: str | None = None,
    ):
        self.chunks = chunks
        self.cache_path = cache_path
        self.model = model or llm_client.EMBEDDING_MODEL
        self.name = self.model
        self._embed_fn = embed_fn
        self._cache = self._load_cache()
        self._doc_vectors = [
            self._embed_cached(c.text, "RETRIEVAL_DOCUMENT") for c in chunks
        ]

    def _load_cache(self) -> dict[str, list[float]]:
        if not self.cache_path.exists():
            return {}
        cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        # Migración de la caché v1, cuyas claves no incluían el modelo
        # ("TASK:hash"): todas se generaron con gemini-embedding-001.
        migrated = {}
        for key, vec in cache.items():
            if key.count(":") == 1:
                key = f"{llm_client.DEFAULT_EMBEDDING_MODEL}:{key}"
            migrated[key] = vec
        return migrated

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")

    def _embed_cached(self, text: str, task_type: str) -> list[float]:
        mode = llm_client.get_cache_mode()
        key = f"{self.model}:{task_type}:{_text_hash(text)}"
        if mode != "off" and key in self._cache:
            return self._cache[key]
        if mode == "readonly":
            raise LLMUnavailableError(
                "GEMINI_CACHE_MODE=readonly y la caché de embeddings no tiene esta entrada."
            )
        vector = self._embed_fn(text, task_type=task_type)
        if mode == "readwrite":
            self._cache[key] = vector
            self._save_cache()
        return vector

    def retrieve(self, query: str, k: int = 5) -> list[RankedChunk]:
        query_vec = self._embed_cached(query, "RETRIEVAL_QUERY")
        scored = [
            RankedChunk(chunk=chunk, score=_cosine_similarity(query_vec, doc_vec))
            for chunk, doc_vec in zip(self.chunks, self._doc_vectors)
        ]
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:k]


class BM25Retriever:
    """BM25 (Okapi) léxico, solo con la biblioteca estándar. Reutiliza
    ``tokenize`` (con el léxico clínico en modo ``clinical_es``). Es la parte
    léxica de la ruta principal híbrida: aporta coincidencia exacta de
    términos clínicos que un modelo denso puede diluir.

    IDF con la variante no negativa de Lucene:
    ``log(1 + (N - df + 0.5) / (df + 0.5))``.
    """

    name = "bm25"

    def __init__(self, chunks: list[Chunk], embedding_mode: str = "clinical_es", k1: float = 1.5, b: float = 0.75):
        if embedding_mode not in {"generic", "clinical_es"}:
            raise ValueError("embedding_mode debe ser 'generic' o 'clinical_es'")
        self.chunks = chunks
        self.embedding_mode = embedding_mode
        self.k1 = k1
        self.b = b
        self.name = f"bm25-{embedding_mode}"
        self._doc_tf = [Counter(tokenize(c.text, embedding_mode)) for c in chunks]
        self._doc_len = [sum(tf.values()) for tf in self._doc_tf]
        self._avgdl = (sum(self._doc_len) / len(chunks)) if chunks else 0.0
        df: Counter = Counter()
        for tf in self._doc_tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def _score(self, query_terms: list[str], i: int) -> float:
        tf = self._doc_tf[i]
        norm = self.k1 * (1 - self.b + self.b * self._doc_len[i] / (self._avgdl or 1.0))
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f:
                score += self._idf[term] * f * (self.k1 + 1) / (f + norm)
        return score

    def retrieve(self, query: str, k: int = 5) -> list[RankedChunk]:
        # Términos únicos de la consulta: repetir una palabra en el relato no
        # debe multiplicar su peso.
        terms = list(dict.fromkeys(tokenize(query, self.embedding_mode)))
        scored = [RankedChunk(chunk=c, score=self._score(terms, i)) for i, c in enumerate(self.chunks)]
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:k]


class DenseRetriever:
    """Adaptador genérico de recuperación densa para CUALQUIER función de
    embeddings ``embed_fn(texts, is_query) -> list[vector]`` (puerto de la
    arquitectura). Permite enchufar un modelo local (``src/local_embeddings``)
    o embeddings falsos en pruebas, sin tocar el resto del pipeline."""

    def __init__(self, chunks: list[Chunk], embed_fn, name: str = "dense"):
        self.chunks = chunks
        self.name = name
        self._embed_fn = embed_fn
        self._doc_vectors = embed_fn([c.text for c in chunks], False) if chunks else []

    def retrieve(self, query: str, k: int = 5) -> list[RankedChunk]:
        query_vec = self._embed_fn([query], True)[0]
        scored = [
            RankedChunk(chunk=c, score=_cosine_similarity(query_vec, v))
            for c, v in zip(self.chunks, self._doc_vectors)
        ]
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:k]


class HybridRetriever:
    """Fusión de varios recuperadores por Reciprocal Rank Fusion (Cormack et
    al., 2009): ``score(d) = Σ_r 1 / (rrf_k + rank_r(d))``, con ``rank``
    1-indexado dentro de los ``pool`` primeros resultados de cada
    recuperador. Devuelve ``RankedChunk`` con el puntaje fusionado (que NO es
    una similitud: su escala es ~1/rrf_k).

    Si un recuperador lanza ``LLMUnavailableError`` (p. ej. Gemini sin red),
    el error se propaga a propósito: el pipeline cae entonces a la ruta de
    respaldo completa y lo marca como degradado, en vez de seguir en silencio
    con solo la mitad léxica de la ruta principal.
    """

    def __init__(self, retrievers: list, rrf_k: int = 60, pool: int = 20):
        if not retrievers:
            raise ValueError("HybridRetriever necesita al menos un recuperador")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.pool = pool
        self.name = "rrf(" + "+".join(getattr(r, "name", type(r).__name__) for r in retrievers) + ")"

    def retrieve(self, query: str, k: int = 5) -> list[RankedChunk]:
        fused: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        first_seen: dict[str, int] = {}
        for retriever in self.retrievers:
            for rank, rc in enumerate(retriever.retrieve(query, k=self.pool), start=1):
                cid = rc.chunk.chunk_id
                chunk_by_id.setdefault(cid, rc.chunk)
                first_seen.setdefault(cid, len(first_seen))
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        # Empates: se rompen por orden de primera aparición (determinista).
        ordered = sorted(fused, key=lambda cid: (-fused[cid], first_seen[cid]))
        return [RankedChunk(chunk=chunk_by_id[cid], score=fused[cid]) for cid in ordered[:k]]
