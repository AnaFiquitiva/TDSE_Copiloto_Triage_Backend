"""Adaptador OPCIONAL de embeddings locales (``intfloat/multilingual-e5-small``)
para el puerto ``DenseRetriever`` de ``src/retrieval.py``.

No se usa por defecto y no es una dependencia del núcleo: requiere instalar
aparte ``pip install -r requirements-embeddings.txt``. La importación de
``sentence_transformers`` es perezosa (solo al construir el adaptador), así
que importar este módulo nunca falla; si falta la biblioteca, se lanza un
``ImportError`` con instrucciones claras.

Uso:

    from src.local_embeddings import E5LocalEmbedder
    from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

    dense = DenseRetriever(chunks, E5LocalEmbedder(), name="e5-small-local")
    retriever = HybridRetriever([BM25Retriever(chunks), dense])

Los modelos E5 exigen los prefijos ``query:`` (consulta) y ``passage:``
(documento): es su forma de codificación asimétrica, análoga al
``task_type`` de ``gemini-embedding-001``.
"""
from __future__ import annotations

DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-small"


class E5LocalEmbedder:
    """``embed_fn(texts, is_query) -> list[vector]`` con un modelo E5 local."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - import perezoso a propósito
        except ImportError as exc:
            raise ImportError(
                "Los embeddings locales son opcionales y requieren 'sentence-transformers'. "
                "Instálalos con: pip install -r requirements-embeddings.txt"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str], is_query: bool) -> list[list[float]]:
        prefix = "query: " if is_query else "passage: "
        vectors = self._model.encode([prefix + t for t in texts], normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]
