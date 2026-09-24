"""Pruebas de la recuperación híbrida (BM25 + densa, fusión RRF), sin red."""
import unittest

from src.ingest import Chunk, load_corpus
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RankedChunk


def _chunks():
    return [
        Chunk("A", "fiebre alta en menores con rigidez de nuca", "s", "reformatted", level=1),
        Chunk("B", "dolor toracico opresivo irradiado al brazo", "s", "reformatted", level=2),
        Chunk("C", "tramite administrativo certificado medico incapacidad", "s", "reformatted", level=5),
    ]


class _StaticRetriever:
    """Devuelve un ranking fijo (por ids), para probar RRF a mano."""

    def __init__(self, chunks_by_id, ids, name):
        self._ranked = [RankedChunk(chunks_by_id[i], 1.0 / (n + 1)) for n, i in enumerate(ids)]
        self.name = name

    def retrieve(self, query, k=5):
        return self._ranked[:k]


class TestBM25(unittest.TestCase):
    def test_exact_term_document_ranks_first(self):
        retriever = BM25Retriever(_chunks(), embedding_mode="generic")
        top = retriever.retrieve("necesito un certificado", k=3)
        self.assertEqual(top[0].chunk.chunk_id, "C")
        self.assertGreater(top[0].score, 0)
        self.assertEqual(top[1].score, 0.0)

    def test_uses_clinical_lexicon_in_clinical_mode(self):
        retriever = BM25Retriever(_chunks(), embedding_mode="clinical_es")
        # "dolor en el pecho" -> "dolor toracico" por el léxico clínico.
        self.assertEqual(retriever.retrieve("tiene dolor en el pecho", k=1)[0].chunk.chunk_id, "B")

    def test_rare_term_weighs_more_than_common_term(self):
        chunks = [
            Chunk("X", "dolor dolor dolor cabeza", "s", "reformatted", level=4),
            Chunk("Y", "dolor petequias", "s", "reformatted", level=1),
            Chunk("Z", "dolor leve", "s", "reformatted", level=4),
        ]
        top = BM25Retriever(chunks, embedding_mode="generic").retrieve("dolor con petequias", k=1)
        self.assertEqual(top[0].chunk.chunk_id, "Y")

    def test_works_on_real_corpus(self):
        retriever = BM25Retriever(load_corpus("reformatted"))
        top = retriever.retrieve("tiene el abdomen en tabla y dolor a la descompresion", k=1)
        self.assertEqual(top[0].chunk.chunk_id, "ALARMA-ABDOMEN-GRAVE")


class TestRRF(unittest.TestCase):
    def test_fusion_of_two_known_rankings(self):
        by_id = {c.chunk_id: c for c in _chunks()}
        r1 = _StaticRetriever(by_id, ["A", "B", "C"], "r1")
        r2 = _StaticRetriever(by_id, ["B", "C", "A"], "r2")
        fused = HybridRetriever([r1, r2], rrf_k=60).retrieve("q", k=3)
        # A: 1/61 + 1/63 ; B: 1/62 + 1/61 ; C: 1/63 + 1/62
        expected = {"A": 1 / 61 + 1 / 63, "B": 1 / 62 + 1 / 61, "C": 1 / 63 + 1 / 62}
        self.assertEqual([rc.chunk.chunk_id for rc in fused], ["B", "A", "C"])
        for rc in fused:
            self.assertAlmostEqual(rc.score, expected[rc.chunk.chunk_id])

    def test_document_found_by_only_one_retriever_still_counts(self):
        by_id = {c.chunk_id: c for c in _chunks()}
        r1 = _StaticRetriever(by_id, ["A"], "r1")
        r2 = _StaticRetriever(by_id, ["C"], "r2")
        fused = HybridRetriever([r1, r2], rrf_k=60).retrieve("q", k=5)
        self.assertEqual({rc.chunk.chunk_id for rc in fused}, {"A", "C"})
        # Empate exacto: se resuelve por orden de primera aparición.
        self.assertEqual(fused[0].chunk.chunk_id, "A")

    def test_name_describes_components(self):
        by_id = {c.chunk_id: c for c in _chunks()}
        hybrid = HybridRetriever([_StaticRetriever(by_id, ["A"], "bm25"), _StaticRetriever(by_id, ["A"], "dense")])
        self.assertEqual(hybrid.name, "rrf(bm25+dense)")


def _fake_embed(texts, is_query):
    """Embeddings falsos: bolsa de palabras sobre un vocabulario fijo."""
    vocab = ["fiebre", "nuca", "pecho", "toracico", "brazo", "certificado", "incapacidad"]
    return [[float(w in t.lower()) for w in vocab] for t in texts]


class TestHybridWithFakeDense(unittest.TestCase):
    def test_hybrid_bm25_plus_fake_dense_without_network(self):
        chunks = _chunks()
        dense = DenseRetriever(chunks, _fake_embed, name="falso")
        hybrid = HybridRetriever([BM25Retriever(chunks, "clinical_es"), dense], pool=3)
        top = hybrid.retrieve("dolor en el pecho que se va al brazo", k=2)
        self.assertEqual(top[0].chunk.chunk_id, "B")
        self.assertEqual(len(top), 2)
        self.assertEqual(hybrid.name, "rrf(bm25-clinical_es+falso)")

    def test_dense_retriever_ranks_by_cosine(self):
        dense = DenseRetriever(_chunks(), _fake_embed, name="falso")
        self.assertEqual(dense.retrieve("fiebre y nuca rigida", k=1)[0].chunk.chunk_id, "A")


class TestLocalEmbeddingsIsOptional(unittest.TestCase):
    def test_module_imports_without_the_dependency(self):
        import importlib.util

        import src.local_embeddings as local

        if importlib.util.find_spec("sentence_transformers") is None:
            with self.assertRaises(ImportError) as ctx:
                local.E5LocalEmbedder()
            self.assertIn("requirements-embeddings.txt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
