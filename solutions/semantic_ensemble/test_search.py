from unittest import TestCase

from vk_mail_search.data import Document
from vk_mail_search.dated import DatedSearcher
from vk_mail_search.period_rank import PeriodRanker
from vk_mail_search.query_parser import parse_query

from .search import SemanticCandidatePool, blend_candidates
from .word_bm25 import SemanticBM25


class SemanticCandidatePoolTest(TestCase):
    def test_blend_considers_both_scores_and_preserves_pure_modes(self):
        lexical = [("a", 10.), ("b", 9.), ("c", 1.)]
        semantic = [("c", 20.), ("b", 18.), ("a", 2.)]
        mixed = blend_candidates(lexical, semantic, .5)
        self.assertEqual([doc_id for doc_id, _ in mixed], ["b", "a", "c"])
        self.assertAlmostEqual(mixed[0][1], .9)
        self.assertEqual(blend_candidates(lexical, semantic, 0.), lexical)
        self.assertEqual(blend_candidates(lexical, semantic, 1.), semantic)

    def test_can_admit_a_document_below_position_thirty_without_changing_final_bm25_values(self):
        day = "2028-09-13"
        documents = [Document(f"doc-match-{i:02}", "box", day, "выгрузка") for i in range(21)]
        documents.extend(Document(f"doc-other-{i:02}", "box", day, "квитанция") for i in range(45))
        documents.extend([
            Document("doc-a", "box", day, "снятие"),
            Document("doc-other-date", "box", "2028-09-12", "снятие"),
            Document("doc-future", "box", "2028-09-15", "выгрузка"),
            Document("doc-different-box", "other", day, "снятие"),
        ])
        period = PeriodRanker({doc.doc_id: doc for doc in documents})
        words = {"выгрузка": (1., 0.), "снятие": (.9, .43), "квитанция": (0., 1.)}
        pool = SemanticCandidatePool(period, SemanticBM25(period.inner, documents, words, .45))
        as_of = "2028-09-14"
        query = parse_query("выгрузка 13.09.2028", as_of)
        plain = DatedSearcher.search_parsed(period, query, "box", as_of, 100)
        self.assertNotIn("doc-a", {doc_id for doc_id, _ in plain[:30]})
        selected, tail = pool._candidates_parsed(query, "box", as_of, 100)
        ids = {doc_id for doc_id, _ in selected}
        self.assertEqual(len(ids), 30)
        self.assertIn("doc-a", ids)
        self.assertNotIn("doc-other-date", ids)
        self.assertEqual(selected, [hit for hit in plain if hit[0] in ids])
        self.assertEqual({doc_id for doc_id, _ in selected + tail}, {doc_id for doc_id, _ in plain})
        self.assertEqual(len(selected) + len(tail), len(plain))
