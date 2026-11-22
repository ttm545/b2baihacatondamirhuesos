from datetime import date
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from solutions.tiny_encoder.ensemble import EnsembleOptions, EnsembleSearch
from solutions.tiny_encoder.qwen_context import ContextModel, QwenContextSearch
from solutions.tiny_encoder.semantic_bm25 import FeedbackProfile, SemanticBM25
from vk_mail_search.data import Document
from vk_mail_search.period_rank import PeriodRanker
from vk_mail_search.query_parser import ParsedQuery, Period


class SemanticBM25Test(TestCase):
    def test_feedback_replaces_lexical_candidates_inside_the_ensemble(self):
        docs = [
            Document("a", "box", "2026-11-21", "Тема: Грузовик\nТягач прибыл."),
            Document("b", "box", "2026-11-21", "Тема: Грузовик\nТягач припаркован."),
            Document("z", "box", "2026-11-21", "Тема: Архив\nСохранена копия."),
            Document("other_date", "box", "2026-11-20", "Тема: Грузовик\nТягач на площадке."),
            Document("future", "box", "2026-11-23", "Тема: Грузовик\nТягач на площадке."),
            Document("other_box", "separate", "2026-11-21", "Тема: Грузовик\nТягач на площадке."),
        ]
        period = PeriodRanker({doc.doc_id: doc for doc in docs}, depth=2)
        query = Mock(return_value=np.ones(2, dtype=np.float32))
        tower = SimpleNamespace(documents=docs, adapter=None, vectors=np.ones((6, 2), dtype=np.float32),
                                encoder=SimpleNamespace(query=query),
                                eligible=lambda mailbox, as_of: np.arange(4))
        profile = FeedbackProfile(4, 4, .9, context_gate=.7)
        engine = SemanticBM25(period, tower, profile, profile)
        day = date(2026, 11, 21)
        parsed = ParsedQuery("фура", "фура", Period(day, day))
        seen = engine.candidates(parsed, "box", "2026-11-22", profile,
                                 query_vector=np.ones(2, dtype=np.float32))
        self.assertEqual(seen.original, {})
        self.assertIn("грузовик", {term for term, _ in seen.expansion})
        self.assertEqual(seen.rank(profile.weight)[-1], ("z", 0.))
        projection = np.eye(3, 2, dtype=np.float32)
        model = ContextModel(projection, projection, np.ones(4), 1., .9)
        context = QwenContextSearch(period, tower, model)
        options = EnsembleOptions((1., 1., 1.), 2, .9, (1., 1., 1., 1.))
        original = EnsembleSearch(context, options)
        ensemble = EnsembleSearch(context, options, semantic_bm25=engine)

        baseline = original.dated_candidates(parsed, "box", "2026-11-22", 2)
        self.assertIn("z", baseline.ids)
        dated = ensemble.dated_candidates(parsed, "box", "2026-11-22", 2)
        self.assertEqual(set(dated.ids), {"a", "b"})
        query.reset_mock()
        undated = ensemble.content_candidates("фура", "box", "2026-11-22")
        evidence = dict(zip(undated.ids, undated.features[:, 0], strict=True))
        self.assertGreater(evidence["a"], 0.)
        self.assertEqual(evidence["z"], 0.)
        query.assert_called_once_with("фура")
