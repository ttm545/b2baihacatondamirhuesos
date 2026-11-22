from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from solutions.tiny_encoder.qwen_context import Candidates, ContextModel
from solutions.tiny_encoder.routed import RoutedSearch


class RoutedSearchTest(TestCase):
    def test_date_uses_period_and_preserves_parameters(self):
        period, vector = Mock(), Mock()
        period.search.return_value = [("dated", 1.0)]
        searcher = RoutedSearch(period, vector, fuse_undated=True)
        args = ("Покажи чеки за вчера", "box", "2026-11-22", 5)
        self.assertEqual(searcher.search(*args), [("dated", 1.0)])
        period.search.assert_called_once_with(*args)
        vector.search.assert_not_called()

    def test_undated_auto_uses_vector(self):
        period, vector = Mock(), Mock()
        vector.search.return_value = [("semantic", 0.8)]
        searcher = RoutedSearch(period, vector)
        args = ("Отклонённые платежи", "box", "2026-11-22", 5)
        self.assertEqual(searcher.search(*args), [("semantic", 0.8)])
        vector.search.assert_called_once_with(*args)
        period.search.assert_not_called()

    def test_undated_hybrid_returns_unique_ids_and_favors_agreement(self):
        period, vector = Mock(), Mock()
        period.search.return_value = [("lexical", 4.0), ("both", 3.0)]
        vector.search.return_value = [("semantic", 0.9), ("both", 0.8)]
        searcher = RoutedSearch(period, vector, fuse_undated=True)
        hits = searcher.search("Отклонённые платежи", "box", "2026-11-22", 10)
        self.assertEqual([doc_id for doc_id, _ in hits], ["both", "lexical", "semantic"])

    def test_bounded_context_preserves_membership_and_order_after_depth(self):
        candidates = Candidates(["a", "b", "c", "d"], [("tail", 0.0)],
                                np.array([1., .9, .7, .8]), np.zeros(4),
                                np.array([[0., 2., 0., 0.], [0., 3., 0., 0.],
                                          [0., 4., 0., 0.], [0., 1., 0., 0.]]), 0.0)
        model = ContextModel(np.zeros((2, 2)), np.zeros((2, 2)), np.ones(4), 1., 1.,
                             strategy="rrf", depth=2)
        hits = candidates.ranked(model, 5)
        self.assertEqual([doc_id for doc_id, _ in hits], ["b", "a", "d", "c", "tail"])
