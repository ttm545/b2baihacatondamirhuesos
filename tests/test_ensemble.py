from datetime import timedelta, date
from unittest import TestCase

import numpy as np

from solutions.tiny_encoder.ensemble import ContentCandidates, DatedCandidates
from solutions.tiny_encoder.train_ensemble import masked_query
from vk_mail_search.data import Query


class EnsembleTest(TestCase):
    def test_dated_depth_preserves_membership_and_tail(self):
        rows = DatedCandidates(["a", "b", "c"], np.arange(3),
                               np.array([[.9, 0., 0.], [.8, .1, 0.], [.7, 1., 0.]]),
                               [("other_day", 0.)], .2)
        bounded = rows.rank((0., 1., 0.), 2, .5, 4)
        self.assertEqual([cid for cid, _ in bounded], ["b", "a", "c", "other_day"])
        self.assertEqual([cid for cid, _ in rows.rank((0., 1., 0.), 2, .1, 4)],
                         ["a", "b", "c", "other_day"])

    def test_no_date_control_uses_changed_availability_anchor(self):
        q = Query("id", "box", "Покажи чеки за вчера", "2026-11-21")
        altered = masked_query(q, 30)
        self.assertEqual(altered.synthetic_user, q.synthetic_user)
        self.assertEqual(altered.query_id, q.query_id)
        self.assertEqual(date.fromisoformat(altered.query_date), date.fromisoformat(q.query_date) + timedelta(days=30))
        self.assertNotIn("вчера", altered.query)

    def test_content_routes_require_a_nonzero_weighted_signal(self):
        rows = ContentCandidates(["lexical", "vector", "fresh"],
                                 np.array([[.9, 0., 0., 0.], [0., .8, 0., 0.], [0., 0., .7, 1.]]))
        self.assertEqual([cid for cid, _ in rows.rank((.5, .5, 0., 0.), 3)], ["lexical", "vector"])
