import tempfile
from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None

from vk_mail_search.data import Document


class FakeEncoder:
    signature = "test-encoder"
    max_tokens = 80
    dimension = 2
    batches_seen = 0

    def batches(self, texts, batch_size):
        self.batches_seen += 1
        rows = [[1.0, 0.0] if "чек" in text else [0.0, 1.0] for text in texts]
        yield list(range(len(texts))), np.asarray(rows, dtype=np.float32)

    def query(self, text):
        return np.asarray([1., 0.], dtype=np.float32)


@unittest.skipIf(np is None or torch is None, "install the optional tiny encoder dependencies")
class TinyIndexTest(unittest.TestCase):
    def setUp(self):
        from solutions.tiny_encoder.index import TinySearch

        scratch = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(scratch) / "test.npz"
        docs = [Document("target", "a", "2028-01-03", "Тема: чек"),
                Document("other-topic", "a", "2028-01-03", "Тема: отгрузка"),
                Document("older", "a", "2028-01-02", "Тема: чек"),
                Document("undated", "a", None, "Тема: чек"),
                Document("future", "a", "2028-01-06", "Тема: чек"),
                Document("foreign", "b", "2028-01-03", "Тема: чек")]
        self.docs = {doc.doc_id: doc for doc in docs}
        self.encoder = FakeEncoder()
        self.engine = TinySearch(self.docs, self.encoder, self.path)

    def test_period_is_first_and_unavailable_letters_are_excluded(self):
        one = self.engine.search("чек за 3 января 2028", "a", "2028-01-04", 1)
        hits = self.engine.search("чек за 3 января 2028", "a", "2028-01-04", 100)
        self.assertEqual(one, hits[:1])
        self.assertEqual([doc_id for doc_id, _ in hits], ["target", "other-topic", "older", "undated"])
        self.assertEqual(self.engine.search("чек", "missing", None, 5), [])

    def test_index_cache_preserves_order_and_rejects_changed_content(self):
        from solutions.tiny_encoder.index import TinySearch

        rebuilt = TinySearch(dict(reversed(list(self.docs.items()))), self.encoder, self.path)
        self.assertTrue(rebuilt.index_info["cached"])
        self.assertEqual(self.encoder.batches_seen, 1)
        self.assertEqual(self.engine.index_info["unique_texts"], 2)
        self.docs["target"] = Document("target", "a", "2028-01-03", "Изменённый текст")
        with self.assertRaisesRegex(ValueError, "does not match"):
            TinySearch(self.docs, self.encoder, self.path)

    def test_metric_adapter_can_run_from_saved_parameters(self):
        from solutions.tiny_encoder.index import TinySearch
        from solutions.tiny_encoder.metric_adapter import MetricAdapter

        adapter = MetricAdapter(2, rank=1)
        with torch.no_grad():
            adapter.query_tower.down.weight.copy_(torch.tensor([[1., 0.]]))
            adapter.query_tower.up.weight.copy_(torch.tensor([[-1.], [1.]]))
        path = self.path.with_name("metric.npz")
        adapter.save(path, self.encoder.signature)
        loaded = MetricAdapter.load(path, self.encoder.signature, 2)
        adapted = TinySearch(self.docs, self.encoder, self.path, metric=loaded)
        self.assertEqual(adapted.search("чек 03.01.2028", "a", "2028-01-04", 1)[0][0], "other-topic")
        with self.assertRaisesRegex(ValueError, "does not match"):
            MetricAdapter.load(path, "different-encoder", 2)

    def test_topic_adapter_breaks_a_dense_tie_without_changing_the_period(self):
        from solutions.tiny_encoder.index import TinySearch
        from solutions.tiny_encoder.topic_adapter import TopicAdapter

        identity = np.eye(2, dtype=np.float32)
        bases = np.array([[1., 1.], [-1., -1.]], dtype=np.float32)
        topic = TopicAdapter(bases, identity, np.zeros(2, dtype=np.float32))
        path = self.path.with_name("topic.npz")
        topic.save(path, self.encoder.signature)
        topic = TopicAdapter.load(path, self.encoder.signature, 2)
        self.encoder.query = lambda _: np.array([1., 1.], dtype=np.float32)
        self.assertEqual(self.engine.search("чек 03.01.2028", "a", "2028-01-04", 1)[0][0], "other-topic")
        ranked = TinySearch(self.docs, self.encoder, self.path, topic=topic)
        ids = [doc_id for doc_id, _ in ranked.search("чек 03.01.2028", "a", "2028-01-04", 100)]
        self.assertEqual(ids, ["target", "other-topic", "older", "undated"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            TopicAdapter.load(path, "different-encoder", 2)


if __name__ == "__main__":
    unittest.main()
