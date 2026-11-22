import importlib.util
import unittest

from vk_mail_search.data import Document
from vk_mail_search.type_groups import GroupConfig, augment, build_groups, parse_config, words


def letter(doc_id, subject, body, box="box", day="2028-09-13"):
    return Document(doc_id, box, day, f"Тема: {subject}\n\n{body}")


class WordsTest(unittest.TestCase):
    def test_template_drops_codes_names_numbers_and_months(self):
        text = "Рожков Нифонт Харламович прибытия: контейнерSYN-0f7d646177, дата 19 февраля 2028, ivan@romashka.ru"
        self.assertEqual(words(text), ["прибытия", "контейнер", "дата"])

    def test_pronouns_are_not_type_words(self):
        self.assertEqual(words("Вам выставлен новый счёт для оплаты"), ["выставлен", "новый", "счет", "оплаты"])


class GroupsTest(unittest.TestCase):
    def setUp(self):
        declined = [letter(f"d{i}", f"Операция отклонена SYN-{i}a", "Недостаточно средств, insufficient funds.")
                    for i in range(3)]
        declined.append(letter("d3", "Операция отклонена SYN-3a", "Платёж не прошёл."))
        paid = [letter(f"p{i}", f"Успешная оплата SYN-{i}b", "Спасибо, платёж прошёл.") for i in range(4)]
        other = [letter("x", "Привет", "Как дела?"), letter("y", "Счёт", "Счёт во вложении", box="other")]
        self.docs = {doc.doc_id: doc for doc in declined + paid + other}
        self.groups = build_groups(self.docs.values(), GroupConfig(min_lift=1.5))

    def test_letters_of_one_template_form_one_group_per_mailbox(self):
        keys = {(group.mailbox, group.key): set(group.doc_ids) for group in self.groups}
        self.assertEqual(keys[("box", "операция отклонена")], {"d0", "d1", "d2", "d3"})
        self.assertEqual(keys[("box", "успешная оплата")], {"p0", "p1", "p2", "p3"})
        self.assertNotIn("y", keys[("box", "успешная оплата")])

    def test_characteristic_words_reach_letters_that_lack_them(self):
        declined = next(group for group in self.groups if group.key == "операция отклонена")
        self.assertIn("insufficient", declined.terms)
        augmented = augment(self.docs, self.groups)
        self.assertIn("insufficient", augmented["d3"].doc)
        self.assertNotIn("insufficient", augmented["p0"].doc)
        self.assertEqual(augmented["d0"].doc.count("insufficient"), 1)  # not repeated where present

    def test_ids_dates_and_mailboxes_are_unchanged(self):
        augmented = augment(self.docs, self.groups)
        for doc_id, doc in self.docs.items():
            self.assertEqual((augmented[doc_id].doc_id, augmented[doc_id].synthetic_user, augmented[doc_id].doc_dt),
                             (doc.doc_id, doc.synthetic_user, doc.doc_dt))
            self.assertTrue(augmented[doc_id].doc.startswith(doc.doc))

    def test_small_groups_get_no_words(self):
        small = [group for group in self.groups if len(group.doc_ids) < 3]
        self.assertTrue(small)
        self.assertTrue(all(not group.terms for group in small))

    def test_config_from_text(self):
        self.assertEqual(parse_config("min_share=0.5,max_terms=6"), GroupConfig(min_share=0.5, max_terms=6))
        with self.assertRaises(ValueError):
            parse_config("unknown=1")


@unittest.skipUnless(importlib.util.find_spec("pymorphy3"), "нужен pymorphy3 из requirements-baseline.txt")
class GroupedPeriodTest(unittest.TestCase):
    def test_grouped_period_keeps_date_and_mailbox_boundaries(self):
        from vk_mail_search.type_groups import build_grouped

        docs = [letter(f"d{i}", f"Операция отклонена SYN-{i}a", "insufficient funds") for i in range(3)]
        docs += [letter("d3", "Операция отклонена SYN-3a", "Платёж не прошёл."),
                 letter("p0", "Успешная оплата SYN-0b", "Платёж прошёл."),
                 letter("late", "Операция отклонена SYN-9a", "insufficient funds", day="2028-09-20"),
                 letter("foreign", "Операция отклонена SYN-8a", "insufficient funds", box="other")]
        engine = build_grouped({doc.doc_id: doc for doc in docs}, config=GroupConfig(min_lift=1.0))
        hits = [doc_id for doc_id, _ in engine.search("insufficient funds 13.09.2028", "box", "2028-09-14", 10)]
        self.assertNotIn("late", hits)
        self.assertNotIn("foreign", hits)
        self.assertIn("d3", hits[:4])


if __name__ == "__main__":
    unittest.main()
