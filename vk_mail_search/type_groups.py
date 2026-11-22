"""Corpus-derived mail type groups: share a group's characteristic words with all its letters.

Nothing here is written by hand for a particular archive. At start-up the letters of each
mailbox are grouped by the template of their subject: codes, e-mails, numbers, full names and
months are dropped, and the first few words that recur in the mailbox form the group key.
Words that most letters of a group carry in the subject or opening text, and that are much
rarer in the rest of the mailbox, become the group's characteristic words. Each letter gets the
characteristic words it lacks appended once to its body, so a query word written in some letters
of a type also finds the letters of that type that phrase it differently.

The ranking itself stays ``period``: dates, mailboxes and the reranker are unchanged.

    python -m vk_mail_search.type_groups stats --data-dir data
    python -m vk_mail_search.type_groups compare --data-dir data --paraphrases tests/paraphrases.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

from .data import HEADER, SENDER, Document

_CODE = re.compile(r"SYN-[A-Za-z0-9-]+|[\w.+%-]+@[\w.-]+|https?://\S+")
_NAME = re.compile(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}\b")
_WORD = re.compile(r"[a-zа-я]{3,}")
_MONTH = re.compile(r"(?:январ|феврал|март|апрел|ма[йяе]|июн|июл|август|сентябр|октябр|ноябр|декабр)")
# General Russian pronouns and particles, not a list for this archive: they carry no mail type.
_STOP = frozenset("""
вам вас ваш ваша ваше ваши вашего вашей вашу нам нас наш наша наше наши мне меня мой моя мое мои
его ее они оно она них ним ней для при про без или что это эта этот эти как так все всё уже еще
тоже также был была было были будет есть нет вот там тут где кто чем чтобы если когда
""".split())


def words(text: str) -> list[str]:
    """Lower-case words of a template: no codes, e-mails, numbers, full names or months."""
    text = _NAME.sub(" ", _CODE.sub(" ", text))
    return [word for word in _WORD.findall(text.casefold().replace("ё", "е")) if not _MONTH.match(word) and word not in _STOP]


def opening(doc: Document, chars: int = 400) -> str:
    """Subject and the first characters of the body without header lines."""
    body = HEADER.sub(" ", doc.doc)
    body = "\n".join(line for line in body.splitlines() if not SENDER.match(line))
    return doc.subject + "\n" + body.strip()[:chars]


@dataclass(frozen=True, slots=True)
class GroupConfig:
    key_words: int = 3       # recurring subject words that form the group key
    key_min_df: int = 3      # a key word occurs in at least this many letters of the mailbox
    min_size: int = 3        # smaller groups get no characteristic words
    min_share: float = 0.4   # a characteristic word is in at least this share of the group
    min_lift: float = 2.0    # ...and this many times more frequent than in the whole mailbox
    max_terms: int = 8

    def __post_init__(self) -> None:
        if (self.key_words < 1 or self.key_min_df < 1 or self.min_size < 2 or not 0 < self.min_share <= 1
                or self.min_lift < 1 or self.max_terms < 0):
            raise ValueError("Некорректные параметры групп.")


@dataclass(frozen=True, slots=True)
class Group:
    mailbox: str
    key: str
    doc_ids: tuple[str, ...]
    terms: tuple[str, ...]


def build_groups(documents: Iterable[Document], config: GroupConfig = GroupConfig()) -> list[Group]:
    by_box: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        by_box[doc.synthetic_user].append(doc)
    groups = []
    for mailbox, docs in sorted(by_box.items()):
        subjects = {doc.doc_id: list(dict.fromkeys(words(doc.subject))) for doc in docs}
        subject_df = Counter(word for terms in subjects.values() for word in terms)
        members: dict[str, list[str]] = defaultdict(list)
        for doc in docs:
            key = [word for word in subjects[doc.doc_id] if subject_df[word] >= config.key_min_df]
            members[" ".join(key[:config.key_words])].append(doc.doc_id)
        texts = {doc.doc_id: set(words(opening(doc))) for doc in docs}
        box_df = Counter(word for terms in texts.values() for word in terms)
        for key, ids in sorted(members.items()):
            terms: tuple[str, ...] = ()
            if key and len(ids) >= config.min_size:
                group_df = Counter(word for doc_id in ids for word in texts[doc_id])
                scored = []
                for word, count in group_df.items():
                    share = count / len(ids)
                    lift = share / (box_df[word] / len(docs))
                    if share >= config.min_share and lift >= config.min_lift:
                        scored.append((-share * math.log(lift), word))
                terms = tuple(word for _, word in sorted(scored)[:config.max_terms])
            groups.append(Group(mailbox, key, tuple(ids), terms))
    return groups


def augment(documents: Mapping[str, Document], groups: Iterable[Group]) -> dict[str, Document]:
    """Copies of the letters with the group words each letter lacks appended to its body."""
    extra: dict[str, tuple[str, ...]] = {}
    for group in groups:
        for doc_id in group.doc_ids:
            if group.terms:
                present = set(words(documents[doc_id].doc))
                missing = tuple(word for word in group.terms if word not in present)
                if missing:
                    extra[doc_id] = missing
    return {doc_id: replace(doc, doc=doc.doc + "\n\n" + " ".join(extra[doc_id])) if doc_id in extra else doc
            for doc_id, doc in documents.items()}


def build_grouped(documents: Mapping[str, Document], *, mode: str = "learned",
                  config: GroupConfig = GroupConfig()):
    """``period`` over the augmented letters. Doc ids, dates and mailboxes are unchanged."""
    from .period_rank import PeriodRanker

    return PeriodRanker(augment(documents, build_groups(documents.values(), config)), mode=mode)


def parse_config(spec: str) -> GroupConfig:
    """``min_share=0.5,max_terms=6`` → GroupConfig."""
    values = {}
    fields = GroupConfig.__dataclass_fields__
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, _, raw = item.partition("=")
        if name not in fields or not raw:
            raise ValueError(f"Неизвестный параметр групп: {item}")
        values[name] = float(raw) if name in ("min_share", "min_lift") else int(raw)
    return GroupConfig(**values)


# --- reports ---------------------------------------------------------------------------------

MAIN = (("Recall@5", "@5", "recall"), ("nDCG@5", "@5", "ndcg"), ("MRR@5", "@5", "mrr"),
        ("nDCG@10", "@10", "ndcg"), ("Recall@100", "@100", "recall"))


def _stats(args) -> None:
    from .data import load_documents

    docs = load_documents(args.data_dir)
    config = parse_config(args.groups)
    start = perf_counter()
    groups = build_groups(docs.values(), config)
    seconds = perf_counter() - start
    boxes: dict[str, list[Group]] = defaultdict(list)
    for group in groups:
        boxes[group.mailbox].append(group)
    rows = []
    for mailbox, items in sorted(boxes.items()):
        useful = [group for group in items if group.terms]
        rows.append(dict(mailbox=mailbox, letters=sum(len(g.doc_ids) for g in items), groups=len(useful),
                         letters_in_groups=sum(len(g.doc_ids) for g in useful)))
    examples = sorted((g for g in groups if g.terms), key=lambda g: -len(g.doc_ids))
    picked, seen = [], set()
    for group in examples:  # the largest group of different mailboxes
        if group.mailbox not in seen:
            picked.append(dict(mailbox=group.mailbox, key=group.key, letters=len(group.doc_ids),
                               terms=list(group.terms)))
            seen.add(group.mailbox)
        if len(picked) == 5:
            break
    augmented = augment(docs, groups)
    changed = sum(augmented[doc_id] is not doc for doc_id, doc in docs.items())
    print(json.dumps(dict(config=asdict(config),
                          grouping_seconds=round(seconds, 3), letters_changed=changed,
                          mailboxes=rows, examples=picked), ensure_ascii=False, indent=2, ))


def _evaluate(dataset, searcher, warmup: int):
    from .evaluate import run_search, score_predictions

    predictions, timing = run_search(dataset, searcher, 100, 10, warmup)
    report = score_predictions(dataset, predictions, (5, 10, 100))
    metrics = {label: report["metrics"][cut][name] for label, cut, name in MAIN}
    return metrics, timing["latency_ms"]["p99"]


def _compare(args) -> None:
    from .data import load_dataset
    from .evaluate import _load_paraphrase_benchmark, _select_queries
    from .period_rank import PeriodRanker

    public = load_dataset(args.data_dir)
    paraphrased, splits = _load_paraphrase_benchmark(public, args.paraphrases)
    sets = {"public": public, **{name: _select_queries(paraphrased, ids) for name, ids in splits.items()}}
    start = perf_counter()
    base = PeriodRanker(public.documents, mode=args.mode)
    base_seconds = perf_counter() - start

    candidates = [parse_config(spec) for spec in args.grid] if args.grid else [parse_config(args.groups)]
    # Weights and thresholds are chosen on tune only; test and public are reported once.
    tuned = []
    for config in candidates:
        engine = build_grouped(public.documents, mode=args.mode, config=config)
        metrics, _ = _evaluate(sets["tune"], engine, args.warmup)
        tuned.append((metrics["nDCG@5"], metrics["Recall@5"], config))
        print(f"tune {config}: nDCG@5 {metrics['nDCG@5']:.4f} Recall@5 {metrics['Recall@5']:.4f}",
              file=sys.stderr)
    best = max(tuned, key=lambda row: row[:2])[2]
    start = perf_counter()
    grouped = build_grouped(public.documents, mode=args.mode, config=best)
    grouped_seconds = perf_counter() - start

    table = {}
    for name, dataset in sets.items():
        for label, engine in (("period", base), ("period+groups", grouped)):
            metrics, p99 = _evaluate(dataset, engine, args.warmup)
            table[f"{name} · {label}"] = {**metrics, "p99_ms": p99, "queries": len(dataset.queries)}
    report = dict(chosen_on="tune", config=repr(best), index_seconds={"period": round(base_seconds, 2),
                  "period+groups": round(grouped_seconds, 2)}, results=table)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n| Набор · поисковик | " + " | ".join(label for label, _, _ in MAIN) + " | p99, мс |")
    print("| --- |" + " ---: |" * (len(MAIN) + 1))
    for name, row in table.items():
        print(f"| {name} | " + " | ".join(f"{row[label]:.4f}" for label, _, _ in MAIN) + f" | {row['p99_ms']} |")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("stats", "compare"))
    parser.add_argument("--data-dir")
    parser.add_argument("--groups", default="", help="параметры групп, например min_share=0.5,max_terms=6")
    parser.add_argument("--grid", nargs="*", help="несколько наборов параметров; выбор только по tune")
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    parser.add_argument("--mode", default="learned", help="профиль period")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)
    (_stats if args.command == "stats" else _compare)(args)


if __name__ == "__main__":
    main()
