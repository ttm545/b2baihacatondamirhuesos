"""Build training, validation and held-out sets from the prepared public mail corpora."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
import re
import tarfile

from vk_mail_search.data import Dataset, Document, Query, jsonl, load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark
from vk_mail_search.query_parser import parse_query


@dataclass(frozen=True)
class Corpus:
    name: str
    data: Dataset
    train: list[str]
    validation: list[str]
    test: list[str]
    protocol: str


def _key(value: str):
    return sha256(f"260621:{value}".encode()).digest()


def _groups(data: Dataset, groups: dict[str, list[str]], train_n: int, valid_n: int):
    ordered = sorted(groups, key=_key)
    parts = [ordered[:train_n], ordered[train_n:train_n + valid_n], ordered[train_n + valid_n:]]
    return tuple([qid for group in part for qid in groups[group] if qid in data.queries] for part in parts)


def vk_corpus(directory: Path, paraphrases: Path, *, expanded: bool = False) -> Corpus:
    public = load_dataset(directory)
    reworded, splits = _load_paraphrase_benchmark(public, paraphrases)
    data = Dataset(public.documents, {**reworded.queries}, {**reworded.qrels})
    holds = sorted({qid.rsplit("-", 1)[0] for qid in splits["tune"]}, key=_key)[:2]
    train, dev = [], []
    for qid in splits["tune"]:
        target = dev if qid.rsplit("-", 1)[0] in holds else train
        target.append(qid)
    added = set()
    topic_part = {}
    for _, row in jsonl(paraphrases):
        source_id = row["source_query_id"]
        source = public.queries[source_id]
        topic = parse_query(source.query, source.query_date).topic
        part = "test" if row["split"] == "test" else "dev" if row["query_id"].rsplit("-", 1)[0] in holds else "train"
        if topic_part.setdefault(topic, part) != part:
            raise ValueError("One VK topic template crosses the train/validation/test family boundaries")
        if row["split"] != "tune" or source_id in added:
            continue
        added.add(source_id)
        data.queries[source_id] = public.queries[source_id]
        data.qrels[source_id] = public.qrels[source_id]
        (dev if row["query_id"].rsplit("-", 1)[0] in holds else train).append(source_id)
    if expanded:
        for qid, query in public.queries.items():
            part = topic_part.get(parse_query(query.query, query.query_date).topic)
            if part not in {"train", "dev"} or qid in added:
                continue
            data.queries[qid] = query
            data.qrels[qid] = public.qrels[qid]
            (dev if part == "dev" else train).append(qid)
    names = {False: ("vk", "Families from repo tune/test; original tune queries join their family"),
             True: ("vk-expanded", "All canonical queries of the eight tune templates; four test templates excluded from fitting and selection")}
    name, protocol = names[expanded]
    return Corpus(name, data, train, dev, splits["test"], protocol)


def vk_trusted_negatives(corpus: Corpus, source: Path, engine):
    source_ids = {row["query_id"]: row["source_query_id"] for _, row in jsonl(source)}
    by_doc, by_query = defaultdict(set), {}
    for qid in corpus.train:
        original = corpus.data.queries[source_ids.get(qid, qid)]
        topic = parse_query(original.query, original.query_date).topic
        by_query[qid] = topic
        for did, grade in corpus.data.qrels[qid].items():
            if grade > 0:
                by_doc[did].add(topic)
    by_mailbox = defaultdict(list)
    for index, document in enumerate(engine.documents):
        if document.doc_id in by_doc:
            by_mailbox[document.synthetic_user].append((index, document, by_doc[document.doc_id]))
    return {qid: [index for index, doc, topics in by_mailbox[corpus.data.queries[qid].synthetic_user]
                  if topic not in topics and doc.available(corpus.data.queries[qid].synthetic_user,
                                                            corpus.data.queries[qid].query_date)]
            for qid, topic in by_query.items()}


def sara_corpus(root: Path) -> Corpus:
    data = load_dataset(root / "vk-jsonl")
    groups = defaultdict(list)
    for qid in data.queries:
        groups[str((int(qid) - 1) // 3)].append(qid)
    if len(groups) != 50 or any(len(group) != 3 or any(data.qrels[q] != data.qrels[group[0]] for q in group)
                                for group in groups.values()):
        raise ValueError("SARA topic grouping no longer matches repeated_qrels")
    train, dev, test = _groups(data, groups, 32, 8)
    return Corpus("sara", data, train, dev, test, "32/8/10 human-judged topics, with their three formulations kept together")


def vectrix_corpus(root: Path) -> Corpus:
    from pyarrow.parquet import read_table

    data = load_dataset(root / "vk-jsonl")
    source_test = read_table(root / "tasks_eval.parquet", columns=["inbox_address"]).to_pylist()
    boxes = {mailbox: f"vectrix-{i}" for i, mailbox in enumerate(
        sorted({row["inbox_address"].casefold() for row in source_test}), start=1)}

    test = list(data.queries)
    kept = []
    test_targets = {did.rsplit("-", 1)[-1] for labels in data.qrels.values() for did, grade in labels.items() if grade > 0}
    table = read_table(root / "tasks_train.parquet", columns=["id", "inbox_address", "question", "query_date", "message_ids"])
    for row in table.to_pylist():
        box = boxes.get(row["inbox_address"].casefold())
        if not box:
            continue
        hashes = {sha256(mid.encode()).hexdigest() for mid in row["message_ids"] or [] if mid}
        if not hashes or not hashes.isdisjoint(test_targets):
            continue
        qid = f"vectrix-train-{row['id']}"
        cutoff = date.fromisoformat(str(row["query_date"])[:10]).isoformat()
        query = Query(qid, box, row["question"], cutoff)
        targets = sorted(f"{box}-{key}" for key in hashes)
        if all(doc_id in data.documents and data.documents[doc_id].available(box, cutoff) for doc_id in targets):
            kept.append((query, targets))

    kept.sort(key=lambda item: _key(item[0].query_id))
    validation = kept[:max(10, min(45, len(kept) // 5))]
    val_targets = {did.rsplit("-", 1)[-1] for _, ids in validation for did in ids}
    training = [(query, targets) for query, targets in kept[len(validation):]
                if all(did.rsplit("-", 1)[-1] not in val_targets for did in targets)][:300]
    for query, targets in training + validation:
        data.queries[query.query_id] = query
        data.qrels[query.query_id] = dict.fromkeys(targets, 1)
    return Corpus("vectrix", data, [q.query_id for q, _ in training], [q.query_id for q, _ in validation], test,
                  "Publisher train/eval split; target Message-ID kept disjoint across all three parts")


def arte_corpus(root: Path) -> Corpus:
    import pyarrow as pa
    from pyarrow.parquet import ParquetFile, read_table

    data = load_dataset(root / "vk-jsonl")
    test = list(data.queries)
    test_boxes = {row["inbox_address"].casefold() for row in read_table(root / "questions-0000.parquet",
                   columns=["inbox_address"]).to_pylist()}
    test_hashes = {did.rsplit("-", 1)[-1] for labels in data.qrels.values() for did, grade in labels.items() if grade > 0}
    queries = read_table(root / "questions-train.parquet", columns=["id", "question", "inbox_address", "query_date", "message_ids"])
    by_box = defaultdict(list)
    for row in queries.to_pylist():
        box = row["inbox_address"].casefold()
        if box not in test_boxes and row["message_ids"]:
            by_box[box].append(row)
    chosen = sorted((box for box, rows in by_box.items() if len(rows) >= 300), key=_key)[:4]
    if len(chosen) < 4:
        raise ValueError("ART·E training split has fewer than four eligible mailboxes")
    boxes = {box: f"arte-learn-{i}" for i, box in enumerate(chosen)}
    wanted = {}
    for box in chosen:
        by_box[box] = sorted(by_box[box], key=lambda row: _key(str(row["id"])))[:180]
        wanted[box] = {mid for row in by_box[box] for mid in row["message_ids"] if mid}

    columns = ["message_id", "subject", "from", "to", "cc", "bcc", "date", "body"]
    paths = sorted(root.glob("emails-*.parquet"))
    if not paths:
        raise ValueError("Missing ART·E email Parquet shards")
    selected = defaultdict(dict)
    for path in paths:
        for batch in ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
            dates = batch.column("date").cast(pa.timestamp("us")).to_pylist()
            rows = batch.select([column for column in columns if column != "date"]).to_pylist()
            for row, received in zip(rows, dates, strict=True):
                mid = row["message_id"]
                if not mid or len(row["body"] or "") > 5000:
                    continue
                recipients = [address for name in ("to", "cc", "bcc") for address in row[name] or [] if address]
                if len(recipients) > 30:
                    continue
                members = {address.casefold() for address in recipients + [row["from"] or ""]}
                for box in members & boxes.keys():
                    doc_id = f"{boxes[box]}-{sha256(mid.encode()).hexdigest()}"
                    day = received.date().isoformat() if received else None
                    document = Document(doc_id, boxes[box], day,
                        f"Subject: {row['subject'] or ''}\nFrom: {row['from'] or ''}\n\n{row['body'] or ''}")
                    selected[box].setdefault(mid, document)

    train, validation = [], []
    for i, box in enumerate(chosen):
        all_docs = selected[box]
        sample = sorted((mid for mid in all_docs if mid not in wanted[box]), key=_key)[:2200]
        used = set(sample) | (wanted[box] & all_docs.keys())
        data.documents.update({all_docs[mid].doc_id: all_docs[mid] for mid in used})
        dest, cap = (validation, 45) if i == 0 else (train, 100)
        accepted = 0
        for row in by_box[box]:
            cutoff = date.fromisoformat(str(row["query_date"])[:10]).isoformat()
            qid = f"arte-learn-{row['id']}"
            ids = {mid for mid in row["message_ids"] if mid}
            if not ids or not ids <= used:
                continue
            targets = [all_docs[mid] for mid in ids]
            if any(item.doc_id.rsplit("-", 1)[-1] in test_hashes or not item.available(boxes[box], cutoff) for item in targets):
                continue
            data.queries[qid] = Query(qid, boxes[box], row["question"], cutoff)
            data.qrels[qid] = {item.doc_id: 1 for item in targets}
            dest.append(qid)
            accepted += 1
            if accepted == cap:
                break
    return Corpus("arte", data, train, validation, test, "Official Train, three mailboxes for fitting and one for validation; separate official Test mailbox (60 prepared questions)")


def w3c_corpus(root: Path) -> Corpus:
    data = load_dataset(root / "vk-jsonl")
    test = list(data.queries)
    labels = defaultdict(set)
    for line in (root / "ent05.known-item.training-qrels").read_text(encoding="ascii").splitlines():
        qid, _, did, grade = line.split()
        if int(grade) > 0:
            labels[qid].add(did)
    wanted = set().union(*labels.values()) - data.documents.keys()
    header = re.compile(r'^([a-z]+)="(.*)"\s*$', re.MULTILINE)
    for archive in sorted(root.glob("w3c-emails-part*.tar.gz")):
        if not wanted:
            break
        with tarfile.open(archive, "r|gz") as tar:
            for entry in tar:
                did = Path(entry.name).stem
                if not entry.isfile() or did not in wanted:
                    continue
                with tar.extractfile(entry) as file:
                    raw = file.read().decode("utf-8", errors="replace").replace("\r\n", "\n")
                fields, _, body = raw.partition("\n\n")
                fields = dict(header.findall(fields))
                if fields.get("docno") != did:
                    continue
                day = (fields.get("isoreceived") or fields.get("isosent") or "")[:8]
                try:
                    dt = date.fromisoformat(day).isoformat()
                except ValueError:
                    dt = None
                data.documents[did] = Document(did, "trec-w3c", dt,
                    f"Subject: {fields.get('subject', '')}\nFrom: {fields.get('email', '')}\n\n{body.lstrip()}")
                wanted.remove(did)
    topics = dict(re.findall(r"<num>\s*Number:\s*(KI\d+)\s*</num>\s*<title>\s*(.*?)\s*</title>",
                             (root / "ent05.known-item.topics").read_text(encoding="latin-1"), re.DOTALL))
    train_ids = []
    for qid, targets in labels.items():
        if qid in topics and targets <= data.documents.keys() and all(data.documents[did].available("trec-w3c", "2004-07-01") for did in targets):
            data.queries[qid] = Query(qid, "trec-w3c", " ".join(topics[qid].split()), "2004-07-01")
            data.qrels[qid] = dict.fromkeys(targets, 1)
            train_ids.append(qid)
    order = sorted(train_ids, key=_key)
    return Corpus("w3c", data, order[5:], order[:5], test, "Official 25 training topics split 20/5; all 125 test topics on sampled W3C pool")
