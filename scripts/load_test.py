"""Measure HTTP search latency under fixed concurrency using the VK query JSONL."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection, HTTPException, HTTPSConnection
import json
import math
from pathlib import Path
from threading import Barrier
from time import perf_counter_ns
from urllib.parse import urlencode, urlsplit


def percentiles(values):
    ordered = sorted(values)
    return {
        f"p{p}": round(ordered[math.ceil(len(ordered) * p / 100) - 1], 3)
        for p in (50, 95, 99)
    } if ordered else None


def read_cases(path, k):
    with path.open(encoding="utf-8-sig") as source:
        rows = (json.loads(line) for line in source if line.strip())
        return [(row["query_id"], urlencode({"q": row["query"], "mailbox": row["synthetic_user"],
                                              "as_of": row["query_date"], "k": k})) for row in rows]


def read_queries(path, k):
    return [target for _, target in read_cases(path, k)]


def attempt(connection, target):
    start = perf_counter_ns()
    try:
        connection.request("GET", target, headers={"Accept": "application/json"})
        response = connection.getresponse()
        body = response.read()
        client_ms = (perf_counter_ns() - start) / 1_000_000
        if response.status != 200:
            return None, client_ms, f"HTTP {response.status}"
        payload = json.loads(body)
        server_ms = float(payload["took_ms"])
        if not isinstance(payload.get("results"), list) or not math.isfinite(server_ms) or server_ms < 0:
            return None, client_ms, "invalid response schema"
        return server_ms, client_ms, None
    except (HTTPException, OSError, ValueError, KeyError, TypeError) as exc:
        connection.close()
        return None, (perf_counter_ns() - start) / 1_000_000, type(exc).__name__


def run(base_url, queries, calls, clients, warmup, timeout, *, query_ids=None):
    url = urlsplit(base_url)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise ValueError("Pass a plain HTTP(S) base URL without credentials, query or fragment")
    endpoint = url.path.rstrip("/") + "/api/search"
    connection_type = HTTPSConnection if url.scheme == "https" else HTTPConnection
    starts = []
    ready = Barrier(clients, action=lambda: starts.append(perf_counter_ns()))

    def worker(worker_id):
        connection = connection_type(url.hostname, url.port, timeout=timeout)
        try:
            warmup_errors = Counter()
            for offset in range(warmup):
                _, _, error = attempt(connection, endpoint + "?" + queries[(worker_id + offset) % len(queries)])
                if error:
                    warmup_errors[error] += 1
            ready.wait()
            records = [(index, attempt(connection, endpoint + "?" + queries[index % len(queries)]))
                       for index in range(worker_id, calls, clients)]
            return records, warmup_errors
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=clients) as pool:
        outputs = list(pool.map(worker, range(clients)))
    elapsed = (perf_counter_ns() - starts[0]) / 1_000_000_000
    records = [record for rows, _ in outputs for record in rows]
    warmup_errors = Counter()
    for _, errors in outputs:
        warmup_errors.update(errors)
    errors = Counter(error for _, (_, _, error) in records if error)
    success = [(server, client) for _, (server, client, error) in records if not error]
    report = {
        "url": base_url, "clients": clients, "unique_queries": len(queries), "calls": len(records),
        "warmup_per_client": warmup, "ok": len(success), "errors": dict(errors),
        "warmup_errors": dict(warmup_errors), "elapsed_s": round(elapsed, 3),
        "requests_per_s": round(calls / elapsed, 2),
        "took_ms": percentiles([server for server, _ in success]),
        "client_ms": percentiles([client for _, client in success]),
    }
    if query_ids is not None:
        longest = sorted(records, key=lambda row: (-row[1][1], row[0]))[:10]
        report["slowest"] = [
            {"query_id": query_ids[index % len(query_ids)], "call": index + 1,
             "client_ms": round(client, 3), "took_ms": round(server, 3) if server is not None else None,
             "error": error}
            for index, (server, client, error) in longest
        ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="base URL of the authorized API deployment")
    parser.add_argument("--queries", required=True, type=Path, help="path to queries_dev.jsonl")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup-per-client", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--out", type=Path, help="write report JSON to this file")
    parser.add_argument("--slowest", action="store_true", help="include IDs of the ten slowest calls")
    args = parser.parse_args()
    if not (1 <= args.clients <= args.calls and 1 <= args.k <= 100
            and args.warmup_per_client >= 0 and math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("invalid client count, request count, k, warmup or timeout")
    try:
        cases = read_cases(args.queries, args.k)
        queries = [target for _, target in cases]
        if not queries:
            raise ValueError("benchmark query file is empty")
        query_ids = [key for key, _ in cases] if args.slowest else None
        report = run(args.url, queries, args.calls, args.clients, args.warmup_per_client, args.timeout,
                     query_ids=query_ids)
        result = json.dumps(report, indent=2) + "\n"
        if args.out:
            if args.out.resolve() == args.queries.resolve():
                raise ValueError("Output must differ from the query file")
            args.out.write_text(result, encoding="utf-8")
        else:
            print(result, end="")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{exc}\n")
    return 1 if report["errors"] or report["warmup_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
