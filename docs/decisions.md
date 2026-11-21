# Технические решения

Координатор поиска — Степан (Claude). Изменения контракта ниже согласуйте в чате до реализации.

## D1. Архитектура: всё в памяти, без OpenSearch

Корпус ~16 тыс. писем, и поиск всегда идёт внутри одного ящика (~1,3 тыс. писем). Векторы и BM25 держим в памяти процесса: numpy-матрица и свой BM25. Перебор по ящику занимает миллисекунды, а отдельный сервис поиска добавил бы сетевые задержки, настройку и точку отказа. HNSW при таком размере не нужен и только снижает точность.

Стек: Python 3.11+, FastAPI, numpy, sentence-transformers (`deepvk/USER2-base` из `vk_mail_search/embeddings.py`), pymorphy3 для лемм.

## D2. Данные

Архив организаторов в Git не кладём. Путь к распакованному архиву — переменная `VK_DATA_DIR` (каталог с `documents.jsonl`, `queries_dev.jsonl`, `qrels_dev.jsonl`). Индексы и векторы — в `VK_CACHE_DIR` (по умолчанию `.cache/`), тоже вне Git.

Поля: документ — `doc_id`, `synthetic_user` (ящик), `doc_dt` (`ГГГГ-ММ-ДД` или `null`), `doc` (текст письма); запрос — `query_id`, `synthetic_user`, `query`, `query_date`; оценка — `query_id`, `doc_id`, `relevance`.

## D3. Контракт поисковика

Пакет `vk_mail_search/`. Любой поисковик (BM25, векторный, гибрид) реализует:

```python
class Searcher(Protocol):
    def search(self, query: str, mailbox: str, as_of: str | None, k: int) -> list[tuple[str, float]]:
        """Ранжированные (doc_id, score), лучший первым, без повторов, не больше k."""
```

Фильтры обязательны в каждом поисковике: только письма ящика `mailbox`; письма с `doc_dt > as_of` исключаются, с `doc_dt = null` остаются.

Модули: `data.py` (загрузка и модель письма), `evaluate.py` (оценка), `bm25.py`, `dense.py`, `hybrid.py`, `api.py` (FastAPI). Фабрика `build_searcher(name, docs)` в `vk_mail_search/__init__.py` или `searchers.py`.

## D4. Оценка

Одна команда для всех вариантов:

```sh
python -m vk_mail_search.evaluate --searcher bm25 --k 10 --out predictions.jsonl
```

Печатает Recall@K, nDCG@K, MRR@K (формулы как в `search-lab/engine.mjs`, графы `2^rel - 1`) для K = 5, 10, 20 и задержку одного вызова `search` p50/p95/p99 после прогрева. `--pred file.jsonl` оценивает готовую выдачу. Формат выдачи: `{"query_id": "...", "doc_ids": [...]}` по строке на запрос.

Сравниваем варианты только этой командой, на одном наборе и одном K. Результаты записываем в таблицу ниже.

## D5. HTTP API

- `GET /api/search?q=&mailbox=&as_of=&k=10` → `{"took_ms": ..., "results": [{"doc_id", "score", "doc_dt", "subject", "snippet"}]}`. `took_ms` — серверное время от приёма до выдачи.
- `GET /api/mailboxes` → список ящиков с числом писем.
- `GET /api/mailboxes/{mailbox}/docs?offset=&limit=` → письма ящика, новые сверху.
- `GET /api/docs/{doc_id}` → письмо целиком.

Модели и индексы загружаются один раз при старте (`lifespan`), запросы их только читают.

## Результаты на публичном наборе (99 запросов)

| Вариант | Recall@10 | nDCG@10 | MRR@10 | p99 поиска | Коммит |
| --- | --- | --- | --- | --- | --- |
| BM25 + леммы (Дамир) | 0.4807 | 0.3489 | 0.3452 | 6,05 мс, вызов search без HTTP | `096b2c3` |
