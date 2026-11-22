# Умный поиск в Почте — кейс VK, хакатон ФПМИ 2026

Веб-прототип почты с поиском по архиву писем организаторов. Пользователь пишет запрос своими словами — «уведомления об отклонённых платежах, полученные вчера» — и получает письма нужного дня и нужного типа, а не все письма со словом «платёж».

**Версия 1.1** — почта с гибридным поиском по локальному архиву. На первом этапе выбираются до 30 писем, затем выдачу сортирует ансамбль. [Открыть почту](https://vk-mail.597d550b.sslip.io/) · [Архитектура и запуск контейнера](docs/architecture.md)

| Набор | Recall@5 | nDCG@10 |
| --- | ---: | ---: |
| 99 публичных запросов | 0,8516 | 0,9906 |
| 32 отложенные перефразировки | 0,5687 | 0,7180 |

## Как это работает

1. **Разбор запроса** (`vk_mail_search/query_parser.py`). Из запроса выделяются период и тема. Период — точные даты с годом и без, «вчера», «позавчера», «3 дня назад», «на прошлой неделе», «в сентябре», «с 1 по 15 ноября», «15.10», дни недели и выходные; относительные даты считаются от даты запроса. Тема — запрос без «Найди/Покажи», даты и «полученные».
2. **Письма периода — первыми** (`vk_mail_search/dated.py`). Внутри периода письма ранжируются по теме, письма вне периода не выбрасываются, а идут следом: ошибка разбора меняет порядок, но не теряет письмо.
3. **BM25 по теме** (`vk_mail_search/bm25.py`), всегда внутри ящика пользователя и без писем позже даты запроса: леммы pymorphy3, совпадение по первым 5 буквам леммы для однокоренных слов («погрузка» ↔ «погружен»), тема письма весит больше тела, служебные слова почти не весят, бонус за совпадение с темой письма.
4. **Гибридный отбор топ‑30.** Для запроса с датой смешиваются 91,5% обычного BM25 и 8,5% BM25, в котором близость слов определяется векторами Navec. Письма из указанного периода получают приоритет.
5. **Финальная сортировка.** При слабых совпадениях ансамбль объединяет текстовый порядок `period`, профиль `period:cover` и контекстный ранг RuBERT. При сильном совпадении сохраняется текстовый порядок внутри пула. Запросы без даты обрабатывает полноархивный ансамбль.

Почему так: во всех публичных запросах есть дата, и все 565 релевантных писем пришли ровно в названный день. Подробности и все замеры — в [docs/decisions.md](docs/decisions.md).

## Быстрый старт

Нужен Python 3.11+.

### 1. Код и зависимости

Linux / macOS:

```sh
git clone https://github.com/ttm545/b2baihacatondamirhuesos.git
cd b2baihacatondamirhuesos
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-baseline.txt -r requirements-api.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/ttm545/b2baihacatondamirhuesos.git
cd b2baihacatondamirhuesos
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt -r requirements-api.txt
```

Дальше в командах вместо `python` используйте `.venv/bin/python` (Linux/macOS) или `& .\.venv\Scripts\python.exe` (Windows).

### 2. Данные

Распакуйте архив организаторов в каталог `data/` в корне репозитория — тогда настраивать ничего не нужно. Архив: [ссылка организаторов](https://cloud.mail.ru/public/8S62/oQDJV17v6) или ветка `data/vk-dataset` этого репозитория:

```sh
git fetch origin data/vk-dataset
git show FETCH_HEAD:mipt_mail_search_participants.zip > dataset.zip
unzip dataset.zip -d data
```

В PowerShell последний шаг — `Expand-Archive dataset.zip -DestinationPath data`. Данные в другом месте — укажите путь в `VK_DATA_DIR` (`export VK_DATA_DIR=/path` или `$env:VK_DATA_DIR = 'C:\path'`). Каталог `data/` не попадает в Git.

### 3. Выдача текстового режима для бенчмарка

```sh
python -m vk_mail_search.predict --queries data/queries_dev.jsonl --out predictions.jsonl --k 100
```

Вход — любой JSONL с полями `query_id`, `synthetic_user`, `query`, `query_date`. Выход — по строке на запрос в порядке входа: `{"query_id": "...", "doc_ids": ["...", ...]}`, только письма своего ящика и не позже даты запроса, без повторов. Плохие строки не останавливают прогон: неизвестный ящик или пустой запрос — пустой список, нет даты — поиск без верхней границы, нечитаемая строка — пропуск; каждое предупреждение и сводка с временем поиска пишутся в stderr.

### 4. Оценка на публичном наборе

```sh
python -m vk_mail_search.evaluate --searcher period --k 10
python -m vk_mail_search.evaluate --searcher dated:bm25 --k 10
python -m vk_mail_search.evaluate --pred predictions.jsonl --k 10
```

Печатает Recall, nDCG, MRR на K = 5, 10, 20 (и на указанном K), MRR по запросам с одним релевантным письмом и p50/p95/p99 вызова поиска после прогрева. Устойчивость к перефразировкам: `--paraphrases tests/paraphrases.jsonl` (76 запросов другими словами, части `tune` и `test`).

### 5. Почта в браузере

Подготовьте модели и индекс по [инструкции гибридного поиска](solutions/semantic_ensemble/README.md). [Запуск в Docker](docs/architecture.md#запуск-контейнера) или запуск из Python после настройки переменных окружения:

```sh
python -m uvicorn solutions.semantic_ensemble.api:create_app --factory --host 127.0.0.1 --port 8000
```

Откройте <http://127.0.0.1:8000/> (описание API — <http://127.0.0.1:8000/docs>). Индекс строится при старте. В почте: выбор ящика, список писем, поиск с показом распознанного периода и темы, быстрые кнопки периода, подсветка совпадений, письма вне периода — под разделителем, «Похожие письма» под открытым письмом. «Сегодня» для ящика — дата его последнего письма. Сценарий показа — [docs/demo.md](docs/demo.md).

Публичный стенд: <https://vk-mail.597d550b.sslip.io/>.

### 6. Тесты

```sh
python -m unittest discover -s tests
```

## HTTP API

| Маршрут | Ответ |
| --- | --- |
| `GET /api/search?q=&mailbox=&as_of=&k=10` | `took_ms`, `period`, `topic`, `results[]` (`doc_id`, `doc_dt`, `subject`, `snippet`, `score`, `in_period`); `k` до 100 |
| `GET /api/mailboxes` | ящики: `mailbox`, `count`, `latest` |
| `GET /api/mailboxes/{mailbox}/docs?offset=&limit=` | письма ящика, новые сверху |
| `GET /api/docs/{doc_id}` | письмо целиком |
| `GET /api/docs/{doc_id}/similar?k=5` | однотипные письма того же ящика |

Публичная почта использует гибридный бэкенд. Отдельный текстовый API `vk_mail_search.api:app` запускает `period`; для сравнения с v1.0 ему можно задать `VK_SEARCHER=dated:bm25`.

## Структура

| Путь | Что внутри |
| --- | --- |
| `vk_mail_search/query_parser.py`, `dated.py` | период и тема запроса, письма периода первыми |
| `vk_mail_search/bm25.py` | BM25 по ящику с леммами, префиксами и весами полей |
| `vk_mail_search/period_rank.py` | переранжирование писем периода по лексическим признакам |
| `vk_mail_search/data.py`, `evaluate.py`, `predict.py` | загрузка данных, оценка, выдача для бенчмарка |
| `vk_mail_search/api.py`, `web/` | FastAPI и интерфейс почты |
| `vk_mail_search/searchers.py` | общий контракт `search(query, mailbox, as_of, k)` и фабрика поисковиков |
| `vk_mail_search/embeddings.py`, `dense.py`, `hybrid.py`, `tune_hybrid.py` | эксперименты с векторами (в v1.0 не используются) |
| `tests/` | тесты, 52 живые формулировки дат, перефразировки |
| `scripts/load_test.py` | нагрузочный замер HTTP p99 |
| `docs/` | требования, решения и замеры, сценарий демо |

### Эксперименты вне v1.0

Векторный поиск `deepvk/USER2-base` и гибрид с BM25 (`--searcher dense`, `dated:hybrid`) — после `pip install -r requirements-embeddings.txt`; первая индексация скачивает модель и строит кэш в `VK_CACHE_DIR` (на GPU ~24 мин). В v1.0 не вошли: внутри периода гибрид дал nDCG@10 0,9632 против 0,9756 у BM25. Отчёт — [docs/dense-search.md](docs/dense-search.md).

## Для команды

Правила работы, ветки и владельцы файлов — [AGENTS.md](AGENTS.md), чат — [TEAM_CHAT.md](TEAM_CHAT.md). Материалы кейса: [сводка](docs/vk-search-summary.md), [разбор требований](docs/brief.md), [условие (PDF)](docs/vk-mail-search.pdf), [сайт хакатона](https://hackathon.fpmi.tech/).
