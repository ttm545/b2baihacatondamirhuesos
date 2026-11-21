# Умный поиск в Почте — кейс VK, хакатон ФПМИ 2026

Веб-прототип почты с поиском по архиву писем организаторов. Пользователь пишет запрос своими словами — «уведомления об отклонённых платежах, полученные вчера» — и получает письма нужного дня и нужного типа, а не все письма со словом «платёж».

**Версия 1.0** на публичном наборе (99 запросов, 16 376 писем, 12 ящиков):

| Вариант | Recall@10 | nDCG@10 | MRR@10 | p99 |
| --- | ---: | ---: | ---: | --- |
| Обычный полнотекстовый поиск (BM25 с леммами) | 0,4807 | 0,3489 | 0,3452 | 6 мс |
| **v1.0 (`dated:bm25`)** | **0,9263** | **0,9756** | **0,9832** | 12 мс вызов поиска; 99 мс HTTP с очередью, 8 клиентов, 1 vCPU |

В поисковом движке v1.0 нет нейросетей и внешних API: 0 параметров моделей при лимите организаторов 3 млрд.

## Как это работает

1. **Разбор запроса** (`vk_mail_search/query_parser.py`). Из запроса выделяются период и тема. Период — точные даты с годом и без, «вчера», «позавчера», «3 дня назад», «на прошлой неделе», «в сентябре», «с 1 по 15 ноября», «15.10», дни недели и выходные; относительные даты считаются от даты запроса. Тема — запрос без «Найди/Покажи», даты и «полученные».
2. **Письма периода — первыми** (`vk_mail_search/dated.py`). Внутри периода письма ранжируются по теме, письма вне периода не выбрасываются, а идут следом: ошибка разбора меняет порядок, но не теряет письмо.
3. **BM25 по теме** (`vk_mail_search/bm25.py`), всегда внутри ящика пользователя и без писем позже даты запроса: леммы pymorphy3, совпадение по первым 5 буквам леммы для однокоренных слов («погрузка» ↔ «погружен»), тема письма весит больше тела, служебные слова почти не весят, бонус за совпадение с темой письма.

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

### 3. Выдача для бенчмарка

```sh
python -m vk_mail_search.predict --queries data/queries_dev.jsonl --out predictions.jsonl --k 100
```

Вход — любой JSONL с полями `query_id`, `synthetic_user`, `query`, `query_date`. Выход — по строке на запрос в порядке входа: `{"query_id": "...", "doc_ids": ["...", ...]}`, только письма своего ящика и не позже даты запроса, без повторов. Плохие строки не останавливают прогон: неизвестный ящик или пустой запрос — пустой список, нет даты — поиск без верхней границы, нечитаемая строка — пропуск; каждое предупреждение и сводка с временем поиска пишутся в stderr.

### 4. Оценка на публичном наборе

```sh
python -m vk_mail_search.evaluate --searcher dated:bm25 --k 10
python -m vk_mail_search.evaluate --pred predictions.jsonl --k 10
```

Печатает Recall, nDCG, MRR на K = 5, 10, 20 (и на указанном K), MRR по запросам с одним релевантным письмом и p50/p95/p99 вызова поиска после прогрева. Устойчивость к перефразировкам: `--paraphrases tests/paraphrases.jsonl` (76 запросов другими словами, части `tune` и `test`).

### 5. Почта в браузере

```sh
python -m uvicorn vk_mail_search.api:app --host 127.0.0.1 --port 8000
```

Откройте <http://127.0.0.1:8000/> (описание API — <http://127.0.0.1:8000/docs>). Индекс строится при старте за ~5 секунд. В почте: выбор ящика, список писем, поиск с показом распознанного периода и темы, быстрые кнопки периода, подсветка совпадений, письма вне периода — под разделителем, «Похожие письма» под открытым письмом. «Сегодня» для ящика — дата его последнего письма. Сценарий показа — [docs/demo.md](docs/demo.md).

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

Поисковик выбирается переменной `VK_SEARCHER` (по умолчанию `dated:bm25`).

## Структура

| Путь | Что внутри |
| --- | --- |
| `vk_mail_search/query_parser.py`, `dated.py` | период и тема запроса, письма периода первыми |
| `vk_mail_search/bm25.py` | BM25 по ящику с леммами, префиксами и весами полей |
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
