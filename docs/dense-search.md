# Векторный поиск VK Почты

`DenseSearcher(docs)` реализует [общий контракт](decisions.md#d3-контракт-поисковика). Передайте документы с полями `doc_id`, `synthetic_user`, `doc_dt` и `doc`. В тексте корпуса уже содержатся заголовки письма, когда организаторы их предоставили.

При первом запуске кодируем письма пакетами локальной `deepvk/USER2-base`. Для писем, которые выходят за рабочее окно 512 токенов, кодируем фрагменты с перекрытием 64 токена и берём лучший косинус на письмо. Матрица и границы фрагментов хранятся в `VK_CACHE_DIR` (по умолчанию `.cache/`). Ключ кэша зависит от содержимого корпуса, ревизии модели и настройки фрагментов. При последующих запусках векторы загружаются с диска, в запросе кодируется только поисковая фраза.

## Установка и проверка

Установите зависимости в чистом окружении Python 3.11–3.14 и задайте путь к распакованному архиву:

```sh
python -m pip install -r requirements-embeddings.txt
export VK_DATA_DIR=/path/to/unpacked/dataset
export VK_CACHE_DIR="$PWD/.cache"
```

Чтобы построить кэш и проверить первый публичный запрос:

```sh
python - <<'PY'
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from vk_mail_search.dense import DenseSearcher

logging.basicConfig(level=logging.INFO)
data = Path(os.environ['VK_DATA_DIR'])
with (data / 'documents.jsonl').open(encoding='utf-8') as files:
    searcher = DenseSearcher(SimpleNamespace(**json.loads(line)) for line in files if line.strip())
with (data / 'queries_dev.jsonl').open(encoding='utf-8') as files:
    item = json.loads(next(files))
print(searcher.search(item['query'], item['synthetic_user'], item['query_date'], 10))
print(searcher.cache_path)
PY
```

В `search(query, mailbox, as_of, k)` учитываются только документы указанного ящика. Письма с известной датой позже `as_of` исключаются, письма с неизвестной датой остаются. Результат содержит максимум `k` уникальных `doc_id` с оценками.
