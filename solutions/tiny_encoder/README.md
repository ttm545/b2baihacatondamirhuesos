# Компактный векторный поиск VK Mail

Поиск использует русскоязычный энкодер RuBERT Tiny Retriever в PyTorch. У модели три слоя, 29 096 112 параметров и векторы из 312 чисел. На запрос выполняется один проход энкодера. На готовом индексе поиск вычисляет скалярное произведение с векторами нужного ящика. Письма с известной датой позже даты запроса исключаются. Письма указанного периода получают приоритет, остальные могут дополнить выдачу.

[Обучение на пяти базах и тематическая модель VK](TRAINING.md).

[Локальный индекс Qwen и режимы маршрутизации по датам](CONTEXT_INDEX.md).

Быстрый энкодер писем усредняет векторы токенов начала и доступного текста, затем применяет обученную линейную проекцию. Для запроса остаётся контекстный RuBERT. В индекс попали все 16 376 писем, по одному вектору на письмо. Текст входа ограничен темой и началом письма, максимум 80 токенов. Индекс FP16 занимает 10,22 МБ на диске, для поиска он загружается в FP32. Приложенные проекция писем и адаптер запроса занимают 0,81 МБ вместе. Веса RuBERT скачиваются отдельно, данные и индексы хранятся вне Git. [Модель RuBERT Tiny Retriever](https://huggingface.co/sergeyzh/rubert-tiny-retriever/tree/6dc0f7c)

## Замеры

CPU Intel Core i5-13420H, Windows, батч 128. Контекстное кодирование всех писем заняло 5 498,7 с на шести потоках. Быстрое кодирование с готовой проекцией заняло 6,8–8,7 с в двух холодных построениях на четырёх потоках. Полный старт процесса с загрузкой весов и архива в первом случае занял 19,45 с. Проекция обучалась 7,326 с на уже вычисленных векторах RuBERT. Она использовала тексты и векторы писем, без запросов и qrels. Коэффициент регуляризации выбран на 1 637 отложенных письмах по близости к вектору RuBERT, средний косинус 0,9757. Адаптер запросов обучен отдельно на 44 перефразировках `tune` и их исходных запросах с проверкой целыми семействами.

| Набор | Вариант | Recall@5 | nDCG@5 | MRR@5 | nDCG@10 | Recall@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Публичный VK, 99 | `dated:bm25` | 0,8479 | 0,9756 | 0,9832 | 0,9756 | 1,0000 |
| Публичный VK, 99 | RuBERT для писем и запросов | 0,7839 | 0,8853 | 0,9355 | 0,8870 | 1,0000 |
| Публичный VK, 99 | Быстрые письма и RuBERT | 0,7903 | 0,8924 | 0,9498 | 0,8944 | 1,0000 |
| Публичный VK, 99 | Быстрые письма и адаптированный RuBERT | **0,8053** | **0,9240** | **0,9742** | 0,9203 | 1,0000 |
| Отложенные перефразировки, 32 | `dated:bm25` | 0,5173 | 0,6285 | 0,7172 | 0,7029 | 1,0000 |
| Отложенные перефразировки, 32 | RuBERT для писем и запросов | 0,5184 | 0,5794 | 0,6599 | 0,6343 | 1,0000 |
| Отложенные перефразировки, 32 | Быстрые письма и RuBERT | 0,5255 | 0,6054 | 0,7109 | 0,6583 | 1,0000 |
| Отложенные перефразировки, 32 | Быстрые письма и адаптированный RuBERT | **0,5401** | **0,6299** | **0,7526** | 0,6802 | 1,0000 |

Этот прогон использует бинарную релевантность. Оценка каждого письма в выдаче принимает 1 для релевантного и 0 для остальных. В варианте с адаптером все пять позиций релевантны у 20 из 32 публичных запросов, где размечены хотя бы пять релевантных писем. У идеального порядка публичного набора Recall@5 составляет 0,8630 из-за запросов с более чем пятью релевантными письмами. В публичной таблице есть исходные запросы из процедуры обучения адаптера. Отложенные перефразировки используют другие семейства.

HTTP p99 для варианта с проекцией и адаптером составил **133,5 мс при K=5** и **192,7 мс при K=100** по серверному `took_ms`. Loopback p99 составил 137,0 и 201,6 мс. Замер выполнен в одном процессе Uvicorn с восемью клиентами и 1 000 запросов без ошибок на каждую глубину. Отдельный p99 одиночного поиска с готовым индексом без HTTP составил 11,3 мс при K=5 для варианта без адаптера запроса.

## Запуск

Выполняйте команды из корня репозитория. Положите распакованный архив организаторов в `data/` или укажите его через `--data-dir`.

```sh
python -m pip install -r solutions/tiny_encoder/requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('sergeyzh/rubert-tiny-retriever', revision='6dc0f7c', local_dir='.cache/rubert-tiny-retriever', allow_patterns=['*.json','model.safetensors','1_Pooling/config.json'])"
python -m solutions.tiny_encoder index --data-dir data --model .cache/rubert-tiny-retriever --index .cache/tiny-fast.npz --projection solutions/tiny_encoder/assets/projection-80.npz
python -m solutions.tiny_encoder evaluate --data-dir data --model .cache/rubert-tiny-retriever --index .cache/tiny-fast.npz --projection solutions/tiny_encoder/assets/projection-80.npz --adapter solutions/tiny_encoder/assets/adapter-80.npz --threads 1 --k 5
python -m solutions.tiny_encoder predict --data-dir data --model .cache/rubert-tiny-retriever --index .cache/tiny-fast.npz --projection solutions/tiny_encoder/assets/projection-80.npz --adapter solutions/tiny_encoder/assets/adapter-80.npz --queries data/queries_dev.jsonl --out .cache/predictions-tiny.jsonl --k 100
```

Для отложенных перефразировок добавьте к `evaluate` параметры `--paraphrases tests/paraphrases.jsonl --split test`. Сам экспорт работает по `documents.jsonl` и входному файлу запросов. Он возвращает JSONL с `query_id` и упорядоченным массивом `doc_ids`.

Для HTTP задайте `VK_DATA_DIR`, `VK_TINY_MODEL`, `VK_TINY_INDEX`, `VK_TINY_PROJECTION`, `VK_TINY_ADAPTER` по путям из примера и запустите существующий интерфейс почты на векторном поиске:

```sh
python -m uvicorn solutions.tiny_encoder.serve:create_tiny_app --factory --host 127.0.0.1 --port 8000
python scripts/load_test.py --url http://127.0.0.1:8000 --queries data/queries_dev.jsonl --calls 1000 --clients 8 --k 5
```

При первом запуске на новом наборе можно использовать готовую проекцию с тем же RuBERT и лимитом токенов. Её качество измерено на архиве VK. Для повторного обучения сначала постройте отдельный контекстный индекс командой `index` без `--projection`, затем запустите `python -m solutions.tiny_encoder.distill --data-dir data --model .cache/rubert-tiny-retriever --teacher-index .cache/tiny-teacher.npz --out .cache/projection-80.npz`. Для переобучения адаптера запросов есть `python -m solutions.tiny_encoder.tune --help`. Для каждого варианта указывайте свой индекс.
