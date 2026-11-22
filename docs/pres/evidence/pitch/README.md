# Данные для презентации

Варианты 99 публичных запросов с теми же `query_id`, `synthetic_user` и `query_date`, поэтому к ним применяется `qrels_dev.jsonl` организаторов. Меняется только текст запроса:

- `queries_nodate.jsonl` — только тема, без даты;
- `queries_week.jsonl` — тема и период «с ДД.ММ.ГГГГ по ДД.ММ.ГГГГ», 7 дней, заканчивается нужным днём;
- `queries_month.jsonl` — тема и «в <месяце> <годе>».

Разметка отмечает только письма нужного дня, поэтому на неделе, месяце и без даты метрики занижены: однотипные письма других дней считаются ошибкой. Это стресс-тест, а не оценка организаторов.

## Выдача финального поисковика

Подготовьте зависимости и переменные моделей из [инструкции запуска](../../../../solutions/semantic_ensemble/README.md). CLI принимает те же переменные, что и API. Для v1.1 нужны режим `VK_CONTEXT_MODE=ensemble`, вес `VK_SEMANTIC_WEIGHT=0.085` и все указанные пути к локальным моделям.

```sh
python -m solutions.semantic_ensemble.predict --queries data/queries_dev.jsonl \
  --out predictions.jsonl --k 100 --data-dir data
python -m vk_mail_search.evaluate --data-dir data --pred predictions.jsonl --k 100
```

В [results](results/) содержатся 20 выдач по четырём публичным наборам и двум частям перефразировок, 12 сценариев нагрузки, повтор v1.0 при K=10 и [паспорт](results/passport.md). `dated:bm25` в имени файлов соответствует тому же значению `--searcher`.

## Локальные модели

Склонируйте ветку `data/models-v1.1` в каталог `models/`. Проверьте детали по [манифесту ветки](results/models-branch.sha256) и соберите веса. Ниже `../pitch` — путь к рабочей копии `task/pitch-data`.

```sh
cd models
sha256sum -c ../pitch/docs/pitch/results/models-branch.sha256
cat rubert-tiny-retriever/model.safetensors.part-* > rubert-tiny-retriever/model.safetensors
sha256sum -c ../pitch/docs/pitch/results/models-original.sha256
```
