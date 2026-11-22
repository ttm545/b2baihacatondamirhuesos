# Семантический отбор топ‑30 для ensemble

Ветка основана на `task/vk-search-qwen-index` Дамира, коммит `da14429`. Новый режим влияет на запросы, где пользователь указывает день или период.

1. Запрос разбирается штатным парсером даты и темы. Семантический BM25 ищет по полному доступному ящику. Штатный календарный отбор сохраняет приоритет писем внутри указанного периода.
2. В BM25 остаются точные леммы, веса темы и служебных слов, IDF, поправка на длину и насыщение частоты. Вместо первых пяти букв используются до 24 соседних лемм из Navec с косинусом от 0,45. Вхождения соседей взвешиваются по косинусу. Коэффициент дополнительного канала остаётся равным 0,15.
3. До 30 писем из нужного периода получают исходные текстовые и контекстные признаки ансамбля. Сохраняются веса `period` / `period:cover` / `context` 0,35 / 0,20 / 0,45 и порог совпадения темы 0,5. Если совпадение сильное, внутри новой выборки используется текстовый порядок `period`. Остальную выдачу дополняет штатный `period`. Запросы без даты обслуживает исходный полноархивный режим ансамбля.

## Запуск

Подготовьте компактный индекс RuBERT по [инструкции Дамира](../tiny_encoder/README.md). Скачайте [русские векторы Navec](https://github.com/natasha/navec/blob/master/README.md#downloads) и храните их локально.

```sh
python -m pip install -r requirements-api.txt -r solutions/semantic_ensemble/requirements.txt
mkdir -p .cache
curl -fL https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar -o .cache/navec.tar
```

```sh
assets=solutions/tiny_encoder/assets
export VK_DATA_DIR=data
export VK_CONTEXT_MODE=ensemble
export VK_TINY_MODEL=.cache/rubert-tiny-retriever
export VK_TINY_PROJECTION="$assets/projection-80.npz"
export VK_TINY_ADAPTER="$assets/adapter-80.npz"
export VK_CONTEXT_INDEX=.cache/tiny-fast.npz
export VK_CONTEXT_ADAPTER="$assets/qwen-context.npz"
export VK_WORD_VECTORS=.cache/navec.tar
python -m uvicorn solutions.semantic_ensemble.api:create_app --factory --port 8001
```

```sh
python -m solutions.semantic_ensemble.benchmark \
  --data-dir "$VK_DATA_DIR" --model "$VK_TINY_MODEL" --projection "$VK_TINY_PROJECTION" \
  --index "$VK_CONTEXT_INDEX" --adapter "$VK_TINY_ADAPTER" --context "$VK_CONTEXT_ADAPTER" \
  --config "$assets/ensemble.json" --word-vectors "$VK_WORD_VECTORS" --out .cache/semantic-ensemble
python -m unittest solutions.semantic_ensemble.test_search
```

Бенчмарк пишет выдачи, метрики финального ранжирования и отдельный Recall кандидатов на глубинах 20 и 30 для обоих способов отбора. Задержка измеряется для K=10, после прогрева, при одном клиенте и без HTTP.

## Результаты

| Набор | Recall кандидатов `period` @20 / @30 | Recall кандидатов BM25 с Navec @20 / @30 |
| --- | ---: | ---: |
| 99 публичных запросов | 97,94% / 98,62% | 95,49% / 96,44% |
| 32 отложенные перефразировки | 88,62% / 91,12% | 88,19% / 93,80% |

| Финальная выдача | Recall@5 исходный → новый | nDCG@10 исходный → новый | p99, мс исходный → новый |
| --- | ---: | ---: | ---: |
| 99 публичных запросов | 85,16% → 84,17% | 0,9906 → 0,9782 | 5,3 → 10,8 |
| 32 отложенные перефразировки | 56,25% → 56,87% | 0,7168 → 0,7147 | 3,8 → 9,3 |

На перефразировках при равной глубине 30 Recall кандидатов вырос на 2,68 п.п., однако nDCG@10 итоговой выдачи немного снизился. На публичном наборе топ‑30 потерял часть писем о погрузке и выгрузке. Итоговый Recall@100 сохранился на 100% на обоих наборах. Замер задержки сделан локально на Apple M4.
