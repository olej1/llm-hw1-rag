# Доказательный RAG по ТФКП — ДЗ 1

**ФИО:** Зажигин Олег Андреевич
**Группа:** Б05-307

Система отвечает на тестовые вопросы по учебнику В. В. Горяйнова и Е. С. Половинкина «Лекции по ТФКП» (2025). Она выбирает варианты, отказывается при недостатке данных и ссылается на блоки корпуса. Генерация — только Qwen3.5-4B через OpenAI-совместимый API. Поиск и эмбеддинги работают на CPU.

## Запуск

Python 3.11, чистое окружение:

```bash
python -m pip install -r requirements.txt
python build_index.py --corpus /path/to/corpus.jsonl --output ./index --config config.json
export LLM_BASE_URL=http://host:port/v1 LLM_MODEL=Qwen3.5-4B LLM_API_KEY=... SUBMISSION_ID=...
python predict.py --index ./index --questions /path/to/questions.jsonl --output ./answers.jsonl --trace ./trace.jsonl --config config.json
```

При первом запуске `build_index.py` скачивает bge-m3 (ревизия зафиксирована в `config.json`) в `models/` рядом с конфигом. `predict.py` потом берёт веса из этого кэша. Пути в `config.json` указаны относительно самого конфига.

Подготовка данных (корпус и dev-набор в репозиторий не входят, они пересобираются кодом):

```bash
python make_corpus.py --pdf tfkp_2025.pdf --output data/corpus.jsonl          # PDF -> блоки с формулами
python build_dev.py --qa tfkp_15_qa.md --corpus data/corpus.jsonl --out-prefix data/dev
python build_dev.py --qa own_qa.md --corpus data/corpus.jsonl --out-prefix own --own
./run_exp.sh final        # прогон configs/final.json на dev + оценка -> experiments.csv
```

`evaluate.py` проверяет формат и лимиты, а затем считает A, F1 ссылок, Q, стоимость C и (с `--cref`) итог по формуле задания.

## Ресурсы

| | |
|---|---|
| Оборудование (поиск, индекс) | MacBook Air M4, 24 ГБ RAM, только CPU |
| Сборка индекса | 125 с, пиковая RAM 2.9 ГБ, размер индекса 3.1 МБ (529 блоков) |
| Эмбеддинги | `BAAI/bge-m3`, 568M параметров, ревизия `5617a9f6…` |
| Reranker (выключен) | `BAAI/bge-reranker-v2-m3`, 568M, ревизия `953dc6f6…`, см. отчёт |
| LLM при разработке | Qwen3.5-4B (ревизия `851bf6e8…`), vLLM 0.19, 1×Tesla V100 32 ГБ, fp16, `max-model-len 8192`; детали в `runtime.json` |
| Расход на вопрос (dev) | 2 вызова, ~3200 входных и ~350 выходных токенов, медиана ~8.5 с |

## Архитектура

```
PDF ──make_corpus.py──► corpus.jsonl (блоки: теорема / определение / пример / абзац, формулы в LaTeX-тексте)
                              │ build_index.py
                              ▼
          index/: BM25-токены (стемминг + нормализация формул), dense.npy (bge-m3), refs.json (метки → блоки)

вопрос ─► [LLM 1] rewrite: 2–3 запроса с названиями теорем/формул
       ─► поиск: вопрос, вопрос+вариант, rewrite-запросы × (BM25, bge-m3) ─► взвешенный RRF ─► top-5
       ─► расширение: следующий блок (доказательство) + блоки, на которые есть ссылки («теорема 5.3», «(5.2)»)
       ─► [LLM 2] ответ JSON: вердикт по каждому варианту, answer_ids, abstain, rationale, citations, missing
       ─► (если missing не пуст) повторный поиск по missing и [LLM 3] — не более 4 вызовов
       ─► проверка формата: варианты из списка, ссылки только на показанные блоки, ≤4, отказ без ссылок
```

| Файл | Назначение |
|---|---|
| `make_corpus.py` | PDF → блоки. Декодирует T2A-шрифты, восстанавливает дроби и черту над z по векторным линиям, индексы и пределы ∑/∫ |
| `textnorm.py` | Общая нормализация LaTeX и Unicode для вопросов и корпуса (`\bar z` и `z̄` → `zbar`), стемминг |
| `retrieval.py` | BM25, dense-поиск, RRF, соседние блоки, перекрёстные ссылки, сборка индекса |
| `build_index.py`, `predict.py` | CLI задания |
| `evaluate.py`, `run_exp.sh`, `configs/` | Оценка и конфигурации экспериментов |
| `build_dev.py`, `own_qa.md` | dev/own-наборы, эталонные блоки задаются якорями (страница + начало текста) |

Подробности, эксперименты и анализ ошибок — в [report.md](report.md).
