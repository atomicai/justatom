# Эмбеддинги на основе языковых моделей

Четыре дополнительных энкодера подключены к общим путям обучения и локального
поиска. Они поддерживают LoRA, InfoNCE и геометрическое ограничение AnchorBank.

| Модель | Семейство | Размер вектора | Получение вектора | Конфигурация |
|---|---|---:|---|---|
| `Qwen/Qwen3-Embedding-4B` | Qwen3 | 2560 | Последний непустой токен | [Qwen](https://github.com/atomicai/justatom/blob/master/configs/experiments/lm-qwen3-4b-anchor.yaml) |
| `nvidia/Nemotron-3-Embed-1B-BF16` | Ministral | 2048 | Двунаправленное внимание, среднее по непустым токенам | [Nemotron](https://github.com/atomicai/justatom/blob/master/configs/experiments/lm-nemotron3-1b-anchor.yaml) |
| `google/embeddinggemma-300m` | Gemma3 | 768 | Двунаправленное внимание, среднее и два обученных слоя проекции | [Gemma](https://github.com/atomicai/justatom/blob/master/configs/experiments/lm-embeddinggemma-300m-anchor.yaml) |
| `tencent/WeMM-Embedding-2B` | Qwen3.5 | 2048 | Текстовый вход, вектор токена `<embedding>` | [WeMM](https://github.com/atomicai/justatom/blob/master/configs/experiments/lm-wemm-2b-anchor.yaml) |

Для новых архитектур предусмотрены дополнительные зависимости `lm-embeddings`:
Transformers ≥ 5.15 и PEFT ≥ 0.20. Они дополняют обычное окружение JustAtom с PyTorch.
Установка из корня репозитория для обучения и локального поиска:

```bash
python -m pip install -e ".[torch,serve,lm-embeddings]"
```

У каждой конфигурации закреплена ревизия весов. Формат запросов и документов
выбирается автоматически при `query_prefix: null` и `content_prefix: null`;
явное значение, в том числе пустая строка, сохраняется. При локальном поиске
аналогично обрабатываются `query_prefix` и `document_prefix`. Сохранённый энкодер
читает собственные префиксы из конфигурации процессора. На стороне удалённого
сервера префиксы задаются явно, поскольку сервер может добавлять их самостоятельно.

WeMM использует родной шаблон сообщения пользователя и завершающий токен из
`tokenizer.json`. Токен остаётся после обрезки длинного текста. Изображения и видео
в эту интеграцию не входят; библиотека для обработки изображений не требуется.
Загружается встроенный `Qwen3_5Model`, без генерационной головы и выполнения
удалённого Python-кода. Визуальная часть заморожена, LoRA подключается только к
линейным слоям языковой части.

Для Nemotron согласованы двунаправленная маска и флаг внимания в модулях
Ministral: в Transformers 5.15 флаг модуля изначально причинный даже при
`config.is_causal=false`. Проверка влияния последующих токенов и независимости
результата от дополнения батча предотвращает случайное переключение внимания.
Маска дополнения передаётся в модель и учитывается при усреднении токенов.

EmbeddingGemma загружает оба слоя проекции из опубликованной последовательности
модулей и сохраняет их вместе с энкодером. При обучении LoRA они заморожены,
поэтому отключение адаптера восстанавливает исходную модель для AnchorBank.
Поддерживаются float32 и bfloat16. Доступ к весам требует принятия условий на
[странице Google](https://huggingface.co/google/embeddinggemma-300m) и авторизации
Hugging Face; без доступа библиотека не сможет загрузить эту модель.

## Тесты

Из корня репозитория с установленными зависимостями `test`:

```bash
python -m pytest tests/test_lm_embeddings.py -q
```

Тесты используют уменьшенные экземпляры архитектур с локальными весами и работают
на CPU без скачивания моделей. Они проверяют получение и нормализацию векторов,
маскирование, градиенты LoRA и AnchorBank, неизменность базовых векторов при
отключённых адаптерах, а также сохранение и повторную загрузку энкодера.

## Обучение и оценка

Конфигурации задают LoRA r16/α32, bfloat16, пересчёт активаций при обратном проходе,
батч 2 и накопление 16 батчей. Набор подготовленных обучающих пар передаётся явно:

```bash
python -m justatom.api.train \
  --config configs/experiments/lm-nemotron3-1b-anchor.yaml \
  --dataset.name_or_path /path/to/prepared-train.parquet \
  --dataset.labels_field query \
  --dataset.content_field content
```

Для парного InfoNCE используются те же веса, данные и параметры, с переопределениями:

```text
--method vanilla --anchor_bank.enabled false --anchor_bank.size 0 --gradient_projection.enabled false
```

Это примеры конфигураций энкодера. Длины текстов, размер батча и банка опорных
векторов выбираются с учётом доступной памяти. Разделение данных и оценка качества
задаются отдельно; для сравнения методов используются одинаковые данные и условия.

Описание исходных моделей: [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-4B),
[NVIDIA](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16),
[Google](https://huggingface.co/google/embeddinggemma-300m),
[Tencent](https://huggingface.co/tencent/WeMM-Embedding-2B).
