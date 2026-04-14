# Справочник программного интерфейса (API)

## 1. Интерфейс командной строки (CLI)

Основная точка входа для работы с помощью командной строки:

```bash
python -m mcd.cli <command> [options]
```

### Доступные команды

- **fit** — обучение модели на размеченных данных
- **predict** — предсказание для одного текста
- **eval** — оценка модели на тестовом наборе с расчётом метрик дрейфа

---

### 1.1 Команда `fit`

**Назначение:** Обучение детектора дрейфа на размеченных данных.

**Пример использования:**

```bash
python -m mcd.cli fit \
  --data data/sample_labeled.csv \
  --label-column queue \
  --text-columns subject body \
  --threshold-quantile 0.99 \
  --min-cluster-size 10 \
  --model-file models/model.joblib
```

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|---------|---------------|
| `--data` | str | Путь к CSV-файлу или ZIP-архиву с данными | требуется |
| `--label-column` | str | Имя колонки с метками кластеров | `queue` |
| `--text-columns` | list[str] | Список колонок для построения текста | `["subject", "body"]` |
| `--threshold-quantile` | float | Квантиль для расчёта порога дрейфа (0.0-1.0) | `0.99` |
| `--min-cluster-size` | int | Минимальный размер кластера для обучения | `10` |
| `--model-file` | str | Путь для сохранения обученной модели | требуется |

**Выход:**

- Модель сохраняется в формате `.joblib` по пути `--model-file`
- Маппинг меток сохраняется в JSON (в соседнем файле с префиксом `_mapping`)
- При использовании обучаемого проектора через Python API рядом с моделью дополнительно сохраняются файлы `{stem}_projector.pt` и `{stem}_projector_config.json` (см. раздел 2.1)

**Примечание:** команда `fit` в CLI создаёт детектор без нейросетевого проектора. Подключение `DeepMahalanobisProjector` выполняется из Python-кода (см. раздел 2.1 и скрипт обучения проектора).

---

### 1.2 Команда `predict`

**Назначение:** Предсказание кластера и детекция дрейфа для одного текста.

**Пример использования:**

```bash
python -m mcd.cli predict \
  --model-file models/model.joblib \
  --text "Проблема с платежом: не могу оплатить заказ"
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|---------|
| `--model-file` | str | Путь к сохранённой модели |
| `--text` | str | Анализируемый текст |

**Выход (в консоль):**

```
Predicted label: billing
Distance: 2.3456
Threshold: 2.1234
Drift detected: False
```

---

### 1.3 Команда `eval`

**Назначение:** Оценка модели на тестовом наборе с расчётом метрик детекции дрейфа.

**Пример использования:**

```bash
python -m mcd.cli eval \
  --data data/archive.zip \
  --label-column queue \
  --train-cluster-frac 0.8 \
  --threshold-quantile 0.99 \
  --min-cluster-size 10 \
  --seed 42 \
  --out-dir reports/eval_run
```

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|---------|---------------|
| `--data` | str | Путь к CSV или ZIP | требуется |
| `--label-column` | str | Колонка с метками | `queue` |
| `--text-columns` | list[str] | Текстовые колонки | `["subject", "body"]` |
| `--train-cluster-frac` | float | Доля кластеров для обучения (0.0-1.0) | `0.8` |
| `--threshold-quantile` | float | Квантиль для порога | `0.99` |
| `--min-cluster-size` | int | Мин. размер кластера | `10` |
| `--seed` | int | Семя для воспроизводимости | `42` |
| `--out-dir` | str | Каталог для сохранения результатов | `reports/demo_run` |
| `--auto-demo` | flag | Использовать демо-параметры по умолчанию | нет |

**Выходные файлы в `--out-dir`:**

- **metrics.json** — Метрики качества:
  - Drift metrics: precision, recall, F1, accuracy, ROC-AUC
  - Classification metrics: accuracy на IN-кластерах
  - Confusion matrix: TP, FP, TN, FN
  - Data stats: размеры кластеров, количество обучающих/тестовых выборок

- **predictions.csv** — Детальные предсказания для каждого образца:
  - `truncated_text`: первые 100 символов текста
  - `text_len`, `text_hash`: метаинформация
  - `true_label`, `predicted_label`: истинный и предсказанный кластер
  - `distance`, `threshold`, `score`: метрики дрейфа
  - `drift_pred`, `drift_true`: флаги дрейфа (предсказанный и истинный)
  - `split`: принадлежит ли образец IN или OOD группе

- **splits.json** — Информация о разбиении:
  - `in_clusters`: список кластеров для обучения
  - `ood_clusters`: список кластеров для тестирования дрейфа

- **README.md** — Краткое резюме результатов

---

## 2. Python API

### 2.1 Класс `MahalanobisDriftDetector`

Основной класс для обучения и предсказания при детекции концептуального дрейфа.

**Импорт:**

```python
from src.mcd.modeling.classifier import MahalanobisDriftDetector
```

**Конструктор:**

```python
MahalanobisDriftDetector(
    embedder=None,
    threshold_quantile: float = 0.99,
    min_cluster_size: int = 10,
    threshold_strategy: ThresholdStrategy | None = None,
    projector: nn.Module | None = None,
    projector_batch_size: int = 512,
)
```

**Параметры:**

- `embedder`: объект класса `Embedder` (по умолчанию — `SBERT`)
- `threshold_quantile`: квантиль для расчёта порогов (0.0-1.0), используется если не задана своя `threshold_strategy`
- `min_cluster_size`: минимальный размер кластера для включения в модель
- `threshold_strategy`: стратегия порога (`QuantileThresholdStrategy`, `ChiSquareThresholdStrategy` и др. из `src.mcd.modeling.thresholds`); по умолчанию — квантильная с параметром `threshold_quantile`
- `projector`: необязательный PyTorch-модуль (рекомендуется `DeepMahalanobisProjector` из `src.mcd.projection`); отображает эмбеддинги SBERT в пространство меньшей размерности перед расчётом Махаланобиса. Требуется установленный пакет `torch` (`pip install .[train]`)
- `projector_batch_size`: размер батча при применении проектора к матрице эмбеддингов (экономия памяти)

---

#### Метод `fit()`

```python
def fit(texts: List[str], labels: List[str]) -> None
```

**Назначение:** Обучение модели на размеченных данных.

**Параметры:**

- `texts` (List[str]): Список текстов (каждый текст предварительно обрабатывается)
- `labels` (List[str]): Список меток кластеров для каждого текста

**Процесс:**

1. Один раз вычисляет эмбеддинги для всех текстов через `embedder.embed(texts)`
2. При заданном `projector` — проецирует матрицу эмбеддингов в меньшую размерность (режим `eval`, без градиентов, батчами)
3. Для каждого кластера по меткам (без повторного вызова эмбеддера):
   - вычисляет среднее (mean) и ковариационную матрицу в **том же** пространстве (сырые или спроецированные эмбеддинги)
   - рассчитывает порог через `threshold_strategy` (например, квантиль расстояний Махаланобиса внутри кластера; размерность признака `feature_dim` соответствует размерности векторов после проекции)

**Пример без проектора:**

```python
detector = MahalanobisDriftDetector(threshold_quantile=0.99)
detector.fit(
    texts=["Это текст 1", "Это текст 2", ...],
    labels=["billing", "billing", "technical", ...]
)
```

**Пример с обученным проектором:**

```python
import json
import torch
from src.mcd.projection.projector import DeepMahalanobisProjector

with open("models/my_projector_config.json", encoding="utf-8") as f:
    cfg = json.load(f)
cfg.pop("schema", None)
proj = DeepMahalanobisProjector.from_architecture_dict(cfg)
proj.load_state_dict(torch.load("models/my_projector.pt", map_location="cpu"))
proj.eval()

detector = MahalanobisDriftDetector(threshold_quantile=0.99, projector=proj)
detector.fit(texts, labels)
```

---

#### Метод `predict()`

```python
def predict(text: str) -> Tuple[str, float, float, bool]
```

**Назначение:** Предсказание кластера и детекция дрейфа для одного текста.

**Параметры:**

- `text` (str): Анализируемый текст

**Возвращает кортеж:**

- `predicted_label` (str): Предсказанный кластер
- `distance` (float): Расстояние Махаланобиса до ближайшего кластера
- `threshold` (float): Пороговое значение для предсказанного кластера
- `is_drift` (bool): `True` если `distance > threshold`, иначе `False`

**Пример:**

```python
label, dist, thresh, is_drift = detector.predict(
    "Новое обращение пользователя"
)
print(f"Label: {label}, Drift: {is_drift}")
```

---

#### Метод `predict_batch()`

```python
def predict_batch(texts: List[str]) -> List[Tuple[str, float, float, bool]]
```

**Назначение:** Предсказание для списка текстов (один вызов `embed` на весь список, затем при необходимости проекция).

**Параметры:**

- `texts` (List[str]): Список текстов для анализа

**Возвращает:**

- List кортежей `(predicted_label, distance, threshold, is_drift)` для каждого текста

---

#### Методы `save()` и `load()`

```python
def save(path: str) -> None
def load(path: str) -> MahalanobisDriftDetector
```

**Назначение:** Сохранение и загрузка обученной модели.

**Параметры:**

- `path` (str): Путь к файлу модели (`.joblib`)

**Файлы на диске:**

| Файл | Содержимое |
|------|------------|
| `{path}` | Словарь joblib: центры кластеров, ковариации, пороги, метаданные (`has_projector`, `projector_batch_size` и др.) |
| `{path с заменой .joblib на _mapping.json}` | JSON: отображение метка → индекс |
| `{stem}_projector.pt` | `state_dict` PyTorch (только если при сохранении был задан `projector`) |
| `{stem}_projector_config.json` | Архитектура сети: `input_dim`, `hidden_dims`, `output_dim`, `dropout`, опционально `schema` |

Здесь `stem` — имя файла без расширения, например для `models/foo.joblib` это `foo_projector.pt` и `foo_projector_config.json`.

При загрузке: если оба файла проектора присутствуют, детектор восстанавливает `DeepMahalanobisProjector` и веса; если их нет — поведение как у классической модели без проекции. Несоответствие (есть только один из двух файлов проектора) приводит к ошибке с пояснением.

**Пример:**

```python
# Сохранение
detector.save("models/my_model.joblib")

# Загрузка
loaded_detector = MahalanobisDriftDetector.load("models/my_model.joblib")
```

---

### 2.2 Модуль `projection` и обучение проектора

**Класс `DeepMahalanobisProjector`**

```python
from src.mcd.projection import DeepMahalanobisProjector

model = DeepMahalanobisProjector(
    input_dim=384,
    hidden_dims=[256, 128],
    output_dim=64,
    dropout=0.1,
)
```

MLP: линейные слои, `BatchNorm1d`, `GELU`, `Dropout`, финальный линейный слой в `output_dim`.

Вспомогательные методы:

- `architecture_dict()` — словарь для JSON;
- `from_architecture_dict(cfg)` — восстановление экземпляра по сохранённому JSON;
- `forward(x: Tensor) -> Tensor` — прямой проход.

**Скрипт `scripts/train_projector.py`**

Обучение проектора на размеченном CSV (те же колонки, что и для детектора: `subject`, `body`, колонка меток). Используются эмбеддинги SBERT и функция потерь `TripletMarginLoss` (сближение векторов одного кластера, отдаление разных).

Требуется: `pip install .[train]` (или установленный `torch`).

```bash
python scripts/train_projector.py \
  --csv data/sample_labeled.csv \
  --label-column queue \
  --output models/projector.pt \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.001 \
  --hidden-dims 256,128 \
  --output-dim 64 \
  --sbert-model all-MiniLM-L6-v2
```

По умолчанию конфиг архитектуры пишется в `projector_config.json` (рядом с `projector.pt`), либо путь задаётся через `--config-out`.

Основные аргументы: `--csv`, `--label-column`, `--output`, `--config-out`, `--sbert-model`, `--embed-batch-size`, `--epochs`, `--batch-size`, `--lr`, `--hidden-dims`, `--output-dim`, `--dropout`, `--margin`, `--seed`, `--device`, `-v`.

---

### 2.3 Модуль `io`

**Функция `load_labeled_tickets_csv()`**

```python
from src.mcd.io import load_labeled_tickets_csv

texts, labels, label_to_index, index_to_label = load_labeled_tickets_csv(
    path="data/sample.csv",
    label_column="queue"
)
```

Загружает размеченные данные из CSV с автоматической предварительной обработкой текста.

**Функция `resolve_dataset_path()`**

```python
from src.mcd.io import resolve_dataset_path

csv_path, selected_csv_name = resolve_dataset_path("data/archive.zip")
```

Автоматически распаковывает ZIP-архив и выбирает CSV в приоритетном порядке.

---

### 2.4 Модуль `embedding`

**Класс `SBERT`**

```python
from src.mcd.embedding import SBERT

embedder = SBERT(model_name='all-MiniLM-L6-v2')
embeddings = embedder.embed(["text1", "text2"])
```

Генерирует SBERT-эмбеддинги для текстов.

---

### 2.5 Модуль `visualization`

**Функция `project_2d()`**

```python
from src.mcd.visualization.projection import project_2d

X_2d = project_2d(X, n_components=2)  # PCA проекция
```

**Функция `plot_scatter_2d()`**

```python
from src.mcd.visualization.scatter import plot_scatter_2d

fig = plot_scatter_2d(X, labels=y, title="Distribution")
```

Создаёт matplotlib Figure с 2D scatter-графиком.

---

## 3. Структура входных/выходных данных

### Входные данные (CSV)

Требуемые колонки:

- `subject` (str): Тема обращения
- `body` (str): Содержание обращения
- `<label_column>` (str): Метка кластера (по умолчанию `queue`)

### Выходные данные (eval)

Все результаты сохраняются в указанном `--out-dir`.

---

## 4. Типичный workflow

### Workflow 1: Обучение и сохранение

```bash
python -m mcd.cli fit \
  --data data/sample_labeled.csv \
  --label-column queue \
  --model-file models/demo.joblib
```

### Workflow 2: Предсказание на новых данных

```bash
python -m mcd.cli predict \
  --model-file models/demo.joblib \
  --text "Новое обращение"
```

### Workflow 3: Полная оценка с дрейфом

```bash
python -m mcd.cli eval \
  --data data/archive.zip \
  --auto-demo \
  --out-dir reports/demo_run
```

### Workflow 4: Использование в Python коде

```python
from src.mcd.modeling.classifier import MahalanobisDriftDetector
from src.mcd.io import load_labeled_tickets_csv

# Загрузить данные
texts, labels, _, _ = load_labeled_tickets_csv("data.csv", "queue")

# Обучить
detector = MahalanobisDriftDetector()
detector.fit(texts, labels)

# Предсказать
label, dist, thresh, is_drift = detector.predict("New ticket text")

# Сохранить
detector.save("models/trained.joblib")
```

### Workflow 5: обучение проектора и детектора с проекцией

```bash
pip install -e ".[train]"
python scripts/train_projector.py --csv data.csv --label-column queue --output models/projector.pt
```

По умолчанию рядом создаётся `models/projector_config.json`.

```python
import json
import torch
from src.mcd.modeling.classifier import MahalanobisDriftDetector
from src.mcd.io import load_labeled_tickets_csv
from src.mcd.projection.projector import DeepMahalanobisProjector

texts, labels, _, _ = load_labeled_tickets_csv("data.csv", "queue")
with open("models/projector_config.json", encoding="utf-8") as f:
    cfg = json.load(f)
cfg.pop("schema", None)
proj = DeepMahalanobisProjector.from_architecture_dict(cfg)
proj.load_state_dict(torch.load("models/projector.pt", map_location="cpu"))
proj.eval()

detector = MahalanobisDriftDetector(projector=proj)
detector.fit(texts, labels)
detector.save("models/with_projector.joblib")
```