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
    min_cluster_size: int = 10
)
```

**Параметры:**

- `embedder`: объект класса `Embedder` (по умолчанию — `SBERT`)
- `threshold_quantile`: квантиль для расчёта порогов (0.0-1.0)
- `min_cluster_size`: минимальный размер кластера для включения в модель

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

1. Вычисляет SBERT-эмбеддинги для всех текстов
2. Для каждого кластера:
   - Вычисляет среднее (mean) и ковариацию
   - Расчитывает пороговое значение как `quantile` расстояний Махаланобиса внутри кластера

**Пример:**

```python
detector = MahalanobisDriftDetector(threshold_quantile=0.99)
detector.fit(
    texts=["Это текст 1", "Это текст 2", ...],
    labels=["billing", "billing", "technical", ...]
)
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

**Назначение:** Предсказание для списка текстов (оптимизировано с кэшированием эмбеддингов).

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

**Пример:**

```python
# Сохранение
detector.save("models/my_model.joblib")

# Загрузка
loaded_detector = MahalanobisDriftDetector.load("models/my_model.joblib")
```

---

### 2.2 Модуль `io`

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

### 2.3 Модуль `embedding`

**Класс `SBERT`**

```python
from src.mcd.embedding import SBERT

embedder = SBERT(model_name='all-MiniLM-L6-v2')
embeddings = embedder.embed(["text1", "text2"])
```

Генерирует SBERT-эмбеддинги для текстов.

---

### 2.4 Модуль `visualization`

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