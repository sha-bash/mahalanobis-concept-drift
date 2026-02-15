# Data directory

Данная директория содержит демонстрационные примеры формата входных данных,
используемых программой Mahalanobis Concept Drift Detector.

---

## Основной датасет

Для запуска полного демо-сценария используется публичный датасет:

Multilingual Customer Support Tickets  
Источник: Kaggle  
Ссылка: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

В соответствии с лицензионной политикой Kaggle датасет
не распространяется в составе репозитория.

---

## Как скачать датасет

1. Зарегистрироваться на платформе Kaggle.
2. Установить Kaggle API (https://www.kaggle.com/docs/api).
3. Выполнить команду:

`kaggle datasets download -d tobiasbueck/multilingual-customer-support-tickets`


4. Поместить скачанный архив в директорию:

`data/archive.zip`


5. При запуске демо-сценария архив автоматически распаковывается
в директорию `data/extracted/`.

---

## Демонстрационные файлы

- `sample_labeled.csv` — пример размеченных данных.
- `sample_unlabeled.csv` — пример данных для демонстрации детекции дрейфа.

Данные файлы используются для быстрой проверки работоспособности системы
без загрузки полного датасета.
