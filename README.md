# Дипломный проект

## Описание

Этот проект представляет собой пример машинного обучения для классификации данных. Проект включает в себя предобработку данных, обучение модели, оценку модели и использование модели для предсказаний.

## Структура проекта

- `data/`: Содержит данные
  - `raw/`: Исходные данные
  - `processed/`: Обработанные данные
- `src/`: Исходный код
  - `data_preprocessing.py`: Предобработка данных
  - `model_training.py`: Обучение модели
  - `model_evaluation.py`: Оценка модели
  - `model_prediction.py`: Использование модели для предсказаний
- `models/`: Сохраненные модели
- `README.md`: Описание проекта
- `requirements.txt`: Зависимости проекта

## Как использовать

1. Установите зависимости: `pip install -r requirements.txt`
2. Запустите предобработку данных: `python src/data_preprocessing.py`
3. Обучите модель: `python src/model_training.py`
4. Оцените модель: `python src/model_evaluation.py`
5. Используйте модель для предсказаний: `python src/model_prediction.py`

## Зависимости

- pandas
- scikit-learn
- imbalanced-learn
- joblib

## Автор

EVK