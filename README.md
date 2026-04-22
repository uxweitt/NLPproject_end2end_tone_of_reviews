# SentimentAnalysisWithGRU

Проект позволяет классифицировать текст на positive, neutral и negative

## Overview

Проект решает задачу семантической классификации текста.
Основной сценарий использования:
- Обучение своей модели с помощью Trainer
- Инференс своей модели с помощью Inferencer
- Поднять базовое приложение на FastAPI с контейнеризацией
Может применяться для анализа отзывов
## Features

Основные реализованные компоненты:
- Preproccesor - для предобработки текста и преобразования в эмбеддинги
- Trainer - для обучения модели с выбором множества параметров
- Inferencer - инференс модели с выбором множества параметров
- app - приложение на FastAPI с endpoint (*/predict*)


## Tech Stack

- Python
- FastAPI
- PyTorch
- Docker

## Architecture

Опиши 3–6 основных компонентов:

- `final/ml/v1_GRU/data_set/` — подготовка и хранение данных.
- `final/ml/v1_GRU/engine/` — обучение и инференс.
- `final/ml/v1_GRU/models/` — наша нейронная модель.
- `final/ml/v1_GRU/utils/` — препроцессинг
- `final/app` — FastAPI приложение.
- `final/tests` — Тесты.

## Project Structure

```text
final/
├── app/
├── ml/
├── tests/
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/uxweitt/NLPproject_end2end_tone_of_reviews.git
cd NLPproject_end2end_tone_of_reviews
uv sync
```

## Run

Если это API:

```bash
uvicorn final.app.app:app --host 127.0.0.1 --port 8080
```

Если это скрипт:

```bash
uv run final/ml/v1_GRU/train.py
```

## Usage

Пример входа:

```json
{
  "text": "Очень понравился сервис"
}
```

Пример выхода:

```json
{
  "text": "Очень понравился сервис"
  "label": "positive",
  "confidence": 0.94
}
```

## Limitations

- Отсутствует парсинг аргументов командной строки.
- Отсуствует пайплайн для трансформеров из `transformers`
- Отсутствует подсчет метрик.

## Next Steps

- Добавить метрики.
- Реализовать pipeline для трансформеров
