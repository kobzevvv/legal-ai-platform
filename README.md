# Legal AI Platform

AI-ассистент по российскому законодательству на базе RAG (Retrieval-Augmented Generation).

## Что это

Чат-бот, который отвечает на юридические вопросы, ссылаясь на конкретные статьи 11 кодексов РФ:
- Гражданский кодекс (части 1-4)
- Уголовный кодекс
- Трудовой кодекс
- Налоговый кодекс (части 1-2)
- Семейный кодекс
- Жилищный кодекс
- КоАП

**4741 статья** в базе знаний, ответы со ссылками на конкретные статьи.

## Архитектура

```
User → Open WebUI → Pipeline (4-step RAG) → LLM (gpt-4o-mini)
                        ↓
                    Qdrant Cloud (vector search)
                        ↓
                    4741 статья из 11 кодексов
```

1. **Keyword Expansion** — LLM расширяет запрос юридическими терминами
2. **Query Rewriting** — переформулирует в точный поисковый запрос
3. **Vector Retrieval** — поиск релевантных статей в Qdrant (top-7)
4. **Generation** — LLM формирует ответ со ссылками на статьи

## Быстрый старт

```bash
# 1. Настроить переменные
cp .env.example .env
# Заполнить: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 2. Загрузить и распарсить кодексы (если нужна своя база)
cd ingestion
pip install -r requirements.txt
python download_codexes.py
python parse_codexes.py
python embed_and_upload.py
cd ..

# 3. Запустить
docker compose up -d
```

Открыть **http://localhost:3002**

### Первый запуск
1. Зарегистрируйте аккаунт (первый пользователь = администратор)
2. Модель **"Legal AI — Российское законодательство"** уже выбрана по умолчанию
3. Задавайте вопросы — ответы будут со ссылками на статьи кодексов

> **Важно:** В списке моделей должна быть выбрана именно "Legal AI — Российское законодательство", а не какая-либо другая модель. Только через наш pipeline ответы содержат ссылки на конкретные статьи.

## Тестирование

```bash
# Автоматический прогон 30 вопросов через pipeline API
cd eval
pip install openai
python run_live_test.py
# Результаты: eval/results/test_report.html
```

## Стоимость

- **~$0.001** за один ответ (gpt-4o-mini + embeddings)
- **~$1** за 1000 вопросов
- Ingestion (одноразово): ~$0.10

## Ограничения

- Только кодексы РФ (нет федеральных законов, подзаконных актов, судебной практики)
- Нет автообновления при изменении законодательства
- Не заменяет консультацию юриста

## Структура проекта

```
├── docker-compose.yml          # Open WebUI + Pipelines
├── .env.example                # Шаблон переменных
├── pipelines/
│   └── legal_rag.py            # 4-step RAG Pipeline
├── ingestion/
│   ├── download_codexes.py     # Скачивание кодексов
│   ├── parse_codexes.py        # HTML → JSON
│   └── embed_and_upload.py     # Embeddings → Qdrant
├── eval/
│   ├── run_live_test.py        # Автотесты (30 вопросов)
│   └── results/                # Отчёты
└── deploy/
    ├── nginx.conf
    └── setup-server.sh
```
