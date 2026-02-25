# Legal AI Platform

AI-ассистент по российскому законодательству на базе RAG (Retrieval-Augmented Generation).

## Что это

Чат-бот, который отвечает на юридические вопросы, ссылаясь на конкретные статьи 7 кодексов РФ:
- Гражданский кодекс (части 1-4)
- Уголовный кодекс
- Трудовой кодекс
- Налоговый кодекс (части 1-2)
- Семейный кодекс
- Жилищный кодекс
- КоАП

## Архитектура

```
User → Open WebUI → Pipeline (RAG) → LLM
                        ↓
                    Qdrant (vector search)
                        ↓
                    Статьи кодексов
```

1. **Ingestion**: HTML кодексов → парсинг → embeddings → Qdrant
2. **Pipeline**: запрос → vector search → контекст + промпт → LLM → ответ со ссылками
3. **UI**: Open WebUI (Docker)

## Быстрый старт

```bash
# 1. Настроить переменные
cp .env.example .env
# Заполнить API ключи в .env

# 2. Загрузить и распарсить кодексы
cd ingestion
pip install -r requirements.txt
python download_codexes.py
python parse_codexes.py
python embed_and_upload.py

# 3. Запустить UI
cd ..
docker compose up -d
```

Открыть http://localhost:3000

## Eval

```bash
cd eval
python run_eval.py
```
