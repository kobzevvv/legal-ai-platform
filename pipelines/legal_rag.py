"""
Legal RAG Pipeline для Open WebUI.

3-step pipeline:
1. Query Rewriting — переформулировка вопроса для лучшего поиска
2. Retrieval — поиск релевантных статей в Qdrant
3. Generation — генерация ответа со ссылками на конкретные статьи

Устанавливается как Pipeline в Open WebUI.
"""

import os
from typing import Generator, Iterator, List, Optional, Union

from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = ""
        QDRANT_URL: str = ""
        QDRANT_API_KEY: str = ""
        QDRANT_COLLECTION: str = "russian_law"
        EMBEDDING_MODEL: str = "text-embedding-3-small"
        LLM_MODEL: str = "gpt-4o-mini"
        TOP_K: int = 7
        SCORE_THRESHOLD: float = 0.3

    def __init__(self):
        self.name = "Legal AI — Российское законодательство"
        self.valves = self.Valves(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            QDRANT_URL=os.getenv("QDRANT_URL", ""),
            QDRANT_API_KEY=os.getenv("QDRANT_API_KEY", ""),
        )
        self.openai_client = None
        self.qdrant_client = None

    async def on_startup(self):
        from openai import OpenAI
        from qdrant_client import QdrantClient

        self.openai_client = OpenAI(api_key=self.valves.OPENAI_API_KEY)
        self.qdrant_client = QdrantClient(
            url=self.valves.QDRANT_URL,
            api_key=self.valves.QDRANT_API_KEY,
        )

    async def on_shutdown(self):
        pass

    async def on_valves_updated(self):
        await self.on_startup()

    def _embed(self, text: str) -> list[float]:
        """Получает embedding для текста."""
        response = self.openai_client.embeddings.create(
            model=self.valves.EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def _expand_keywords(self, user_message: str) -> str:
        """Step 0: генерирует юридические ключевые слова из бытового вопроса.

        Пользователь часто спрашивает бытовым языком («пьяная езда», «кинули с деньгами»),
        а в кодексах используются юридические термины («управление в состоянии опьянения»,
        «мошенничество»). Дешёвая модель генерирует 5-10 ключевых слов-синонимов.
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — юридический терминолог. Получив бытовой вопрос, "
                        "выведи 5-10 юридических ключевых слов и фраз на русском, "
                        "которые используются в кодексах РФ для описания этой ситуации. "
                        "Формат: слова через запятую, без нумерации и пояснений. "
                        "Пример: вопрос «пьяная езда» → "
                        "управление транспортным средством в состоянии опьянения, "
                        "нетрезвое вождение, медицинское освидетельствование, "
                        "лишение права управления, административное правонарушение"
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()

    def _rewrite_query(self, user_message: str, keywords: str, chat_history: list) -> str:
        """Step 1: переформулирует вопрос для лучшего поиска с учётом ключевых слов."""
        # Берём последние 3 сообщения для контекста
        context_messages = []
        for msg in chat_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                context_messages.append(f"{role}: {content}")

        context = "\n".join(context_messages)

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — помощник для поиска по российским кодексам. "
                        "Переформулируй вопрос пользователя в поисковый запрос, "
                        "который лучше всего найдёт релевантные статьи кодексов РФ. "
                        "Используй предоставленные юридические ключевые слова. "
                        "Верни ТОЛЬКО переформулированный запрос, без пояснений."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Контекст диалога:\n{context}\n\n"
                        f"Юридические ключевые слова: {keywords}\n\n"
                        f"Вопрос: {user_message}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    def _retrieve(self, query: str) -> list[dict]:
        """Step 2: поиск статей в Qdrant."""
        query_embedding = self._embed(query)

        results = self.qdrant_client.query_points(
            collection_name=self.valves.QDRANT_COLLECTION,
            query=query_embedding,
            limit=self.valves.TOP_K,
            score_threshold=self.valves.SCORE_THRESHOLD,
        )

        articles = []
        for hit in results.points:
            p = hit.payload
            articles.append({
                "codex": p.get("codex", ""),
                "article_num": p.get("article_num", ""),
                "article_title": p.get("article_title", ""),
                "chapter": p.get("chapter", ""),
                "text": p.get("text", ""),
                "score": hit.score,
            })

        return articles

    def _format_context(self, articles: list[dict]) -> str:
        """Форматирует найденные статьи для промпта."""
        parts = []
        for i, art in enumerate(articles, 1):
            parts.append(
                f"[{i}] {art['codex']} — Статья {art['article_num']}. "
                f"{art['article_title']}\n"
                f"(Релевантность: {art['score']:.2f})\n"
                f"{art['text']}\n"
            )
        return "\n---\n".join(parts)

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list,
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline entry point."""

        if not self.openai_client or not self.qdrant_client:
            return "Pipeline не инициализирован. Проверьте Valves (API ключи)."

        # Step 0: Keyword Expansion (бытовой язык → юридические термины)
        keywords = self._expand_keywords(user_message)

        # Step 1: Query Rewriting (с учётом ключевых слов)
        search_query = self._rewrite_query(user_message, keywords, messages)

        # Step 2: Retrieval
        articles = self._retrieve(search_query)

        if not articles:
            return (
                "К сожалению, я не нашёл релевантных статей по вашему вопросу. "
                "Попробуйте переформулировать вопрос или уточнить, "
                "какой именно кодекс вас интересует."
            )

        # Step 3: Generation
        context = self._format_context(articles)
        sources_list = "\n".join(
            f"- {a['codex']}, ст. {a['article_num']}. {a['article_title']}"
            for a in articles
        )

        system_prompt = (
            "Ты — юридический AI-ассистент по российскому законодательству. "
            "Отвечай на вопросы СТРОГО на основе предоставленных статей кодексов. "
            "Правила:\n"
            "1. Каждое утверждение подкрепляй ссылкой на конкретную статью: "
            "(ст. N Кодекса)\n"
            "2. Если в предоставленных статьях нет ответа — честно скажи об этом\n"
            "3. НЕ выдумывай статьи или нормы, которых нет в контексте\n"
            "4. Отвечай понятным языком, но с юридической точностью\n"
            "5. В конце ответа выведи список использованных источников\n"
            "6. Если вопрос не юридический — вежливо скажи, что специализируешься "
            "только на российском законодательстве"
        )

        gen_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Найденные статьи кодексов:\n\n{context}\n\n"
                    f"---\n\nВопрос пользователя: {user_message}"
                ),
            },
        ]

        # Streaming response
        response = self.openai_client.chat.completions.create(
            model=self.valves.LLM_MODEL,
            messages=gen_messages,
            temperature=0.1,
            max_tokens=2000,
            stream=True,
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
