"""
Создание embeddings для статей и загрузка в Qdrant Cloud.

Использует OpenAI text-embedding-3-small (1536 dim, дёшево и быстро).
Каждая статья = один вектор с metadata.
"""

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

PARSED_DIR = Path(__file__).parent / "data" / "parsed"
COLLECTION = os.getenv("QDRANT_COLLECTION", "russian_law")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
BATCH_SIZE = 100  # OpenAI embeddings batch limit


def load_articles() -> list[dict]:
    """Загружает все статьи из combined JSON."""
    combined = PARSED_DIR / "all_codexes.json"
    if not combined.exists():
        print("ERROR: all_codexes.json not found. Run parse_codexes.py first.")
        sys.exit(1)

    with open(combined, encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Loaded {len(articles)} articles")
    return articles


def create_embedding_text(article: dict) -> str:
    """Формирует текст для embedding из статьи."""
    parts = [
        article["codex"],
        f"Статья {article['article_num']}. {article['article_title']}",
        article["text"],
    ]
    if article.get("chapter"):
        parts.insert(1, article["chapter"])
    return "\n".join(parts)


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Получает embeddings для батча текстов."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def _upload_with_retry(client, collection: str, points: list, max_retries: int = 3):
    """Upload points to Qdrant with retry on timeout."""
    import time
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection, points=points, timeout=120)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"\n  Upload failed (attempt {attempt + 1}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def main():
    # Проверяем ключи
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env")
        sys.exit(1)
    if not os.getenv("QDRANT_URL"):
        print("ERROR: QDRANT_URL not set. Add it to .env")
        sys.exit(1)

    openai_client = OpenAI()
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    articles = load_articles()

    # Создаём коллекцию
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION not in collections:
        print(f"Creating collection '{COLLECTION}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print(f"Collection '{COLLECTION}' already exists")

    # Embed & upload батчами
    print(f"\nEmbedding + uploading ({EMBEDDING_MODEL}, batch={BATCH_SIZE})...")
    points = []

    for i in tqdm(range(0, len(articles), BATCH_SIZE), desc="Batches"):
        batch = articles[i : i + BATCH_SIZE]
        texts = [create_embedding_text(a) for a in batch]

        # Truncate long texts (OpenAI limit ~8191 tokens ≈ ~24000 chars)
        # Use 7000 chars per text to stay safe within batch limits
        texts = [t[:7000] for t in texts]

        try:
            embeddings = embed_batch(openai_client, texts)
        except Exception as e:
            # If batch still too long, process one by one with shorter truncation
            print(f"\n  Batch {i//BATCH_SIZE} failed: {e}")
            print("  Retrying individually with shorter texts...")
            embeddings = []
            for t in texts:
                try:
                    emb = embed_batch(openai_client, [t[:4000]])
                    embeddings.append(emb[0])
                except Exception as e2:
                    print(f"    Single embed failed, truncating more: {e2}")
                    emb = embed_batch(openai_client, [t[:2000]])
                    embeddings.append(emb[0])

        for article, embedding in zip(batch, embeddings):
            # Deterministic ID based on codex_id + article_num (idempotent)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                                       f"{article['codex_id']}:{article['article_num']}"))
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "codex": article["codex"],
                    "codex_id": article["codex_id"],
                    "chapter": article.get("chapter", ""),
                    "article_num": article["article_num"],
                    "article_title": article["article_title"],
                    "text": article["text"],
                    "url": article.get("url", ""),
                },
            )
            points.append(point)

        # Upload every BATCH_SIZE points (100) — small batches for free tier
        if len(points) >= BATCH_SIZE:
            _upload_with_retry(qdrant_client, COLLECTION, points)
            points = []

    # Upload remaining
    if points:
        _upload_with_retry(qdrant_client, COLLECTION, points)

    # Проверка
    info = qdrant_client.get_collection(COLLECTION)
    print(f"\n=== Done ===")
    print(f"Collection: {COLLECTION}")
    print(f"Points: {info.points_count}")

    # Тестовый поиск
    print("\nTest search: 'увольнение по собственному желанию'")
    test_embedding = embed_batch(openai_client, ["увольнение по собственному желанию"])[0]
    results = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=test_embedding,
        limit=3,
    )
    for hit in results.points:
        payload = hit.payload
        print(f"  [{hit.score:.3f}] {payload['codex']} — "
              f"Ст. {payload['article_num']}. {payload['article_title']}")


if __name__ == "__main__":
    main()
