"""
Автоматическая оценка Legal AI Pipeline.

Прогоняет вопросы из eval_dataset.jsonl через pipeline,
сравнивает ответы с эталонными статьями.

Метрики:
- Article Recall: % правильно найденных статей
- Hallucination Rate: % ответов с несуществующими ссылками
- Answer Quality: LLM-as-judge оценка (1-5)
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def load_dataset(path: str = None) -> list[dict]:
    """Загружает eval dataset."""
    if path is None:
        path = Path(__file__).parent / "eval_dataset.jsonl"
    else:
        path = Path(path)

    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Loaded {len(items)} eval questions")
    return items


def search_qdrant(query: str, qdrant_client, openai_client, collection: str, top_k: int = 7) -> list[dict]:
    """Поиск в Qdrant."""
    response = openai_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        input=query,
    )
    embedding = response.data[0].embedding

    results = qdrant_client.query_points(
        collection_name=collection,
        query=embedding,
        limit=top_k,
    )

    return [
        {
            "codex_id": hit.payload.get("codex_id", ""),
            "article_num": hit.payload.get("article_num", ""),
            "article_title": hit.payload.get("article_title", ""),
            "score": hit.score,
        }
        for hit in results.points
    ]


def compute_article_recall(retrieved: list[dict], expected: list[str]) -> float:
    """Считает долю ожидаемых статей, найденных в retrieved."""
    if not expected:
        return 1.0

    retrieved_keys = set()
    for r in retrieved:
        key = f"{r['codex_id']}:{r['article_num']}"
        retrieved_keys.add(key)

    found = sum(1 for e in expected if e in retrieved_keys)
    return found / len(expected)


def extract_article_refs(text: str) -> list[str]:
    """Извлекает ссылки на статьи из текста ответа."""
    # Паттерн: ст. 123, статья 123, ст.123
    pattern = r"(?:ст(?:атья|\.)\s*)(\d+(?:\.\d+)?)"
    return re.findall(pattern, text, re.IGNORECASE)


def main():
    from openai import OpenAI
    from qdrant_client import QdrantClient

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env")
        sys.exit(1)
    if not os.getenv("QDRANT_URL"):
        print("ERROR: Set QDRANT_URL in .env")
        sys.exit(1)

    openai_client = OpenAI()
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    collection = os.getenv("QDRANT_COLLECTION", "russian_law")

    dataset = load_dataset()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    total_recall = 0.0
    total_items = 0
    results = []

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_articles = item.get("expected_articles", [])

        print(f"[{i + 1}/{len(dataset)}] {question[:60]}...")

        # Поиск
        retrieved = search_qdrant(question, qdrant_client, openai_client, collection)
        recall = compute_article_recall(retrieved, expected_articles)
        total_recall += recall
        total_items += 1

        result = {
            "question": question,
            "expected": expected_articles,
            "retrieved": [f"{r['codex_id']}:{r['article_num']}" for r in retrieved[:5]],
            "recall": recall,
            "top_scores": [r["score"] for r in retrieved[:3]],
        }
        results.append(result)

        if recall < 1.0:
            print(f"  ⚠ Recall={recall:.2f} | Expected: {expected_articles}")
            print(f"    Retrieved: {result['retrieved'][:5]}")

    # Сохраняем результаты
    avg_recall = total_recall / total_items if total_items > 0 else 0
    summary = {
        "total_questions": total_items,
        "average_recall": round(avg_recall, 4),
        "recall_at_100": sum(1 for r in results if r["recall"] == 1.0),
        "recall_at_0": sum(1 for r in results if r["recall"] == 0.0),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    out_file = results_dir / "eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== Eval Results ===")
    print(f"Questions: {summary['total_questions']}")
    print(f"Avg Article Recall: {summary['average_recall']:.2%}")
    print(f"Perfect Recall (100%): {summary['recall_at_100']}")
    print(f"Zero Recall (0%): {summary['recall_at_0']}")
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
