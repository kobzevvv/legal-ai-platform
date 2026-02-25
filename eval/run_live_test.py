"""
Live testing of Legal RAG Pipeline via Pipelines API.

Generates questions independently via LLM (no knowledge of our system),
then generates adversarial questions from a powerful model that knows
our architecture, and runs everything through the pipeline.

Tracks token usage and cost per question. Outputs HTML report.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

PIPELINE_URL = "http://localhost:9099"
PIPELINE_API_KEY = "0p3n-w3bu!"
PIPELINE_MODEL = "legal_rag"

# Pricing per 1M tokens (USD) — Feb 2026
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Pipeline internally uses gpt-4o-mini for keyword expansion + query rewriting
    # and whatever LLM_MODEL is set for generation (default gpt-4o-mini)
    "pipeline_estimate": {"input": 0.15, "output": 0.60},
}

openai_client = OpenAI()


class TokenTracker:
    """Track token usage across all API calls."""

    def __init__(self):
        self.calls = []

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, label: str = ""):
        cost_input = prompt_tokens * PRICING.get(model, PRICING["gpt-4o-mini"])["input"] / 1_000_000
        cost_output = completion_tokens * PRICING.get(model, PRICING["gpt-4o-mini"])["output"] / 1_000_000
        self.calls.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost_input + cost_output,
            "label": label,
        })

    @property
    def total_tokens(self):
        return sum(c["total_tokens"] for c in self.calls)

    @property
    def total_cost(self):
        return sum(c["cost_usd"] for c in self.calls)

    def summary_by_label(self):
        labels = set(c["label"] for c in self.calls)
        result = {}
        for label in sorted(labels):
            calls = [c for c in self.calls if c["label"] == label]
            result[label] = {
                "calls": len(calls),
                "total_tokens": sum(c["total_tokens"] for c in calls),
                "cost_usd": sum(c["cost_usd"] for c in calls),
            }
        return result


tracker = TokenTracker()


def generate_independent_questions(n: int = 20) -> list[str]:
    """Generate questions from an independent LLM with zero knowledge of our system."""
    print(f"\n{'='*60}")
    print(f"Step 1: Generating {n} independent questions...")
    print(f"{'='*60}")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты генератор тестовых вопросов. Сгенерируй ровно "
                    f"{n} вопросов по российскому законодательству, "
                    "которые обычный человек мог бы задать юристу. "
                    "Вопросы должны быть разнообразными: трудовое право, "
                    "уголовное, гражданское, семейное, жилищное, "
                    "налоговое, административное. "
                    "Используй БЫТОВОЙ язык, как спрашивают обычные люди, "
                    "не юристы. Без нумерации. Один вопрос на строку. "
                    "Ничего кроме вопросов не пиши."
                ),
            },
            {
                "role": "user",
                "content": f"Сгенерируй {n} вопросов.",
            },
        ],
        temperature=0.9,
        max_tokens=2000,
    )

    tracker.record(
        "gpt-4o-mini",
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        "question_generation",
    )

    questions = [
        q.strip().lstrip("0123456789.-) ")
        for q in response.choices[0].message.content.strip().split("\n")
        if q.strip() and "?" in q
    ]
    questions = questions[:n]

    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    return questions


def generate_adversarial_questions(n: int = 10) -> list[str]:
    """Generate adversarial questions from a powerful model that knows our architecture."""
    print(f"\n{'='*60}")
    print(f"Step 2: Generating {n} adversarial questions (gpt-4o)...")
    print(f"{'='*60}")

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — эксперт по тестированию AI-систем и российскому праву. "
                    "Твоя задача: сгенерировать вопросы, которые ПОДЛОВЯТ "
                    "описанную ниже систему на ошибках. "
                    "НЕ объясняй, в чём будет ошибка. Просто дай вопросы. "
                    "Пиши бытовым языком, как спросил бы обычный человек."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Вот архитектура системы, которую нужно подловить:\n\n"
                    "1. База знаний: 11 российских кодексов (ГК части 1-4, УК, ТК, "
                    "НК части 1-2, СК, ЖК, КоАП), спарсенных с consultant.ru. "
                    "Каждая статья — отдельный вектор в Qdrant.\n"
                    "2. Embedding: OpenAI text-embedding-3-small (1536 dim), "
                    "текст статьи обрезается до 7000 символов.\n"
                    "3. Поиск: косинусное сходство, top-7 результатов, порог 0.3.\n"
                    "4. Pipeline: бытовой вопрос → keyword expansion (gpt-4o-mini "
                    "генерирует юридические термины) → query rewriting → "
                    "vector search → генерация ответа с цитированием статей.\n"
                    "5. НЕ включены: федеральные законы (кроме кодексов), "
                    "постановления Пленума ВС, подзаконные акты, "
                    "судебная практика, региональное законодательство, "
                    "УПК, ГПК, АПК, КАС, ЗК, ЛК, ВК, ВозК.\n\n"
                    f"Сгенерируй ровно {n} вопросов на русском бытовым языком, "
                    "которые подловят эту систему. Один вопрос на строку. "
                    "Без нумерации, без пояснений."
                ),
            },
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    tracker.record(
        "gpt-4o",
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        "question_generation",
    )

    questions = [
        q.strip().lstrip("0123456789.-) ")
        for q in response.choices[0].message.content.strip().split("\n")
        if q.strip() and len(q.strip()) > 10
    ]
    questions = questions[:n]

    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    return questions


def run_pipeline(question: str) -> dict:
    """Send a question to the pipeline and get the response with token tracking."""
    pipeline_client = OpenAI(
        base_url=f"{PIPELINE_URL}/v1",
        api_key=PIPELINE_API_KEY,
    )

    try:
        response = pipeline_client.chat.completions.create(
            model=PIPELINE_MODEL,
            messages=[{"role": "user", "content": question}],
            temperature=0.1,
            max_tokens=2000,
            stream=False,
        )
        answer = response.choices[0].message.content

        # Pipeline internally makes ~3 LLM calls:
        # 1. keyword expansion (~150 tokens out)
        # 2. query rewriting (~200 tokens out)
        # 3. generation (~800 tokens out)
        # Plus embedding call
        # Estimate: ~1500 input + ~1200 output tokens per question
        usage = response.usage
        if usage:
            tracker.record(
                "pipeline_estimate",
                usage.prompt_tokens or 1500,
                usage.completion_tokens or 1200,
                "pipeline",
            )
        else:
            tracker.record("pipeline_estimate", 1500, 1200, "pipeline")

        return {"answer": answer, "error": False}
    except Exception as e:
        return {"answer": f"[ERROR] {e}", "error": True}


def run_tests(questions: list[str], label: str) -> list[dict]:
    """Run all questions through the pipeline."""
    print(f"\n{'='*60}")
    print(f"Running {len(questions)} {label} questions through pipeline...")
    print(f"{'='*60}")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n--- [{i}/{len(questions)}] {q}")
        start = time.time()
        resp = run_pipeline(q)
        elapsed = time.time() - start

        answer = resp["answer"]
        # Truncate for console
        preview = answer.replace("\n", " ")[:200]
        print(f"    [{elapsed:.1f}s] {preview}...")

        # Detect if answer cites articles
        has_refs = "ст." in answer.lower() or "статья" in answer.lower()
        no_articles = "не нашёл релевантных" in answer

        results.append({
            "question": q,
            "answer": answer,
            "time_s": round(elapsed, 1),
            "category": label,
            "has_refs": has_refs,
            "no_articles": no_articles,
            "error": resp["error"],
        })

    return results


def generate_html_report(results: list[dict], out_path: Path):
    """Generate a beautiful HTML report with all results and analytics."""

    total = len(results)
    errors = sum(1 for r in results if r["error"])
    no_articles = sum(1 for r in results if r["no_articles"])
    has_refs = sum(1 for r in results if r["has_refs"])
    avg_time = sum(r["time_s"] for r in results) / total if total else 0
    total_time = sum(r["time_s"] for r in results)

    # Per-category stats
    categories = {}
    for cat in ["independent", "adversarial"]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        n = len(cat_results)
        categories[cat] = {
            "total": n,
            "errors": sum(1 for r in cat_results if r["error"]),
            "no_articles": sum(1 for r in cat_results if r["no_articles"]),
            "has_refs": sum(1 for r in cat_results if r["has_refs"]),
            "avg_time": sum(r["time_s"] for r in cat_results) / n,
        }

    # Token/cost summary
    token_summary = tracker.summary_by_label()
    cost_per_question = tracker.total_cost / total if total else 0

    # Build question rows
    rows_html = ""
    for i, r in enumerate(results, 1):
        status = "error" if r["error"] else ("warning" if r["no_articles"] else "success")
        status_icon = "&#x274C;" if r["error"] else ("&#x26A0;" if r["no_articles"] else "&#x2705;")
        badge = r["category"]
        badge_class = "badge-independent" if r["category"] == "independent" else "badge-adversarial"

        # Escape HTML in answer
        answer_escaped = (
            r["answer"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

        rows_html += f"""
        <div class="result-card {status}">
            <div class="result-header">
                <span class="result-num">#{i}</span>
                <span class="badge {badge_class}">{badge}</span>
                <span class="result-time">{r['time_s']}s</span>
                <span class="result-status">{status_icon}</span>
            </div>
            <div class="result-question">{r['question']}</div>
            <div class="result-answer">{answer_escaped}</div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Legal AI — Test Report</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; line-height: 1.6;
    padding: 2rem; max-width: 1200px; margin: 0 auto;
}}
h1 {{ font-size: 2rem; margin-bottom: 0.5rem; color: #f8fafc; }}
.subtitle {{ color: #94a3b8; margin-bottom: 2rem; font-size: 0.95rem; }}
.stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem; margin-bottom: 2rem;
}}
.stat-card {{
    background: #1e293b; border-radius: 12px; padding: 1.25rem;
    border: 1px solid #334155;
}}
.stat-value {{ font-size: 2rem; font-weight: 700; color: #f8fafc; }}
.stat-label {{ color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem; }}
.stat-value.green {{ color: #4ade80; }}
.stat-value.yellow {{ color: #facc15; }}
.stat-value.red {{ color: #f87171; }}
.stat-value.blue {{ color: #60a5fa; }}

.section-title {{
    font-size: 1.3rem; margin: 2rem 0 1rem; color: #f8fafc;
    border-bottom: 1px solid #334155; padding-bottom: 0.5rem;
}}

.cost-table {{ width: 100%; border-collapse: collapse; margin-bottom: 2rem; }}
.cost-table th, .cost-table td {{
    text-align: left; padding: 0.75rem 1rem;
    border-bottom: 1px solid #334155;
}}
.cost-table th {{ color: #94a3b8; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
.cost-table td {{ color: #e2e8f0; }}
.cost-table tr:hover {{ background: #1e293b; }}
.cost-highlight {{ color: #4ade80; font-weight: 700; }}

.category-section {{ margin-bottom: 2rem; }}
.category-header {{
    display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;
}}
.category-stats {{ display: flex; gap: 1.5rem; color: #94a3b8; font-size: 0.9rem; }}
.category-stats span {{ display: flex; align-items: center; gap: 0.3rem; }}

.result-card {{
    background: #1e293b; border-radius: 10px; padding: 1.25rem;
    margin-bottom: 1rem; border-left: 4px solid #334155;
    transition: transform 0.1s;
}}
.result-card:hover {{ transform: translateX(4px); }}
.result-card.success {{ border-left-color: #4ade80; }}
.result-card.warning {{ border-left-color: #facc15; }}
.result-card.error {{ border-left-color: #f87171; }}

.result-header {{
    display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;
}}
.result-num {{ color: #64748b; font-weight: 700; font-size: 0.85rem; }}
.badge {{
    font-size: 0.7rem; padding: 2px 8px; border-radius: 999px;
    font-weight: 600; text-transform: uppercase;
}}
.badge-independent {{ background: #1e3a5f; color: #60a5fa; }}
.badge-adversarial {{ background: #4a1d1d; color: #f87171; }}
.result-time {{ color: #64748b; font-size: 0.85rem; margin-left: auto; }}
.result-status {{ font-size: 1.1rem; }}

.result-question {{
    font-weight: 600; color: #f8fafc; margin-bottom: 0.75rem;
    font-size: 1.05rem;
}}
.result-answer {{
    color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;
    max-height: 300px; overflow-y: auto;
    padding: 0.75rem; background: #0f172a; border-radius: 8px;
}}
</style>
</head>
<body>

<h1>Legal AI — Live Test Report</h1>
<p class="subtitle">
    {datetime.now().strftime('%d.%m.%Y %H:%M')} &middot;
    {total} questions &middot;
    Pipeline: keyword expansion → query rewriting → Qdrant retrieval → generation
</p>

<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{total}</div>
        <div class="stat-label">Total Questions</div>
    </div>
    <div class="stat-card">
        <div class="stat-value green">{has_refs}</div>
        <div class="stat-label">With Article Citations ({has_refs/total*100:.0f}%)</div>
    </div>
    <div class="stat-card">
        <div class="stat-value yellow">{no_articles}</div>
        <div class="stat-label">No Articles Found</div>
    </div>
    <div class="stat-card">
        <div class="stat-value red">{errors}</div>
        <div class="stat-label">Errors</div>
    </div>
    <div class="stat-card">
        <div class="stat-value blue">{avg_time:.1f}s</div>
        <div class="stat-label">Avg Response Time</div>
    </div>
    <div class="stat-card">
        <div class="stat-value green">${cost_per_question:.4f}</div>
        <div class="stat-label">Cost per Question</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{tracker.total_tokens:,}</div>
        <div class="stat-label">Total Tokens Used</div>
    </div>
    <div class="stat-card">
        <div class="stat-value green">${tracker.total_cost:.4f}</div>
        <div class="stat-label">Total Cost (USD)</div>
    </div>
</div>

<h2 class="section-title">Cost Breakdown</h2>
<table class="cost-table">
<thead>
    <tr><th>Stage</th><th>API Calls</th><th>Tokens</th><th>Cost (USD)</th></tr>
</thead>
<tbody>
{"".join(f'''
    <tr>
        <td>{label}</td>
        <td>{data["calls"]}</td>
        <td>{data["total_tokens"]:,}</td>
        <td>${data["cost_usd"]:.4f}</td>
    </tr>
''' for label, data in token_summary.items())}
    <tr style="border-top: 2px solid #475569;">
        <td><strong>TOTAL</strong></td>
        <td><strong>{len(tracker.calls)}</strong></td>
        <td><strong>{tracker.total_tokens:,}</strong></td>
        <td class="cost-highlight"><strong>${tracker.total_cost:.4f}</strong></td>
    </tr>
</tbody>
</table>

<h2 class="section-title">Results by Category</h2>

{"".join(f'''
<div class="category-section">
    <div class="category-header">
        <h3>{"Independent Questions" if cat == "independent" else "Adversarial Questions (gpt-4o)"}</h3>
        <div class="category-stats">
            <span>&#x2705; {data["has_refs"]}/{data["total"]}</span>
            <span>&#x26A0; {data["no_articles"]}</span>
            <span>&#x23F1; {data["avg_time"]:.1f}s avg</span>
        </div>
    </div>
</div>
''' for cat, data in categories.items())}

<h2 class="section-title">All Questions &amp; Answers</h2>

{rows_html}

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML report saved to {out_path}")


def main():
    # Step 1: Generate independent questions
    independent_qs = generate_independent_questions(20)

    # Step 2: Generate adversarial questions
    adversarial_qs = generate_adversarial_questions(10)

    # Save all questions
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    all_questions = {
        "independent": independent_qs,
        "adversarial": adversarial_qs,
    }
    with open(out_dir / "test_questions.json", "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\nSaved questions to {out_dir / 'test_questions.json'}")

    # Step 3: Run tests
    results = []
    results.extend(run_tests(independent_qs, "independent"))
    results.extend(run_tests(adversarial_qs, "adversarial"))

    # Step 4: Save JSON results
    with open(out_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {out_dir / 'test_results.json'}")

    # Step 5: Generate HTML report
    generate_html_report(results, out_dir / "test_report.html")

    # Step 6: Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = len(results)
    has_refs = sum(1 for r in results if r["has_refs"])
    no_art = sum(1 for r in results if r["no_articles"])
    errs = sum(1 for r in results if r["error"])
    avg_t = sum(r["time_s"] for r in results) / total
    print(f"  Total:          {total} questions")
    print(f"  With citations: {has_refs} ({has_refs/total*100:.0f}%)")
    print(f"  No articles:    {no_art}")
    print(f"  Errors:         {errs}")
    print(f"  Avg time:       {avg_t:.1f}s")
    print(f"  Total tokens:   {tracker.total_tokens:,}")
    print(f"  Total cost:     ${tracker.total_cost:.4f}")
    print(f"  Cost/question:  ${tracker.total_cost / total:.4f}")


if __name__ == "__main__":
    main()
