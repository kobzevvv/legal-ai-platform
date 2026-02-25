"""
Парсинг скачанных HTML кодексов → JSON.

Каждая статья = один объект:
{
    "codex": "Гражданский кодекс РФ (часть 1)",
    "codex_id": "gk1",
    "chapter": "Глава 1. ...",
    "article_num": "21",
    "article_title": "Дееспособность гражданина",
    "text": "1. Способность гражданина...",
    "url": "https://www.consultant.ru/document/cons_doc_LAW_5142/hash/"
}

Структура HTML подстраниц consultant.ru:
- <div class="doc-style"> содержит "ГК РФ Статья 21. Название"
- <p> содержат пункты текста статьи
- Каждая подстраница = одна статья (или группа связанных)
"""

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

CODEX_NAMES = {
    "gk1": "Гражданский кодекс РФ (часть 1)",
    "gk2": "Гражданский кодекс РФ (часть 2)",
    "gk3": "Гражданский кодекс РФ (часть 3)",
    "gk4": "Гражданский кодекс РФ (часть 4)",
    "uk": "Уголовный кодекс РФ",
    "tk": "Трудовой кодекс РФ",
    "nk1": "Налоговый кодекс РФ (часть 1)",
    "nk2": "Налоговый кодекс РФ (часть 2)",
    "sk": "Семейный кодекс РФ",
    "jk": "Жилищный кодекс РФ",
    "koap": "КоАП РФ",
}

CODEX_URLS = {
    "gk1": "https://www.consultant.ru/document/cons_doc_LAW_5142/",
    "gk2": "https://www.consultant.ru/document/cons_doc_LAW_9027/",
    "gk3": "https://www.consultant.ru/document/cons_doc_LAW_34154/",
    "gk4": "https://www.consultant.ru/document/cons_doc_LAW_64629/",
    "uk": "https://www.consultant.ru/document/cons_doc_LAW_10699/",
    "tk": "https://www.consultant.ru/document/cons_doc_LAW_34683/",
    "nk1": "https://www.consultant.ru/document/cons_doc_LAW_19671/",
    "nk2": "https://www.consultant.ru/document/cons_doc_LAW_28165/",
    "sk": "https://www.consultant.ru/document/cons_doc_LAW_8982/",
    "jk": "https://www.consultant.ru/document/cons_doc_LAW_51057/",
    "koap": "https://www.consultant.ru/document/cons_doc_LAW_34661/",
}

RAW_DIR = Path(__file__).parent / "data" / "raw"
PARSED_DIR = Path(__file__).parent / "data" / "parsed"

# Паттерн заголовка статьи в подстранице
# "ГК РФ Статья 21. Дееспособность гражданина" или "Статья 21. ..."
ARTICLE_HEADER = re.compile(
    r"(?:.*?\s)?Статья\s+(\d+(?:\.\d+)?)\s*\.\s*(.+)", re.IGNORECASE
)

# Слова-маркеры для фильтрации мусора
SKIP_MARKERS = [
    "консультантплюс", "подготовлен", "изменяющих документов",
    "путеводител", "судебная практика", "перспективы и риски",
    "примечание.", "(в ред.", "открыть полный текст",
    "готовые решения", "см. также", "позиции высших судов",
    "истец (заявитель)", "ответчик хочет", "истец хочет",
    "что нужно доказать", "какие обстоятельства",
    "рекомендации по составлению", "как составить",
]


def clean_text(text: str) -> str:
    """Очищает текст от лишних пробелов."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\xa0", " ")
    return text


def extract_chapter_from_index(codex_id: str, page_hash: str) -> str:
    """Пытается определить главу статьи из индекса (оглавления)."""
    index_file = RAW_DIR / codex_id / "index.html"
    if not index_file.exists():
        return ""

    # Кэш для index HTML
    if not hasattr(extract_chapter_from_index, "_cache"):
        extract_chapter_from_index._cache = {}

    if codex_id not in extract_chapter_from_index._cache:
        html = index_file.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        # Строим маппинг hash → chapter
        chapter_map = {}
        current_chapter = ""

        for el in soup.find_all(["a", "p", "div", "h2", "h3", "h4"]):
            text = el.get_text(strip=True)

            # Заголовки глав/разделов
            chapter_match = re.match(
                r"(Часть|Раздел|Подраздел|Глава)\s+[\dIVXLCDM]+[\.\s]*(.*)",
                text, re.IGNORECASE
            )
            if chapter_match:
                current_chapter = clean_text(text)

            # Ссылки на подстраницы
            if el.name == "a" and el.get("href"):
                href = el["href"]
                h = href.rstrip("/").split("/")[-1]
                if len(h) > 10:  # hash
                    chapter_map[h] = current_chapter

        extract_chapter_from_index._cache[codex_id] = chapter_map

    return extract_chapter_from_index._cache[codex_id].get(page_hash, "")


def parse_article_page(html: str, codex_id: str, page_hash: str) -> list[dict]:
    """Парсит одну подстраницу статьи."""
    soup = BeautifulSoup(html, "lxml")
    articles = []

    content = soup.find("div", class_="document-page__content")
    if not content:
        return articles

    # Собираем все <p> напрямую
    all_paragraphs = content.find_all("p", recursive=True)

    current_num = None
    current_title = ""
    current_parts = []

    def save_article():
        nonlocal current_num, current_title, current_parts
        if current_num and current_parts:
            text = clean_text(" ".join(current_parts))
            if len(text) > 20:
                chapter = extract_chapter_from_index(codex_id, page_hash)
                base_url = CODEX_URLS.get(codex_id, "")
                articles.append({
                    "codex": CODEX_NAMES.get(codex_id, codex_id),
                    "codex_id": codex_id,
                    "chapter": chapter,
                    "article_num": current_num,
                    "article_title": current_title,
                    "text": text,
                    "url": f"{base_url}{page_hash}/",
                })

    for p in all_paragraphs:
        # Пропускаем <p> внутри doc-edit, doc-insert (редакции и примечания КП)
        parent = p.parent
        if parent:
            parent_classes = parent.get("class", [])
            if any(c in parent_classes for c in ["doc-edit", "document__edit",
                                                   "doc-insert", "document__insert"]):
                continue

        text = p.get_text(strip=True)
        if not text or len(text) < 3:
            continue

        # Пропускаем мусор
        text_lower = text.lower()
        if any(marker in text_lower for marker in SKIP_MARKERS):
            continue

        # Проверяем заголовок статьи (в div.doc-style)
        header_match = ARTICLE_HEADER.match(text)
        if header_match:
            save_article()
            current_num = header_match.group(1)
            current_title = clean_text(header_match.group(2))
            current_parts = []
            continue

        # Текст статьи
        if current_num:
            current_parts.append(text)

    save_article()
    return articles


def parse_codex(codex_id: str) -> list[dict]:
    """Парсит все HTML файлы одного кодекса."""
    codex_dir = RAW_DIR / codex_id
    if not codex_dir.exists():
        print(f"  [skip] {codex_id}: directory not found")
        return []

    all_articles = []
    seen = set()

    html_files = sorted(codex_dir.glob("*.html"))
    for html_file in html_files:
        if html_file.name == "index.html":
            continue

        html = html_file.read_text(encoding="utf-8", errors="replace")
        page_hash = html_file.stem
        articles = parse_article_page(html, codex_id, page_hash)

        for art in articles:
            key = (art["codex_id"], art["article_num"])
            if key not in seen:
                seen.add(key)
                all_articles.append(art)

    # Сортировка по номеру статьи
    def sort_key(a):
        parts = a["article_num"].split(".")
        return tuple(int(p) for p in parts)

    try:
        all_articles.sort(key=sort_key)
    except (ValueError, TypeError):
        pass

    return all_articles


def main():
    print("=== Parsing Codexes: HTML → JSON ===\n")
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    summary = {}

    for codex_id, codex_name in CODEX_NAMES.items():
        print(f"Parsing {codex_name}...")
        articles = parse_codex(codex_id)

        if articles:
            out_file = PARSED_DIR / f"{codex_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"  → {len(articles)} articles → {out_file.name}")
        else:
            print(f"  → 0 articles (no HTML files downloaded yet?)")

        summary[codex_id] = len(articles)
        total += len(articles)

    # Объединённый файл
    all_articles = []
    for codex_id in CODEX_NAMES:
        json_file = PARSED_DIR / f"{codex_id}.json"
        if json_file.exists():
            with open(json_file, encoding="utf-8") as f:
                all_articles.extend(json.load(f))

    combined_file = PARSED_DIR / "all_codexes.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    for codex_id, count in summary.items():
        print(f"  {codex_id}: {count} articles")
    print(f"  TOTAL: {total} articles → {combined_file.name}")

    # Показать примеры
    if all_articles:
        print(f"\n=== Sample articles ===")
        for art in all_articles[:3]:
            print(f"  [{art['codex_id']}] Ст. {art['article_num']}. {art['article_title']}")
            print(f"    {art['text'][:100]}...")
            print()


if __name__ == "__main__":
    main()
