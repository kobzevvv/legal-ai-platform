"""
Скачивание 7 основных кодексов РФ с consultant.ru.

Каждый кодекс скачивается постранично (consultant.ru разбивает на страницы).
Результат: HTML файлы в data/raw/<codex_name>/
"""

import os
import time
import requests
from pathlib import Path

# Кодексы и их URL на consultant.ru
CODEXES = {
    "gk1": {
        "name": "Гражданский кодекс РФ (часть 1)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_5142/",
    },
    "gk2": {
        "name": "Гражданский кодекс РФ (часть 2)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_9027/",
    },
    "gk3": {
        "name": "Гражданский кодекс РФ (часть 3)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_34154/",
    },
    "gk4": {
        "name": "Гражданский кодекс РФ (часть 4)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_64629/",
    },
    "uk": {
        "name": "Уголовный кодекс РФ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/",
    },
    "tk": {
        "name": "Трудовой кодекс РФ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_34683/",
    },
    "nk1": {
        "name": "Налоговый кодекс РФ (часть 1)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_19671/",
    },
    "nk2": {
        "name": "Налоговый кодекс РФ (часть 2)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_28165/",
    },
    "sk": {
        "name": "Семейный кодекс РФ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_8982/",
    },
    "jk": {
        "name": "Жилищный кодекс РФ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_51057/",
    },
    "koap": {
        "name": "КоАП РФ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_34661/",
    },
}

RAW_DIR = Path(__file__).parent / "data" / "raw"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "ru-RU,ru;q=0.9",
}


def download_codex(codex_id: str, info: dict) -> None:
    """Скачивает главную страницу кодекса (оглавление со ссылками на статьи)."""
    out_dir = RAW_DIR / codex_id
    out_dir.mkdir(parents=True, exist_ok=True)

    index_file = out_dir / "index.html"
    if index_file.exists():
        print(f"  [skip] {codex_id}/index.html already exists")
        return

    print(f"  Downloading {info['name']}...")
    resp = requests.get(info["url"], headers=HEADERS, timeout=30)
    resp.raise_for_status()
    index_file.write_text(resp.text, encoding="utf-8")
    print(f"  Saved {codex_id}/index.html ({len(resp.text):,} chars)")


def download_article_pages(codex_id: str, info: dict) -> None:
    """Скачивает отдельные страницы статей из оглавления."""
    from bs4 import BeautifulSoup

    out_dir = RAW_DIR / codex_id
    index_file = out_dir / "index.html"
    if not index_file.exists():
        print(f"  [skip] No index for {codex_id}")
        return

    html = index_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    # Собираем ссылки на статьи из оглавления
    # URL вида https://www.consultant.ru/document/cons_doc_LAW_34683/
    # href вида /document/cons_doc_LAW_34683/hash/
    from urllib.parse import urlparse
    parsed_url = urlparse(info["url"])
    doc_path = parsed_url.path.rstrip("/")  # /document/cons_doc_LAW_34683

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Ссылки на подстраницы статей: /document/cons_doc_LAW_XXXX/hash/
        if href.startswith(doc_path + "/") and href.rstrip("/") != doc_path:
            full_url = f"https://www.consultant.ru{href}"
            page_id = href.rstrip("/").split("/")[-1]
            if page_id and len(page_id) > 5:  # hash-идентификаторы длинные
                links.append((page_id, full_url))

    # Дедупликация
    seen = set()
    unique_links = []
    for page_id, url in links:
        if page_id not in seen:
            seen.add(page_id)
            unique_links.append((page_id, url))

    print(f"  Found {len(unique_links)} article pages for {codex_id}")

    for i, (page_id, url) in enumerate(unique_links):
        page_file = out_dir / f"{page_id}.html"
        if page_file.exists():
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            page_file.write_text(resp.text, encoding="utf-8")
            if (i + 1) % 50 == 0:
                print(f"    {codex_id}: {i + 1}/{len(unique_links)} pages")
            time.sleep(0.2)  # Вежливая пауза
        except Exception as e:
            print(f"    [error] {page_id}: {e}")
            time.sleep(1)


def main():
    print("=== Downloading Russian Legal Codexes ===\n")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Шаг 1: скачать индексные страницы
    print("Step 1: Downloading index pages...")
    for codex_id, info in CODEXES.items():
        download_codex(codex_id, info)
        time.sleep(1)

    # Шаг 2: скачать страницы статей
    print("\nStep 2: Downloading article pages...")
    for codex_id, info in CODEXES.items():
        download_article_pages(codex_id, info)
        time.sleep(1)

    print("\n=== Done ===")
    # Статистика
    for codex_id in CODEXES:
        d = RAW_DIR / codex_id
        if d.exists():
            count = len(list(d.glob("*.html")))
            print(f"  {codex_id}: {count} files")


if __name__ == "__main__":
    main()
