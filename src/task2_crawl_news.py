"""Task 2 - Crawl 5 bai tu van phap luat lao dong cua Bo Tu phap.

Moi bai viet duoc luu thanh mot file JSON trong ``data/landing/news``.
Crawler chi doc cac trang cong khai, khong vuot CAPTCHA hay co che chong bot.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
SOURCE_OWNER = "Trung tam Ho tro phap ly cho doanh nghiep nho va vua - Bo Tu phap"

# Slug va topic do nhom gan de ten file, loc va danh gia retrieval de dang hon.
ARTICLES = [
    {
        "url": "https://htpldn.moj.gov.vn/Pages/chi-tiet-tin.aspx?ItemID=207&l=Tuvanphapluat",
        "slug": "quy-dinh-lam-them-gio",
        "topic": "lam them gio",
    },
    {
        "url": "https://htpldn.moj.gov.vn/Pages/chi-tiet-tin.aspx?ItemID=152&l=Tuvanphapluat",
        "slug": "cong-ty-tu-y-giam-luong",
        "topic": "tien luong",
    },
    {
        "url": "https://htpldn.moj.gov.vn/Pages/chi-tiet-tin.aspx?ItemID=151&l=Tuvanphapluat",
        "slug": "hop-dong-lao-dong-dien-tu",
        "topic": "hop dong lao dong",
    },
    {
        "url": "https://htpldn.moj.gov.vn/Pages/chi-tiet-tin.aspx?ItemID=149&l=Tuvanphapluat",
        "slug": "nghi-viec-khong-bao-truoc-va-bhxh",
        "topic": "cham dut hop dong lao dong",
    },
    {
        "url": "https://htpldn.moj.gov.vn/Pages/chi-tiet-tin.aspx?ItemID=4&l=Tuvanphapluat",
        "slug": "tro-cap-khi-bi-sa-thai",
        "topic": "sa thai",
    },
]


def setup_directory() -> None:
    """Tao thu muc output neu chua ton tai."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _clean_markdown(markdown: str) -> str:
    """Chuan hoa khoang trang nhung van giu cau truc Markdown."""
    markdown = markdown.replace("\u00a0", " ")
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def _crawl_article_sync(article: dict) -> dict:
    url = article["url"]
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    # Tren website nay, the h1 chi la ten chuyen muc "Tu van phap luat";
    # the title moi chua dung tieu de cua bai viet.
    if soup.title is None:
        raise ValueError(f"Khong tim thay tieu de tai {url}")
    title = soup.title.get_text(" ", strip=True)

    content_node = soup.select_one("#News-content")
    if content_node is None:
        raise ValueError(f"Khong tim thay #News-content tai {url}")

    for unwanted in content_node.select("script, style, iframe, noscript"):
        unwanted.decompose()

    content_markdown = _clean_markdown(
        html_to_markdown(str(content_node), heading_style="ATX")
    )
    if len(content_markdown) < 500:
        raise ValueError(
            f"Noi dung qua ngan ({len(content_markdown)} ky tu): {url}"
        )

    article_container = soup.select_one(".article-detail")
    surrounding_text = (
        article_container.get_text(" ", strip=True)
        if article_container
        else soup.get_text(" ", strip=True)
    )
    date_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", surrounding_text)

    return {
        "url": url,
        "title": title,
        "author": SOURCE_OWNER,
        "published_date": date_match.group(0) if date_match else None,
        "date_crawled": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_domain": urlparse(url).netloc,
        "source_owner": SOURCE_OWNER,
        "language": "vi",
        "document_type": "legal_advice_article",
        "topic": article["topic"],
        "content_markdown": content_markdown,
    }


async def crawl_article(article: dict) -> dict:
    """Crawl mot bai trong worker thread de khong chan event loop."""
    return await asyncio.to_thread(_crawl_article_sync, article)


async def crawl_all() -> None:
    """Crawl lan luot de han che tai cho website nguon."""
    setup_directory()

    for index, article_config in enumerate(ARTICLES, 1):
        print(f"[{index}/{len(ARTICLES)}] Crawling: {article_config['url']}")
        article = await crawl_article(article_config)
        filepath = DATA_DIR / f"{article_config['slug']}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  Saved: {filepath.name} "
            f"({len(article['content_markdown'])} characters)"
        )


if __name__ == "__main__":
    asyncio.run(crawl_all())
