import requests
from bs4 import BeautifulSoup
import html2text
from typing import Optional


def scrape_job_offer(url: str, timeout: int = 15) -> Optional[str]:
    """
    Scrape a job offer from a URL and return clean text content.
    Supports LinkedIn, WTTJ, Indeed, and generic pages.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Convert to clean markdown text
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0

    text = converter.handle(str(soup))

    # Truncate to avoid LLM context overflow (~6000 chars is enough)
    return text[:6000].strip()


def scrape_multiple(urls: list[str]) -> dict[str, Optional[str]]:
    """Scrape multiple URLs and return a dict {url: content}."""
    results = {}
    for url in urls:
        results[url] = scrape_job_offer(url)
    return results
