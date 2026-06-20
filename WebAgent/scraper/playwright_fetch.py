"""
scraper/playwright_fetch.py
===========================
Fetches JavaScript-rendered web pages using Playwright and extracts
clean readable text using BeautifulSoup.

Returns None on any failure so the caller can skip gracefully.

Usage
-----
    from scraper.playwright_fetch import PageFetcher

    fetcher = PageFetcher()
    text = fetcher.fetch("https://example.com")
    if text is not None:
        print(text)
"""

import logging
from typing import Optional

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup


class PageFetcher:

    # ── Initialise all variables ───────────────────────────────
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.TIMEOUT_MS   = 15000   # 15 seconds — light on CPU, avoids hanging
        self.MAX_CHARS    = 500    # truncation limit — fits comfortably in LLM context
        self.USER_AGENT   = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

        # HTML tags whose content is noise — stripped before text extraction
        self.NOISE_TAGS = [
            "script", "style", "noscript",
            "header", "footer", "nav", "aside",
            "form", "button", "input", "select",
            "iframe", "svg", "img",
        ]

    # ── Fetch raw HTML from a URL using Playwright ──────────────
    def _fetch_html(self, url: str) -> Optional[str]:
        """
        Launch a headless Chromium browser, navigate to the URL,
        and return the fully rendered HTML as a string.

        Returns None on timeout or any navigation error.
        """
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page    = browser.new_page(user_agent=self.USER_AGENT)

                page.goto(url, timeout=self.TIMEOUT_MS, wait_until="domcontentloaded")
                html = page.content()

                browser.close()
                return html

        except PlaywrightTimeoutError:
            self.logger.warning("⏱️  Timeout fetching '%s' (limit: %dms).", url, self.TIMEOUT_MS)
            return None

        except Exception as exc:
            self.logger.warning("❌ Failed to fetch '%s': %s", url, exc)
            return None

    # ── Extract clean text from raw HTML ────────────────────────
    def _extract_text(self, html: str) -> str:
        """
        Parse HTML with BeautifulSoup, strip all noise tags,
        and return clean readable text.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove all noise tags and their contents
        for tag in self.NOISE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract text with a single space as separator
        text = soup.get_text(separator=" ", strip=True)

        # Collapse multiple whitespace characters into a single space
        import re
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ── Truncate text to stay within LLM context limits ─────────
    def _truncate(self, text: str) -> str:
        """
        Truncate text to MAX_CHARS characters.
        Truncates at the last full sentence within the limit where possible.
        """
        if len(text) <= self.MAX_CHARS:
            return text

        truncated = text[:self.MAX_CHARS]

        # Try to end at the last full sentence
        last_period = truncated.rfind(".")
        if last_period > self.MAX_CHARS * 0.8:  # only if sentence end is reasonably close
            truncated = truncated[:last_period + 1]

        self.logger.info(
            "✂️  Text truncated from %d to %d characters.", len(text), len(truncated)
        )

        return truncated

    # ── Main public method — fetch, extract, truncate ───────────
    def fetch(self, url: str) -> Optional[str]:
        """
        Fetch a URL and return clean, truncated text content.

        Args:
            url: The full URL to fetch.

        Returns:
            Clean text string, or None if the page could not be fetched.
        """
        self.logger.info("🌐 Fetching: %s", url)

        html = self._fetch_html(url)
        if html is None:
            return None

        text = self._extract_text(html)
        if not text:
            self.logger.warning("⚠️  No text extracted from '%s'.", url)
            return None

        text = self._truncate(text)

        self.logger.info(
            "✅ Extracted %d characters from '%s'.", len(text), url
        )

        return text

    # ── Self-test ─────────────────────────────────────────────────
    def run(self) -> None:
        logging.basicConfig(level=logging.INFO)

        print("\n🌐 PageFetcher — Self Test\n" + "─" * 40)

        test_urls = [
            "https://en.wikipedia.org/wiki/LangChain",
            "https://this-url-does-not-exist-xyz.com",  # failure case
        ]

        for url in test_urls:
            print(f"\n🔗 URL: {url}")
            result = self.fetch(url)

            if result is not None:
                print(f"✅ Extracted {len(result)} characters.")
                print(f"📄 Preview: {result[:300]}…")
            else:
                print("⛔ Returned None — fetch failed as expected.")

        print("\n✅ PageFetcher self-test complete.")


if __name__ == "__main__":
    PageFetcher().run()
