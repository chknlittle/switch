"""Fetch and extract content from URLs using headless Chromium."""

from __future__ import annotations

import logging

from playwright.async_api import async_playwright

log = logging.getLogger(__name__)

MAX_CONTENT_LEN = 12_000  # chars

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)


async def read_url(url: str) -> str:
    """Fetch URL content using headless Chromium via Playwright."""
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            page = await browser.new_page(user_agent=USER_AGENT)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_selector("body", timeout=10_000)
            text = await page.inner_text("body")
            await browser.close()
            return text[:MAX_CONTENT_LEN]
    except Exception as e:
        log.warning("playwright fetch failed for %s: %s", url, e)
        return f"Error fetching {url}: {e}"
