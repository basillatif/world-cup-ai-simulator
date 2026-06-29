"""
Keep a Streamlit Community Cloud app awake.

Why not curl? A plain HTTP GET to a Streamlit app returns 200 but only serves a
static HTML shell. The Python process starts only after JavaScript runs and opens
a WebSocket to /_stcore/stream. So we drive a real headless browser, let the page
render, and click the "Yes, get this app back up!" button if the app was asleep.

Set the app URL via the STREAMLIT_URL env var (one URL), or list several in URLS.
"""

import os
import sys

from playwright.sync_api import sync_playwright

URLS = [
    os.environ.get("STREAMLIT_URL", "https://YOUR-APP.streamlit.app/"),
]

WAKE_BUTTON_TEXT = "get this app back up"  # substring of the sleep-page button


def visit(page, url: str) -> str:
    page.goto(url, wait_until="networkidle", timeout=90_000)
    # If the app is asleep, the wake button is present -> click it.
    button = page.get_by_text(WAKE_BUTTON_TEXT, exact=False)
    if button.count() > 0:
        button.first.click()
        page.wait_for_load_state("networkidle", timeout=90_000)
        # give the container time to spin the Python process back up
        page.wait_for_timeout(20_000)
        return "WAKE"
    return "OK"


def main() -> int:
    failures = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context()
        page = ctx.new_page()
        for url in URLS:
            try:
                status = visit(page, url)
                print(f"{status} {url}", flush=True)
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"FAIL {url} :: {exc}", flush=True)
        browser.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
