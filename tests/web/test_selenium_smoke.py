from __future__ import annotations

import threading
from contextlib import suppress

import pytest

try:
    from selenium.common.exceptions import \
        WebDriverException  # type: ignore[import]
    from selenium.webdriver.common.by import By  # type: ignore[import]
except ImportError:
    pytestmark = pytest.mark.skip(reason="Selenium is not installed")

from werkzeug.serving import make_server


def _start_server(app):
    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _make_driver(browser_name: str):
    name = browser_name.lower()
    try:
        if name == "chrome":
            from selenium.webdriver import Chrome
            from selenium.webdriver.chrome.options import \
                Options as ChromeOptions

            chrome_options = ChromeOptions()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            return Chrome(options=chrome_options)

        if name == "firefox":
            from selenium.webdriver import Firefox
            from selenium.webdriver.firefox.options import \
                Options as FirefoxOptions

            firefox_options = FirefoxOptions()
            firefox_options.add_argument("--headless")
            return Firefox(options=firefox_options)
    except (ImportError, WebDriverException):
        return None
    return None


@pytest.mark.selenium
def test_dashboard_heading_visible(web_app, selenium_browser_name):
    if not selenium_browser_name:
        pytest.skip(
            "Selenium browser not configured. "
            "Set SELENIUM_BROWSER to enable."
        )

    driver = _make_driver(selenium_browser_name)
    if driver is None:
        pytest.skip(
            f"Unable to start Selenium driver for {selenium_browser_name}."
        )

    server, thread = _start_server(web_app)
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base_url}/")
        heading = driver.find_element(By.TAG_NAME, "h2")
        assert "Registry Overview" in heading.text
    finally:
        with suppress(Exception):
            driver.quit()
        server.shutdown()
        thread.join(timeout=2)
