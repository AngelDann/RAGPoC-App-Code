from pathlib import Path

from playwright.sync_api import sync_playwright

values = {}
for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        values[k] = v

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda err: errors.append(str(err)))
    page.context.set_extra_http_headers(
        {
            "Authorization": "Basic "
            + __import__("base64")
            .b64encode(f"{values['RAGPOC_UI_USERNAME']}:{values['RAGPOC_UI_PASSWORD']}".encode())
            .decode()
        }
    )
    response = page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
    assert response.status == 200
    assert page.get_by_text("Buscar conocimiento").is_visible()
    assert page.get_by_text("Subir archivos").is_visible()
    assert page.locator("#upload-button").is_disabled()
    page.locator("#query").fill("arquitectura multi-agente para validación de pagos")
    page.locator("#search-form button").click()
    page.wait_for_timeout(1500)
    assert page.locator(".result").count() >= 1
    page.screenshot(path="qa-output/ragpoc-ui-smoke.png", full_page=True)
    print("ui_http=", response.status)
    print("result_count=", page.locator(".result").count())
    print("console_errors=", len(errors))
    for error in errors:
        print("console_error=", error)
    browser.close()
