import base64
from pathlib import Path

from playwright.sync_api import sync_playwright

values = {}
for line in Path('.env').read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        key, value = line.split('=', 1)
        values[key] = value

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1200})
    credential = f"{values['RAGPOC_UI_USERNAME']}:{values['RAGPOC_UI_PASSWORD']}"
    context.set_extra_http_headers(
        {"Authorization": "Basic " + base64.b64encode(credential.encode()).decode()}
    )
    page = context.new_page()
    errors = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    response = page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
    assert response and response.status == 200
    page.locator("#chat-question").fill("¿Qué información hay sobre la arquitectura multi-agente?")
    page.locator("#chat-form").get_by_role("button", name="Preguntar").click()
    page.wait_for_function(
        "document.querySelector('.message.assistant .message-text')?.textContent.trim().length > 0",
        timeout=30000,
    )
    answer = page.locator(".message.assistant .message-text").inner_text()
    citations = page.locator(".message.assistant .citation").count()
    assert answer
    assert citations >= 1
    page.screenshot(path="qa-output/chat-real-response.png", full_page=True)
    print("chat_http=", response.status)
    print("answer_characters=", len(answer))
    print("citation_count=", citations)
    print("console_errors=", len(errors))
    print("answer_preview=", answer[:280].replace("\n", " "))
    browser.close()
