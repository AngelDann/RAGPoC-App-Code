import base64
from pathlib import Path

from playwright.sync_api import sync_playwright

question = "¿Puedo usar Remotion con Claude Code?"
values = {}
for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        values[key] = value

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1200})
    credential = f"{values['RAGPOC_UI_USERNAME']}:{values['RAGPOC_UI_PASSWORD']}"
    context.set_extra_http_headers(
        {"Authorization": "Basic " + base64.b64encode(credential.encode()).decode()}
    )
    page = context.new_page()
    console_errors = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    response = page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
    assert response and response.status == 200
    page.locator("#chat-question").fill(question)
    page.get_by_role("button", name="Preguntar").click()
    page.wait_for_function(
        "document.querySelector('.message.assistant .message-text')?.textContent.trim().length > 0",
        timeout=45000,
    )
    page.wait_for_function(
        "document.querySelector('#chat-send').disabled === false",
        timeout=60000,
    )
    answer = page.locator(".message.assistant .message-text").inner_text()
    citations = page.locator(".message.assistant .citation").all_inner_texts()
    page.screenshot(path="qa-output/remotion-claudecode-video-proof.png", full_page=True)
    print("http=", response.status)
    print("question=", question)
    print("answer=", answer)
    print("citations=", citations)
    print("video_citation_present=", any("video_7cbca5f727d6.mp4" in citation for citation in citations))
    print("console_errors=", len(console_errors))
    browser.close()
