import os
import sys
import time
import base64
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

values = {}
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()

user = values.get("RAGPOC_UI_USERNAME", "admin")
pwd = values.get("RAGPOC_UI_PASSWORD", "ragpoc")

# Start Django server in background
env = os.environ.copy()
env["PYTHONPATH"] = "src"

server = subprocess.Popen(
    [sys.executable, "manage.py", "runserver", "127.0.0.1:8001", "--noreload"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

import urllib.request

try:
    for _ in range(20):
        try:
            req = urllib.request.Request("http://127.0.0.1:8001/")
            req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode())
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.5)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        credential = f"{user}:{pwd}"
        context.set_extra_http_headers(
            {"Authorization": "Basic " + base64.b64encode(credential.encode()).decode()}
        )
        page = context.new_page()
        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: errors.append(str(err)))

        response = page.goto("http://127.0.0.1:8001/", wait_until="networkidle")
        print("Page load HTTP:", response.status if response else "None")
        assert response and response.status == 200

        # Verify initial layout: Editor Mode by default
        app_shell = page.locator(".app-shell")
        has_chat_mode = "chat-focus-mode" in (app_shell.get_attribute("class") or "")
        print("Initial chat-focus-mode:", has_chat_mode)
        
        # Take screenshot of default mode
        os.makedirs("qa-output", exist_ok=True)
        page.screenshot(path="qa-output/layout_1_default_editor_mode.png")

        # Open first page from tree if available
        first_page = page.locator(".page-link").first
        if first_page.count() > 0:
            print("Clicking first page...")
            first_page.click()
            page.wait_for_timeout(500)
            
            # Verify editor toolbar toggle button is visible
            editor_toggle = page.locator("#btn-toggle-layout-mode")
            assert editor_toggle.is_visible()
            print("Editor toolbar toggle button found:", editor_toggle.text_content().strip())
            
            # Type something in the editor
            editor_body = page.locator(".editor-body")
            assert editor_body.is_visible()
            
            # Take screenshot of open page in Editor Mode
            page.screenshot(path="qa-output/layout_1b_page_open_editor_mode.png")

        # Click the chat layout button in the chat header
        chat_toggle_btn = page.locator("#btn-toggle-chat-layout")
        assert chat_toggle_btn.is_visible()
        chat_toggle_btn.click()
        page.wait_for_timeout(400)

        # Verify app-shell now has chat-focus-mode
        class_name = app_shell.get_attribute("class") or ""
        print("After toggle class:", class_name)
        assert "chat-focus-mode" in class_name

        # Verify localStorage
        storage_mode = page.evaluate("() => localStorage.getItem('ragpocLayoutMode')")
        print("localStorage ragpocLayoutMode:", storage_mode)
        assert storage_mode == "chat"

        # Screenshot Modo Chat with page open
        page.screenshot(path="qa-output/layout_2_modo_chat_active.png")

        # Test toggling via Alt + M
        page.keyboard.press("Alt+KeyM")
        page.wait_for_timeout(400)
        class_name_after_alt_m = app_shell.get_attribute("class") or ""
        print("After Alt+M class:", class_name_after_alt_m)
        assert "chat-focus-mode" not in class_name_after_alt_m
        assert page.evaluate("() => localStorage.getItem('ragpocLayoutMode')") == "editor"

        # Toggle back to chat mode with Alt+M
        page.keyboard.press("Alt+KeyM")
        page.wait_for_timeout(400)
        assert "chat-focus-mode" in (app_shell.get_attribute("class") or "")

        print("Console errors count:", len(errors))
        if errors:
            for err in errors:
                print("Console error:", err)

        print("ALL LAYOUT TESTS PASSED!")
        browser.close()
finally:
    server.terminate()
    server.wait()
