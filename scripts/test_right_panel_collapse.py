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
    [sys.executable, "manage.py", "runserver", "127.0.0.1:8002", "--noreload"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

import urllib.request

try:
    for _ in range(25):
        try:
            req = urllib.request.Request("http://127.0.0.1:8002/")
            req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode())
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.4)

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

        response = page.goto("http://127.0.0.1:8002/", wait_until="networkidle")
        print("Page load HTTP:", response.status if response else "None")
        assert response and response.status == 200

        app_shell = page.locator(".app-shell")
        page_area = page.locator("#page-area")
        chat_drawer = page.locator("#chat-drawer")

        # Step 1: Default Editor Mode
        print("--- Step 1: Default Editor Mode ---")
        assert "chat-focus-mode" not in (app_shell.get_attribute("class") or "")
        top_pill = page.locator("#page-center-nav")
        assert top_pill.is_visible()
        print("Top pill is visible in Editor Mode")

        os.makedirs("qa-output", exist_ok=True)
        page.screenshot(path="qa-output/test_1_default_editor.png")

        # Step 2: Collapse right panel in Editor Mode
        print("--- Step 2: Collapse right panel in Editor Mode ---")
        btn_collapse_chat = page.locator("#btn-collapse-chat")
        assert btn_collapse_chat.is_visible()
        btn_collapse_chat.click()
        page.wait_for_timeout(300)

        assert "chat-collapsed" in (app_shell.get_attribute("class") or "")
        assert "panel-collapsed" in (chat_drawer.get_attribute("class") or "")
        assert "panel-collapsed" not in (page_area.get_attribute("class") or "")
        chat_rail = page.locator("#chat-rail")
        assert chat_rail.is_visible()
        assert page.locator("#btn-expand-chat").is_visible()
        print("Right panel is collapsed, #chat-rail is visible on the right")
        page.screenshot(path="qa-output/test_2_editor_mode_collapsed.png")

        # Step 3: Switch to Modo Chat while right panel is collapsed
        print("--- Step 3: Switch to Modo Chat while right panel is collapsed ---")
        chat_mode_btn = page.locator("#page-center-nav [data-switch-mode='chat']")
        assert chat_mode_btn.is_visible()
        chat_mode_btn.click()
        page.wait_for_timeout(300)

        # In Modo Chat:
        # appShell should have chat-focus-mode and chat-collapsed
        assert "chat-focus-mode" in (app_shell.get_attribute("class") or "")
        assert "chat-collapsed" in (app_shell.get_attribute("class") or "")
        
        # Center is chat_drawer -> MUST NOT be panel-collapsed
        assert "panel-collapsed" not in (chat_drawer.get_attribute("class") or "")
        
        # Central switch [ Chat | Notas ] MUST be visible!
        chat_center_nav = page.locator("#chat-center-nav-row")
        assert chat_center_nav.is_visible()
        print("Center [ Chat | Notas ] segmented switch is VISIBLE in Modo Chat!")

        # Chat stream / input must be visible
        chat_view = page.locator("#chat-view-container")
        assert chat_view.is_visible()
        print("Chat view container is VISIBLE in center in Modo Chat!")

        # Right is page_area -> MUST BE panel-collapsed and show #page-rail
        assert "panel-collapsed" in (page_area.get_attribute("class") or "")
        page_rail = page.locator("#page-rail")
        assert page_rail.is_visible()
        btn_expand_page = page.locator("#btn-expand-page")
        assert btn_expand_page.is_visible()
        print("Right rail #page-rail with expand button is VISIBLE on the right in Modo Chat!")
        page.screenshot(path="qa-output/test_3_chat_mode_collapsed.png")

        # Step 4: Expand right panel while in Modo Chat
        print("--- Step 4: Expand right panel while in Modo Chat ---")
        btn_expand_page.click()
        page.wait_for_timeout(300)

        assert "chat-collapsed" not in (app_shell.get_attribute("class") or "")
        assert "panel-collapsed" not in (page_area.get_attribute("class") or "")
        assert "panel-collapsed" not in (chat_drawer.get_attribute("class") or "")
        side_editor_hdr = page.locator("#side-editor-header")
        assert side_editor_hdr.is_visible()
        btn_collapse_side = page.locator("#btn-collapse-side-editor")
        assert btn_collapse_side.is_visible()
        print("Right panel is expanded in Modo Chat, header and collapse button are VISIBLE")
        page.screenshot(path="qa-output/test_4_chat_mode_expanded.png")

        # Step 5: Collapse right panel from Modo Chat button
        print("--- Step 5: Collapse right panel from Modo Chat button ---")
        btn_collapse_side.click()
        page.wait_for_timeout(300)

        assert "chat-collapsed" in (app_shell.get_attribute("class") or "")
        assert "panel-collapsed" in (page_area.get_attribute("class") or "")
        assert "panel-collapsed" not in (chat_drawer.get_attribute("class") or "")
        assert page.locator("#btn-expand-page").is_visible()
        print("Right panel collapsed successfully via #btn-collapse-side-editor")
        page.screenshot(path="qa-output/test_5_chat_mode_collapsed_again.png")

        # Step 6: Switch back to Modo Notas while collapsed
        print("--- Step 6: Switch back to Modo Notas while collapsed ---")
        editor_mode_btn = page.locator("#chat-center-nav-row [data-switch-mode='editor']")
        assert editor_mode_btn.is_visible()
        editor_mode_btn.click()
        page.wait_for_timeout(300)

        assert "chat-focus-mode" not in (app_shell.get_attribute("class") or "")
        assert "chat-collapsed" in (app_shell.get_attribute("class") or "")
        assert "panel-collapsed" not in (page_area.get_attribute("class") or "")
        assert "panel-collapsed" in (chat_drawer.get_attribute("class") or "")
        assert page.locator("#page-center-nav").is_visible()
        assert page.locator("#btn-expand-chat").is_visible()
        print("Switched back to Modo Notas: Center editor and right collapsed rail intact!")
        page.screenshot(path="qa-output/test_6_back_to_editor_mode.png")

        # Step 7: Open page with right panel collapsed and test layout toggle
        print("--- Step 7: Open page with right panel collapsed and test layout toggle ---")
        first_page = page.locator(".page-link").first
        if first_page.count() > 0:
            first_page.click()
            page.wait_for_timeout(500)
            
            # Center editor should be visible
            assert page.locator(".editor-container").is_visible()
            assert page.locator("#page-center-nav").is_visible()
            print("Note opened in Editor Mode while right panel collapsed")
            
            # Switch to Chat mode
            page.locator("#page-center-nav [data-switch-mode='chat']").click()
            page.wait_for_timeout(300)
            
            assert "chat-focus-mode" in (app_shell.get_attribute("class") or "")
            assert page.locator("#chat-center-nav-row").is_visible()
            assert page.locator("#btn-expand-page").is_visible()
            print("Switched to Modo Chat with note loaded: center chat toggle and right expand button intact!")
            page.screenshot(path="qa-output/test_7_note_chat_mode_collapsed.png")

            # Expand right panel
            page.locator("#btn-expand-page").click()
            page.wait_for_timeout(300)
            assert page.locator(".editor-container").is_visible()
            assert page.locator("#btn-collapse-side-editor").is_visible()
            print("Expanded right panel: note editor is visible in side panel!")
            page.screenshot(path="qa-output/test_8_note_chat_mode_expanded.png")

        print("Console errors count:", len(errors))
        if errors:
            for err in errors:
                print("Console error:", err)

        print("\n>>> ALL RIGHT PANEL COLLAPSE AND MODE SWITCH TESTS PASSED! <<<")
        browser.close()
finally:
    server.terminate()
    server.wait()
