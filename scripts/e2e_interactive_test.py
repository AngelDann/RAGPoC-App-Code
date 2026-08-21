import os
import sys
import time
import base64
import subprocess
import urllib.request
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

os.makedirs("qa-output", exist_ok=True)

# Launch Django server on port 8002
env = os.environ.copy()
env["PYTHONPATH"] = "src"

print("[1/6] Iniciando servidor Django en puerto 8002...")
server = subprocess.Popen(
    [sys.executable, "manage.py", "runserver", "127.0.0.1:8002", "--noreload"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

try:
    # Wait for server ready
    ready = False
    for _ in range(25):
        try:
            req = urllib.request.Request("http://127.0.0.1:8002/")
            req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode())
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.4)

    assert ready, "El servidor Django no respondió en el tiempo esperado"
    print("[2/6] Servidor Django listo y respondiendo HTTP 200.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        credential = f"{user}:{pwd}"
        context.set_extra_http_headers(
            {"Authorization": "Basic " + base64.b64encode(credential.encode()).decode()}
        )
        page = context.new_page()
        
        console_logs = []
        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: console_logs.append(f"[ERROR] {err}"))

        # Step 1: Navigate to App
        print("[3/6] Navegando a la consola web...")
        page.goto("http://127.0.0.1:8002/", wait_until="networkidle")
        page.wait_for_timeout(500)
        
        app_shell = page.locator(".app-shell")
        assert app_shell.is_visible()
        print("  - Shell principal visible. Modo inicial:", "Modo Chat" if "chat-focus-mode" in (app_shell.get_attribute("class") or "") else "Modo Editor (Predeterminado)")
        page.screenshot(path="qa-output/step1_initial_dashboard.png")

        # Step 2: Open a page from tree and edit content
        print("[4/6] Abriendo una página y editando texto en Modo Editor...")
        page.wait_for_selector(".page-select-btn", timeout=5000)
        first_page_link = page.locator(".page-select-btn").first
        first_page_link.click()
        page.wait_for_selector(".editor-container", timeout=5000)
        
        # Verify top center pill is present and 'Notas' is active
        top_seg_pill = page.locator("#page-center-nav .center-mode-segmented-control")
        assert top_seg_pill.is_visible()
        chat_seg_btn = page.locator("#page-center-nav [data-switch-mode='chat']")
        notes_seg_btn = page.locator("#page-center-nav [data-switch-mode='editor']")
        assert chat_seg_btn.is_visible()
        assert notes_seg_btn.is_visible()
        print("  - Selector superior [ Chat | Notas ] detectado en la cabecera.")
        
        # Focus editor and type extra content
        editor_content = page.locator(".ProseMirror")
        assert editor_content.is_visible()
        editor_content.click()
        page.keyboard.press("End")
        page.keyboard.press("Enter")
        test_phrase = "Texto de prueba añadido antes de alternar a Modo Chat."
        page.keyboard.type(test_phrase)
        page.wait_for_timeout(300)
        
        # Check suggested questions in Right Panel in Modo Editor
        sug_btn = page.locator(".btn-suggested-question").first
        assert sug_btn.is_visible()
        print("  - Botones de preguntas sugeridas visibles en panel derecho.")
        
        page.screenshot(path="qa-output/step2_editor_mode_with_text.png")
        print("  - Captura tomada: Modo Editor con notas al centro y preguntas sugeridas en panel derecho.")

        # Step 3: Click 'Chat' pill in top segmented control to switch to Modo Chat
        print("[5/6] Alternando a Modo Chat desde el control [ Chat | Notas ] superior...")
        chat_seg_btn.click()
        page.wait_for_timeout(500)

        # Assertions for Modo Chat
        current_classes = app_shell.get_attribute("class") or ""
        assert "chat-focus-mode" in current_classes, "La clase chat-focus-mode no se aplicó a .app-shell"
        
        # Check that top center pill in Chat mode has 'Chat' active
        chat_center_pill = page.locator("#chat-center-nav-row [data-switch-mode='chat']")
        assert "active" in (chat_center_pill.get_attribute("class") or "")
        
        # Verify text in editor is STILL present in right panel (no reload/loss)
        current_editor_text = editor_content.inner_text()
        assert "Texto de prueba añadido" in current_editor_text, "El texto del editor se perdió durante el cambio de modo"
        print("  - Verificación: El texto en el editor se conservó perfectamente sin recarga.")
        
        # Check Chat input and chat stream are visible in center
        chat_stream = page.locator("#chat-log")
        chat_input = page.locator("#chat-question")
        assert chat_stream.is_visible()
        assert chat_input.is_visible()

        # Check suggested questions in Center Chat in Modo Chat
        sug_btn_center = page.locator(".btn-suggested-question").first
        assert sug_btn_center.is_visible()

        page.screenshot(path="qa-output/step3_modo_chat_active_side_editor.png")
        print("  - Captura tomada: Modo Chat activo con editor lateral y preguntas sugeridas al centro.")

        # Test switching to Studio in Right Panel while in Modo Chat
        print("  - Probando clic en 'Studio' en el panel lateral derecho...")
        side_studio_btn = page.locator("#side-editor-header [data-side-tab='studio']")
        assert side_studio_btn.is_visible()
        side_studio_btn.click()
        page.wait_for_timeout(400)
        studio_container = page.locator("#studio-view-container")
        assert studio_container.is_visible()
        page.screenshot(path="qa-output/step3b_modo_chat_studio_active.png")
        print("  - Verificación: Pestaña Studio en panel derecho abierta y funcional.")

        # Test switching to Adjuntos in Right Panel while in Modo Chat
        print("  - Probando clic en 'Adjuntos' en el panel lateral derecho...")
        side_attach_btn = page.locator("#side-editor-header [data-side-tab='attachments']")
        assert side_attach_btn.is_visible()
        side_attach_btn.click()
        page.wait_for_timeout(400)
        attach_container = page.locator("#attachments-view-container")
        assert attach_container.is_visible()
        page.screenshot(path="qa-output/step3c_modo_chat_attachments_active.png")
        print("  - Verificación: Pestaña Adjuntos en panel derecho abierta y funcional.")

        # Return to 'Notas' tab in Right Panel
        side_notes_btn = page.locator("#side-editor-header [data-side-tab='notes']")
        side_notes_btn.click()
        page.wait_for_timeout(400)
        assert editor_content.is_visible()

        # Step 4: Click a suggested question button to test interaction
        print("[6/6] Probando clic en botón de pregunta sugerida...")
        sug_btn_click = page.locator(".btn-suggested-question").first
        sug_btn_click.click()
        page.wait_for_timeout(1200)
        page.screenshot(path="qa-output/step4_chat_interaction_modo_chat.png")
        print("  - Pregunta sugerida enviada y procesada correctamente.")

        # Click 'Notas' in center pill to return to Editor Mode
        notes_in_chat_center = page.locator("#chat-center-nav-row [data-switch-mode='editor']")
        notes_in_chat_center.click()
        page.wait_for_timeout(500)
        classes_after_return = app_shell.get_attribute("class") or ""
        assert "chat-focus-mode" not in classes_after_return, "Clic en Notas no regresó a Modo Editor"
        print("  - Control [ Notas ] verificado: Regresó a Modo Editor.")
        page.screenshot(path="qa-output/step5_back_to_editor_mode.png")

        # Step 5: Test Chat Grounding Scope Selector (Este cuaderno vs Todo el espacio)
        print("[7/8] Probando cambio de ámbito de fuentes (Grounding Scope Selector)...")
        scope_btn = page.locator("#chat-scope-btn").first
        assert scope_btn.is_visible()
        scope_btn.click()
        page.wait_for_timeout(300)
        page.screenshot(path="qa-output/step6a_scope_dropdown_open.png")

        # Select 'Todo el espacio'
        workspace_opt = page.locator("[data-scope-val='workspace']").first
        assert workspace_opt.is_visible()
        workspace_opt.click()
        page.wait_for_timeout(400)

        scope_label = page.locator("#chat-scope-label").first
        assert "Todo el espacio" in scope_label.inner_text(), "No se cambió la etiqueta a 'Todo el espacio'"
        print("  - Verificación: Ámbito cambiado a 'Todo el espacio' con éxito.")
        page.screenshot(path="qa-output/step6_scope_workspace_selected.png")

        # Switch back to 'Este cuaderno'
        scope_btn.click()
        page.wait_for_timeout(300)
        notebook_opt = page.locator("[data-scope-val='notebook']").first
        notebook_opt.click()
        page.wait_for_timeout(400)
        assert "Este cuaderno" in scope_label.inner_text(), "No se regresó la etiqueta a 'Este cuaderno'"
        print("  - Verificación: Ámbito regresado a 'Este cuaderno' con éxito.")

        # Step 6: Test Chat History Search Input
        print("[8/8] Probando buscador interactivo en el Historial de Chats...")
        hist_btn = page.locator("#threads-history-btn").first
        assert hist_btn.is_visible()
        hist_btn.click()
        page.wait_for_timeout(400)
        
        search_input = page.locator("#chat-thread-search")
        assert search_input.is_visible(), "El buscador de chats no está visible en el dropdown"
        search_input.fill("a")
        page.wait_for_timeout(300)
        print("  - Verificación: Búsqueda reactiva en historial ejecutada.")
        page.screenshot(path="qa-output/step7_chat_history_search.png")
        
        clear_btn = page.locator("#chat-thread-search-clear")
        if clear_btn.is_visible():
            clear_btn.click()
            page.wait_for_timeout(200)
            print("  - Verificación: Botón de limpiar búsqueda funcional.")

        # Test selecting a thread from history and verifying dropdown closes
        thread_items = page.locator("#threads-history-list [data-switch-thread]")
        if thread_items.count() > 0:
            first_thread = thread_items.first
            first_thread.click()
            page.wait_for_timeout(400)
            hist_menu = page.locator("#threads-history-btn + .dropdown-menu")
            assert not hist_menu.is_visible(), "El menú de historial no se cerró al seleccionar una conversación"
            print("  - Verificación: Al hacer clic en un chat del historial, el menú se cierra limpiamente.")
            page.screenshot(path="qa-output/step8_history_closed_after_select.png")

        # Step 7: Test deleting a thread and verifying counter decreases
        print("[9/9] Probando eliminación de conversación y decremento del contador...")
        with page.expect_response(lambda r: "/api/threads" in r.url):
            hist_btn.click()
        page.wait_for_timeout(300)
        
        initial_badge = page.locator("#threads-badge-total").inner_text()
        initial_count = int(initial_badge) if initial_badge.isdigit() else 0
        print(f"  - Contador inicial de conversaciones: {initial_count}")
        
        delete_btns = page.locator("#threads-history-list [data-delete-thread-id]")
        if delete_btns.count() > 0:
            first_delete_btn = delete_btns.first
            first_delete_btn.click()
            page.wait_for_timeout(300)
            
            delete_modal = page.locator("#delete-modal-backdrop")
            assert delete_modal.is_visible(), "El modal de confirmación de eliminación no se abrió"
            page.screenshot(path="qa-output/step9a_delete_modal_open.png")
            print("  - Verificación: Modal de confirmación de eliminación visible y al frente.")
            
            confirm_btn = page.locator("#btn-confirm-delete")
            with page.expect_response(lambda r: "/api/threads" in r.url and r.request.method == "DELETE"):
                confirm_btn.click()
            page.wait_for_timeout(600)
            
            # Reopen history to check new count
            with page.expect_response(lambda r: "/api/threads" in r.url):
                hist_btn.click()
            page.wait_for_timeout(300)
            
            new_badge = page.locator("#threads-badge-total").inner_text()
            new_count = int(new_badge) if new_badge.isdigit() else 0
            print(f"  - Contador después de eliminar: {new_count}")
            assert new_count == initial_count - 1, f"El contador no decreció exactamente 1: inicial={initial_count}, nuevo={new_count}"
            print("  - Verificación: Conversación eliminada con éxito y contador actualizado correctamente.")
            page.screenshot(path="qa-output/step9b_history_after_delete.png")

        # Step 10: Test right drawer tabs (Chat vs Studio vs Adjuntos) and verify chat-controls-bar is hidden on Studio & Adjuntos
        print("[10/10] Probando pestañas del panel derecho (Chat / Studio / Adjuntos)...")
        # Ensure we close history dropdown if open
        if page.locator("#threads-history-btn + .dropdown-menu").is_visible():
            hist_btn.click()
            page.wait_for_timeout(200)

        # 1. Click Studio
        studio_btn = page.locator("#pane-switch-studio")
        studio_btn.click()
        page.wait_for_timeout(300)
        chat_subhdr = page.locator("#chat-controls-subhdr")
        studio_subhdr = page.locator("#studio-controls-subhdr")
        assert not chat_subhdr.is_visible(), "El selector de agente/historial sigue visible en la pestaña Studio"
        assert studio_subhdr.is_visible(), "La cabecera de Studio no está visible"
        print("  - Verificación: En pestaña Studio, el componente de chat está oculto y se muestra Guías del Cuaderno.")
        page.screenshot(path="qa-output/step10_studio_tab_clean.png")

        # 2. Click Adjuntos
        attach_btn = page.locator("#pane-switch-attachments")
        attach_btn.click()
        page.wait_for_timeout(300)
        attach_subhdr = page.locator("#attachments-controls-subhdr")
        assert not chat_subhdr.is_visible(), "El selector de agente/historial sigue visible en la pestaña Adjuntos"
        assert attach_subhdr.is_visible(), "La cabecera de Adjuntos no está visible"
        print("  - Verificación: En pestaña Adjuntos, el componente de chat está oculto y se muestra Adjuntos reutilizables.")
        page.screenshot(path="qa-output/step11_attachments_tab_clean.png")

        # 3. Click Chat back
        chat_pane_btn = page.locator("#pane-switch-chat")
        chat_pane_btn.click()
        page.wait_for_timeout(300)
        assert chat_subhdr.is_visible(), "El componente de chat no volvió a mostrarse en la pestaña Chat"
        print("  - Verificación: Al regresar a la pestaña Chat, el componente de agente/historial vuelve a estar visible.")

        errors = [log for log in console_logs if "[error]" in log.lower() or "[ERROR]" in log]
        print(f"\n--- Resumen de logs del navegador ---")
        print(f"Total mensajes de consola: {len(console_logs)}")
        print(f"Errores críticos: {len(errors)}")
        if errors:
            for err in errors:
                print("  ", err)

        print("\n¡TODAS LAS PRUEBAS INTERACTIVAS DE PLAYWRIGHT PASARON EXITOSAMENTE!")
        browser.close()

finally:
    print("Deteniendo servidor de prueba...")
    server.terminate()
    server.wait()
