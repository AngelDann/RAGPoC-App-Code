import os

# Must run before pydantic is imported (directly or via django/pydantic_ai).
# Otherwise pydantic auto-loads the `logfire` plugin, which calls
# inspect.getsource() to instrument models — that fails under PyInstaller
# because frozen bytecode has no retrievable source text.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")

import asyncio
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

# Windows consoles default to the system codepage (e.g. cp1252), not UTF-8,
# so printing emoji below would otherwise crash with UnicodeEncodeError.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# asyncio's default ProactorEventLoop on Windows logs a noisy, self-repeating
# "socket.send() raised exception." loop whenever a client resets a connection
# mid-request (e.g. closing a tab during a file download) — a known CPython/asyncio
# issue that otherwise floods the log and stalls the server. The selector loop
# doesn't have this bug and is otherwise equivalent for a plain HTTP dev server.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set root dir and PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if sys.path[0] != str(ROOT_DIR):
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ragpoc_django.settings")
# Compiled/local desktop build: single local user on 127.0.0.1, no login screen.
os.environ["RAGPOC_DESKTOP_MODE"] = "1"


def wait_for_server(host: str, port: int, timeout: float = 20.0) -> None:
    """Polls the port instead of a blind sleep, since server startup time (migrations, RAG
    service init) varies between machines and a fixed delay would either open the window too
    early (blank/connection-refused page) or waste time waiting longer than necessary."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.15)
    raise RuntimeError("El servidor local no respondió a tiempo.")


def main():
    import django
    from django.core.management import call_command

    from ragpoc.config import get_settings

    # The sqlite file (and uploads/renders/derived dirs) live under a data/
    # folder that may not exist yet on a fresh install — Django's sqlite
    # backend errors with "unable to open database file" if the parent
    # directory is missing, so create it before django.setup()/migrate touch it.
    get_settings().ensure_directories()

    django.setup()

    # Run auto-migrations on startup
    try:
        print("Verificando base de datos y migraciones...")
        call_command("migrate", interactive=False)
    except Exception as e:
        print(f"Aviso en migraciones: {e}")

    host = "127.0.0.1"
    port = 8080
    url = f"http://{host}:{port}/"

    print("=" * 60)
    print("🚀 RAGPoC Desktop / Knowledge Studio")
    print(f"📍 Servidor local activo en: {url}")
    print("=" * 60)

    from ragpoc_django.asgi import application

    # uvicorn runs in a background thread so the main thread is free for pywebview's native
    # window loop, which on Windows/macOS must run on the main thread.
    server_config = uvicorn.Config(application, host=host, port=port, log_level="info")
    server = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    wait_for_server(host, port)

    try:
        import webview

        webview.create_window("RAGPoC — Knowledge Studio", url, width=1400, height=900, min_size=(960, 640))
        webview.start()
    except Exception as e:
        # No native webview backend available (e.g. WebView2 runtime missing) — fall back to
        # the browser tab behavior instead of leaving the user with no way to reach the app.
        print(f"No se pudo abrir la ventana nativa ({e}); abriendo en el navegador…")
        webbrowser.open(url)
        try:
            while server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    server.should_exit = True


if __name__ == "__main__":
    main()
