import os

# Must run before pydantic is imported (directly or via django/pydantic_ai).
# Otherwise pydantic auto-loads the `logfire` plugin, which calls
# inspect.getsource() to instrument models — that fails under PyInstaller
# because frozen bytecode has no retrievable source text.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")

import asyncio
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

import uvicorn

# A windowed (console=False) PyInstaller build has no console, so sys.stdout/
# stderr are None -- every bare print() below (and uvicorn's own logging)
# would crash with AttributeError on first use. Redirect them to a log file
# next to the executable instead, which also gives us something to inspect
# if desktop startup fails with no window ever appearing.
if sys.stdout is None or sys.stderr is None:
    log_dir = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    log_file = open(log_dir / "ragpoc.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log_file

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


def open_browser_app_window(url: str) -> bool:
    """Opens `url` as a Chromium "app window": a normal Edge/Chrome process rendering one
    chrome-less window with its own taskbar entry -- no tabs, no address bar, no bookmarks
    bar. Returns False if no such browser is installed.

    This is what the app degrades to when the pywebview/WebView2 path cannot start at all, so
    that "the native window failed" still means a dedicated desktop window rather than a tab
    lost somewhere in the user's browsing session. It needs nothing installed beyond the Edge
    that ships with every supported version of Windows.

    Deliberately no --user-data-dir: reusing the default profile keeps this window's
    localStorage (where the UI remembers which workspace was open) shared with the plain-tab
    fallback below, and lets an already-running browser serve the window immediately instead
    of cold-starting a second, isolated browser process."""
    if sys.platform != "win32":
        return False
    relative = ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe")
    roots = [os.environ.get(var) for var in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    for root in roots:
        if not root:
            continue
        for name in relative:
            browser = Path(root) / name
            if not browser.is_file():
                continue
            try:
                subprocess.Popen([str(browser), f"--app={url}"], close_fds=True)
                return True
            except OSError:
                continue
    return False


def main():
    import django
    from django.core.management import call_command

    from ragpoc.clr_host import ensure_clr_host_config
    from ragpoc.config import get_settings
    from ragpoc.updater import cleanup_stale_update_files, unblock_downloaded_install

    # Sweeps any *.exe.old left behind by a self-update whose final cleanup step lost a file
    # lock race (see ragpoc.updater._write_updater_script) — by now that lock is long gone.
    cleanup_stale_update_files()

    # Both of these make the native window below survive being installed from a downloaded
    # zip, whose files Windows tags as untrusted: the first tells the .NET runtime to load
    # pythonnet's assemblies anyway, the second strips the tags. They must run before anything
    # imports webview, since that is what starts the CLR that reads the config file.
    ensure_clr_host_config()
    unblock_downloaded_install()

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

    # Django only imports urls.py (and therefore views.py, and therefore pydantic_ai and every
    # other heavy dependency views.py pulls in) the first time it needs to resolve a URL --
    # migrate above never touches it. Left alone, that import (~4-5s, mostly pydantic_ai) would
    # happen lazily on the *first HTTP request the webview window makes*, i.e. right after the
    # window is already visible to the user -- it would sit there looking open but unresponsive
    # for several seconds. Forcing it here instead pays that cost during the splash period,
    # before wait_for_server()/webview.create_window() below, so the window is instantly usable
    # the moment it appears.
    print("Preparando la aplicación...")
    from django.urls import get_resolver

    get_resolver().url_patterns

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
        # No native webview backend available (e.g. the WebView2 runtime really is missing).
        # Log the whole traceback, not just str(e): every past instance of this has been a
        # CLR/pythonnet load failure whose one-line message named a symptom rather than a
        # cause, and this log file is the only diagnostic a user can send back.
        print(f"No se pudo abrir la ventana nativa ({e}); usando una ventana del navegador…")
        print(traceback.format_exc())
        # Still a dedicated, chrome-less window if Edge/Chrome is present; only a plain tab
        # if neither is, which on Windows means someone removed the bundled Edge.
        if not open_browser_app_window(url):
            print("Sin navegador compatible; abriendo una pestaña en el navegador predeterminado…")
            webbrowser.open(url)
        try:
            while server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    server.should_exit = True


if __name__ == "__main__":
    main()
