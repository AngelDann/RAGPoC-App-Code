import os

# Must run before pydantic is imported (directly or via django/pydantic_ai).
# Otherwise pydantic auto-loads the `logfire` plugin, which calls
# inspect.getsource() to instrument models — that fails under PyInstaller
# because frozen bytecode has no retrievable source text.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")

import asyncio
import json
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
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

HOST = "127.0.0.1"
# Deliberately not 8080 (nor 8000/3000/5000): those are the ports a machine that does any
# development at all is most likely to already have something on, and this app has no reason to
# compete for one. The port still wants to be *stable* between launches, though -- the UI
# remembers the open workspace and the panel widths in localStorage, which browsers scope per
# origin (host:port), so a port that changed every run would silently reset the layout each time.
PREFERRED_PORT = 47823
# Consecutive ports tried after the preferred one before giving up on a predictable URL.
PORT_SCAN_SPAN = 20


def probe_port(host: str, port: int) -> str:
    """Says who owns a port we could not bind: "ragpoc" (another instance of this app, which the
    caller can attach to), "other" (someone else's server — leave it alone), or "silent" (held by
    something that is not accepting connections yet).

    Reading /health rather than settling for "something accepted the connection" is the whole
    point of the distinction: a bare port probe cannot tell our own second launch apart from an
    unrelated dev server, and pointing the desktop window at a stranger's server is precisely the
    failure this exists to prevent."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError:
        return "silent"
    try:
        # Generous timeout: it only ever delays a launch whose port is occupied, and the reply it
        # waits for can be slow for reasons that are not "this is not RAGPoC" — a busy instance
        # mid-ingest, a cold sqlite read. Treating that as a stranger is the expensive mistake
        # here, since it starts a second server on the same database.
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=8.0) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return "other"
    return "ragpoc" if isinstance(payload, dict) and payload.get("app") == "ragpoc" else "other"


def port_is_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def wait_for_starting_instance(host: str, port: int, timeout: float = 20.0) -> str:
    """Resolves a "silent" port by waiting for whoever holds it to reveal what it is, returning
    the same values as probe_port() plus "free" if the owner disappears meanwhile.

    A silent port is normal, not broken: it is exactly what this launcher looks like during the
    several seconds between reserving its socket and uvicorn serving on it. Someone who
    double-clicks the icon while the first window is still coming up lands inside that gap, and
    without waiting it out the second launch would write the first one off as a stranger and
    start a rival server against the same sqlite file."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = probe_port(host, port)
        if state != "silent":
            return state
        if port_is_free(host, port):
            return "free"
        time.sleep(0.3)
    return "other"


def acquire_port(host: str, preferred_port: int) -> tuple[socket.socket | None, int]:
    """Reserves the port the local server will listen on, returning (bound socket, port) — or
    (None, port) when that port already belongs to a RAGPoC that is still running, which is the
    caller's cue to attach to that instance instead of starting a second one.

    Binding here instead of letting uvicorn bind inside its own thread is what turns "the port is
    taken" from a fatal error into a decision we can act on. uvicorn's bind happens on a
    background thread, where a WinError 10048 could only be logged before the thread died; the
    launcher itself carried on regardless, found *something* listening, and opened the window
    onto whatever process that was.

    Note the absence of SO_REUSEADDR: on Windows that flag lets a bind succeed on a port another
    socket is actively using, which would defeat this check rather than help it."""
    for port in range(preferred_port, preferred_port + PORT_SCAN_SPAN):
        # Two attempts per port so that an owner who releases it while we are still working out
        # who they are (see "free" below) doesn't cost us the port itself.
        for _ in range(2):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((host, port))
            except OSError:
                sock.close()
            else:
                return sock, port

            state = probe_port(host, port)
            # Only the preferred port is worth waiting on: an instance of ours always claims that
            # one first, so a silent occupant anywhere else is not us starting up.
            if state == "silent" and port == preferred_port:
                state = wait_for_starting_instance(host, port)
            if state == "ragpoc":
                return None, port
            if state != "free":
                break

    # Every candidate was taken. An OS-assigned port is less pleasant — being a new origin, it
    # resets the UI's remembered layout — but it always works, and a working window beats a
    # predictable URL.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    return sock, sock.getsockname()[1]


def wait_for_server(host: str, port: int, server_thread: threading.Thread, timeout: float = 20.0) -> None:
    """Polls the port instead of a blind sleep, since server startup time (migrations, RAG
    service init) varies between machines and a fixed delay would either open the window too
    early (blank/connection-refused page) or waste time waiting longer than necessary.

    Watching the thread matters as much as watching the port: if uvicorn dies during startup,
    the port never opens, and without this check the wait would burn the full timeout before
    reporting a timeout rather than the actual failure sitting in the log."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not server_thread.is_alive():
            raise RuntimeError("El servidor local se detuvo durante el arranque; revisa ragpoc.log.")
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


def open_app_window(url: str) -> bool:
    """Shows the desktop window on `url`. Returns True once the native window has been opened
    and then closed by the user, False if no native window was possible and the app fell back
    to a browser window -- which, unlike the native one, gives us no close event to wait on."""
    try:
        import webview

        webview.create_window("RAGPoC — Knowledge Studio", url, width=1400, height=900, min_size=(960, 640))
        webview.start()
        return True
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

    host = HOST
    # Claim the port before the expensive Django startup below, not after: holding the socket
    # from here on removes any window in which another process could take it, and the answer
    # also decides whether that startup is needed at all.
    server_socket, port = acquire_port(host, PREFERRED_PORT)
    url = f"http://{host}:{port}/"

    if server_socket is None:
        # The app is already open — someone launched it a second time. Two instances would each
        # run their own server against the same sqlite file and fight over it, so this launch
        # contributes a window and nothing else, which is also why it can skip django.setup()
        # entirely and appear almost instantly.
        print(f"RAGPoC ya se está ejecutando en {url}; se abre otra ventana sobre esa instancia.")
        open_app_window(url)
        return

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

    print("=" * 60)
    print("🚀 RAGPoC Desktop / Knowledge Studio")
    print(f"📍 Servidor local activo en: {url}")
    print("=" * 60)

    from ragpoc_django.asgi import application

    # uvicorn runs in a background thread so the main thread is free for pywebview's native
    # window loop, which on Windows/macOS must run on the main thread. It serves the socket
    # acquired above rather than binding its own; host/port here only feed its log lines.
    server_config = uvicorn.Config(application, host=host, port=port, log_level="info")
    server = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server.run, kwargs={"sockets": [server_socket]}, daemon=True)
    server_thread.start()

    wait_for_server(host, port, server_thread)

    if not open_app_window(url):
        try:
            while server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    server.should_exit = True
    server_thread.join(timeout=5)


if __name__ == "__main__":
    try:
        main()
        status = 0
    except Exception:
        traceback.print_exc()
        status = 1

    # os._exit rather than falling off the end of the process: closing the window leaves live
    # threads that nobody joins -- pythonnet's CLR, and the ThreadPoolExecutors that several
    # imported libraries register with atexit -- and normal interpreter shutdown blocks waiting
    # on them. That is what left a windowless RAGPoC.exe running after every "close the app",
    # still holding its port, so the next launch could not bind and opened onto the corpse of
    # the previous one. The server was asked to stop and joined above; this makes it certain.
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(status)
