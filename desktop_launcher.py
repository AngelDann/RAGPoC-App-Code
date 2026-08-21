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

HOST = "127.0.0.1"
# Deliberately not 8080 (nor 8000/3000/5000): those are the ports a machine that does any
# development at all is most likely to already have something on, and this app has no reason to
# compete for one. The port still wants to be *stable* between launches, though -- the UI
# remembers the open workspace and the panel widths in localStorage, which browsers scope per
# origin (host:port), so a port that changed every run would silently reset the layout each time.
PREFERRED_PORT = 47823
# Consecutive ports tried after the preferred one before giving up on a predictable URL.
PORT_SCAN_SPAN = 20


def probe_port(host: str, port: int, timeout: float = 8.0) -> str:
    """Says who owns a port we could not bind: "ragpoc" (another instance of this app, which the
    caller can attach to), "other" (someone else's server — leave it alone), or "silent" (held by
    something that is not accepting connections yet).

    Reading /health rather than settling for "something accepted the connection" is the whole
    point of the distinction: a bare port probe cannot tell our own second launch apart from an
    unrelated dev server, and pointing the desktop window at a stranger's server is precisely the
    failure this exists to prevent.

    `timeout` defaults to the generous 8s the preferred port deserves (see acquire_port's call
    sites for why); callers checking a fallback port that was never ours to begin with pass a
    short one instead, since there both the wait and the eventual answer are pure overhead."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError:
        return "silent"
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as response:
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
    socket is actively using, which would defeat this check rather than help it.

    Bounded by ACQUIRE_DEADLINE_SECONDS overall: the preferred port alone can legitimately take
    up to ~20s to resolve (see wait_for_starting_instance) when it's genuinely another launch of
    this same app still starting up, and that budget is left untouched here since misreading it
    as a stranger is what starts a second server against the same sqlite file. What used to be
    unbounded was scanning *past* the preferred port -- PORT_SCAN_SPAN candidates each waiting
    out a full probe_port() timeout with no window on screen yet, measured at 8.1s/24.1s/48.2s
    for 1/3/6 occupied ports on a real machine. Every port past the first is not ours to begin
    with if it wasn't silent-and-preferred, so it gets a short probe instead of the generous one,
    and the deadline below caps the worst case regardless of how many are occupied."""
    ACQUIRE_DEADLINE_SECONDS = 25.0
    deadline = time.monotonic() + ACQUIRE_DEADLINE_SECONDS
    for port in range(preferred_port, preferred_port + PORT_SCAN_SPAN):
        if time.monotonic() > deadline:
            print(
                f"acquire_port: se agotó el plazo de {ACQUIRE_DEADLINE_SECONDS:.0f}s buscando un "
                "puerto; se usará uno asignado por el sistema."
            )
            break
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

            is_preferred = port == preferred_port
            state = probe_port(host, port, timeout=8.0 if is_preferred else 1.0)
            # Only the preferred port is worth waiting on: an instance of ours always claims that
            # one first, so a silent occupant anywhere else is not us starting up.
            if state == "silent" and is_preferred:
                state = wait_for_starting_instance(host, port)
            if state == "ragpoc":
                return None, port
            if state != "free":
                print(f"acquire_port: puerto {port} descartado ({state})")
                break

    # Every candidate was taken (or the deadline above cut the scan short). An OS-assigned port
    # is less pleasant — being a new origin, it resets the UI's remembered layout — but it always
    # works, and a working window beats a predictable URL.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    return sock, sock.getsockname()[1]


def _log_path() -> Path:
    """Same rule the sys.stdout redirect above uses to place ragpoc.log, computed independently
    so it can be named in an error dialog even on the (normal) path where nothing redirected
    stdout -- a dev run with a real console never opens the file at all."""
    base = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    return base / "ragpoc.log"


def _show_startup_failure(detail: str) -> None:
    """Puts the failure in front of the user, not just in a log file they don't know exists.

    Without this, main() raising meant the process printed a traceback to ragpoc.log (invisible
    on a windowed build -- sys.stdout was redirected there specifically because there is no
    console to show it on) and then exited via os._exit() in __main__ below. Double-click,
    nothing happens: no window, no dialog, no Windows Event Log entry. This is the single most
    "frozen" a startup failure can look, since there is no process left to even seem stuck.

    ctypes straight to user32 rather than a GUI toolkit: nothing else in this file needs one,
    and MessageBoxW is a couple of lines that work from a windowless process with no event loop
    of its own already running (unlike e.g. tkinter, which wants one). Deliberately blocking --
    that's what keeps the dialog on screen instead of vanishing with the process, and it does
    not hold the port: acquire_port()'s socket close (or the process never having reserved one
    yet) happens before main() can raise past this call, so a launch that fails this way can
    never itself be the reason a *later* launch can't bind.

    Never allowed to raise: a broken dialog must not hide the real error this exists to surface,
    which the caller still prints to ragpoc.log regardless of whether this succeeds.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        # Last non-empty line of the traceback: the exception's own message, which for
        # everything main() raises (see wait_for_server, the migration check) is already a
        # user-facing sentence in Spanish -- more useful here than the full stack.
        last_line = next((line for line in reversed(detail.strip().splitlines()) if line.strip()), detail)
        message = (
            f"RAGPoC no pudo iniciarse.\n\n{last_line}\n\n"
            f"Detalles completos en:\n{_log_path()}\n\n"
            "Si el problema continúa, comparte ese archivo para recibir ayuda."
        )
        MB_ICONERROR = 0x10
        MB_OK = 0x0
        ctypes.windll.user32.MessageBoxW(None, message, "RAGPoC — Error de inicio", MB_OK | MB_ICONERROR)
    except Exception:
        pass


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


# 176x176 PNG of assets/ragpoc_logo.png (the same mark PyInstaller embeds as the .exe's own
# icon -- ragpoc.spec's EXE(icon=...) and installer.iss's SetupIconFile both point at the .ico
# version of this same image), re-encoded here as a data URI. Has to be embedded rather than
# referenced by path or URL: this splash is shown via create_window(html=...) specifically
# *before* any server exists to serve a file from, and pywebview's window has no filesystem
# access of its own to fall back on. Regenerate with:
#   from PIL import Image; import base64, io
#   im = Image.open("assets/ragpoc_logo.png").convert("RGBA").resize((176, 176), Image.LANCZOS)
#   buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
#   base64.b64encode(buf.getvalue()).decode("ascii")
_SPLASH_LOGO_B64 = (
    "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"
)

# Colors and font stack lifted straight from console.html's :root token block (--canvas-bg,
# --text-main/-secondary/-muted, --border-warm, --accent-warm) rather than invented separately,
# so the splash reads as the first frame of the same app instead of a generic loading screen
# that happens to precede it. accent-warm specifically, not accent-craft or accent-amber: it's
# the one actually driving interactive/active states throughout console.html (10 uses there
# against 2 each for the other two), so it's the color a user already associates with "this app
# is doing something" by the time they've used it once.
#
# No @media (prefers-color-scheme: dark) block: console.html itself is light-only (confirmed --
# it has zero dark-mode handling of its own beyond one dropdown variant), so a themed splash
# would be inventing a dark identity the app immediately contradicts one frame later.
_SPLASH_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>RAGPoC</title><style>
  html,body{height:100%;margin:0;background:#f9f8f5;color:#1c1b18;
    font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,Roboto,Arial,sans-serif;overflow:hidden}
  .wrap{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:30px}
  .brand{display:flex;flex-direction:column;align-items:center;gap:16px}
  .logo{width:84px;height:84px;border-radius:19px;display:block;
    box-shadow:0 4px 14px rgba(32,28,20,.10),0 1px 3px rgba(32,28,20,.06);
    border:1px solid #e6e2d8}
  .mark{font-size:13px;font-weight:700;letter-spacing:.15em;color:#1c1b18}
  .progress{display:flex;flex-direction:column;align-items:center;gap:11px}
  .spin{width:20px;height:20px;border-radius:50%;border:2px solid #e6e2d8;
    border-top-color:#2b6cb0;animation:spin .8s linear infinite}
  @media (prefers-reduced-motion: reduce){.spin{animation:none;border-top-color:#e6e2d8;border-right-color:#2b6cb0}}
  #status{font-size:12.5px;color:#8e897d;min-height:16px}
  @keyframes spin{to{transform:rotate(360deg)}}
</style></head><body>
  <div class="wrap">
    <div class="brand">
      <img class="logo" src="data:image/png;base64,__RAGPOC_SPLASH_LOGO__" alt="RAGPoC">
      <div class="mark">RAGPOC</div>
    </div>
    <div class="progress">
      <div class="spin"></div>
      <div id="status">Iniciando…</div>
    </div>
  </div>
  <script>
    // Called from Python via window.evaluate_js() while this splash is showing, before
    // load_url() replaces it with the real app -- see _run_boot()'s on_progress callback.
    window.setStatus = function (text) {
      var el = document.getElementById('status');
      if (el) el.textContent = text;
    };
  </script>
</body></html>"""

# A plain string template, not an f-string: the HTML/CSS above is dense with literal { }
# (every CSS rule, the inline <script> block) that would otherwise all need doubling to
# escape them from Python's f-string interpolation -- one stray un-doubled brace anywhere
# in that block is a SyntaxError at import time, i.e. the app fails to start at all. A
# plain .replace() on one unambiguous placeholder token sidesteps that class of mistake
# entirely.
_SPLASH_HTML = _SPLASH_HTML_TEMPLATE.replace("__RAGPOC_SPLASH_LOGO__", _SPLASH_LOGO_B64)


def _run_boot(state: dict, on_progress) -> None:
    """Does everything main() used to do inline before any window existed: claim a port, stand
    up Django, run migrations (with the D-03 database sanity check), warm the URL resolver, and
    start uvicorn. Extracted so the native-splash path (this runs in a background thread while
    the splash window is already on screen) and the no-webview-backend fallback (this runs
    exactly as it always did, before anything is shown) share one implementation rather than
    drift apart as two copies.

    Always sets state["url"]. Only sets state["server"] / state["server_thread"] when this
    launch actually started a server -- the "attach to an already-running instance" case has
    nothing of its own to serve or later shut down, so the caller uses their absence to tell
    the two cases apart.

    Raises on genuine failure (bad migration state, uvicorn never coming up); callers decide
    what "genuine failure" should look like to the user (see _show_startup_failure)."""
    import django
    from django.core.management import call_command

    from ragpoc.config import get_settings

    host = HOST
    # Claim the port before the expensive Django startup below, not after: holding the socket
    # from here on removes any window in which another process could take it, and the answer
    # also decides whether that startup is needed at all.
    server_socket, port = acquire_port(host, PREFERRED_PORT)
    url = f"http://{host}:{port}/"
    state["url"] = url

    if server_socket is None:
        # The app is already open — someone launched it a second time. Two instances would each
        # run their own server against the same sqlite file and fight over it, so this launch
        # contributes a window and nothing else, which is also why it can skip django.setup()
        # entirely and appear almost instantly.
        on_progress(f"RAGPoC ya se está ejecutando en {url}; se abre otra ventana sobre esa instancia.")
        return

    # The sqlite file (and uploads/renders/derived dirs) live under a data/
    # folder that may not exist yet on a fresh install — Django's sqlite
    # backend errors with "unable to open database file" if the parent
    # directory is missing, so create it before django.setup()/migrate touch it.
    get_settings().ensure_directories()

    django.setup()

    # Run auto-migrations on startup
    try:
        on_progress("Verificando base de datos y migraciones...")
        call_command("migrate", interactive=False)
    except Exception as e:
        print(f"Aviso en migraciones: {e}")
        # Not every migrate failure means the database is actually unusable -- some are benign
        # and this except has always tolerated them on purpose (see the print above). But
        # swallowing all of them unconditionally meant a *fatal* one (a corrupted file, a locked
        # database) let the app carry on into a window that opens, paints fully, and then sits
        # on "Cargando espacio…" forever -- every API view failing with 500 the moment it
        # touches the ORM, with nothing to tell the user why. So: try one trivial query against
        # a real model. If the schema is actually broken this fails too, and *that* is worth
        # stopping for; if it succeeds, the migrate warning above was noise.
        try:
            from knowledge.models import Workspace

            Workspace.objects.exists()
        except Exception as verify_error:
            raise RuntimeError(
                "La base de datos de RAGPoC no se pudo preparar correctamente "
                f"({verify_error}). Revisa ragpoc.log para más detalles."
            ) from verify_error

    # Django only imports urls.py (and therefore views.py, and therefore pydantic_ai and every
    # other heavy dependency views.py pulls in) the first time it needs to resolve a URL --
    # migrate above never touches it. Left alone, that import (~4-5s, mostly pydantic_ai) would
    # happen lazily on the *first HTTP request the webview window makes*, i.e. right after the
    # window is already visible to the user -- it would sit there looking open but unresponsive
    # for several seconds. Forcing it here instead pays that cost during the splash period,
    # before wait_for_server()/load_url() below, so the window is instantly usable the moment
    # it stops showing the splash.
    on_progress("Preparando la aplicación...")
    from django.urls import get_resolver

    get_resolver().url_patterns

    print("=" * 60)
    print("🚀 RAGPoC Desktop / Knowledge Studio")
    print(f"📍 Servidor local activo en: {url}")
    print("=" * 60)

    from ragpoc_django.asgi import application

    # uvicorn runs in a background thread so this thread is free to keep polling wait_for_server
    # below (and, in the native-splash path, so the actual GUI loop on the main thread stays
    # free too). It serves the socket acquired above rather than binding its own; host/port here
    # only feed its log lines.
    server_config = uvicorn.Config(application, host=host, port=port, log_level="info")
    server = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server.run, kwargs={"sockets": [server_socket]}, daemon=True)
    server_thread.start()

    wait_for_server(host, port, server_thread)

    state["server"] = server
    state["server_thread"] = server_thread


def main():
    from ragpoc.clr_host import ensure_clr_host_config
    from ragpoc.updater import cleanup_stale_update_files, unblock_downloaded_install

    # Compiled/local desktop build: single local user on 127.0.0.1, no login screen. Set here
    # rather than at module import time: this module is imported (not run) by anything that
    # does `import desktop_launcher`, including test collection, and a module-level
    # os.environ[...] = "1" would flip every other test's Settings() to desktop_mode=True for
    # the rest of that process — silently disabling BasicAuthMiddleware for tests that never
    # touch this module directly. main() only ever runs for a real launch, where this belongs.
    os.environ["RAGPOC_DESKTOP_MODE"] = "1"

    # Sweeps any *.exe.old left behind by a self-update whose final cleanup step lost a file
    # lock race (see ragpoc.updater._write_updater_script) — by now that lock is long gone.
    cleanup_stale_update_files()

    # Both of these make the native window below survive being installed from a downloaded
    # zip, whose files Windows tags as untrusted: the first tells the .NET runtime to load
    # pythonnet's assemblies anyway, the second strips the tags. They must run before anything
    # imports webview, since that is what starts the CLR that reads the config file.
    ensure_clr_host_config()
    unblock_downloaded_install()

    # Filled in by _run_boot(): "url" always; "server"/"server_thread" only for a launch that
    # actually started one (not the attach-to-existing-instance case). Shared across the
    # native-splash thread below and the code that runs after it, since a plain local variable
    # inside the boot() closure wouldn't be visible out here.
    state: dict = {}

    def boot():
        # Runs on the thread pywebview hands to `func` once its GUI loop is already live and
        # the splash is already on screen — see webview.start() below. Never lets an exception
        # escape: everything this calls is now reachable from a state where a real window
        # exists but nothing behind it works yet, and the old behavior for that (silently
        # dying, see D-02) is exactly what the dialog + destroy() here replaces. Because this
        # never re-raises, an exception surfacing from webview.start() itself afterwards can
        # only be the native backend failing, never an app-boot failure — that distinction is
        # what lets the fallback below decide whether _run_boot() needs to run again.
        def on_progress(msg):
            # Both, not just the splash: ragpoc.log is what a user actually sends back when
            # asked, and losing these two lines from it (they used to be unconditional prints)
            # would make "which stage did it get stuck on" one log grep harder to answer.
            print(msg)
            window.evaluate_js(f"setStatus({json.dumps(msg)})")

        try:
            _run_boot(state, on_progress=on_progress)
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            _show_startup_failure(tb)
            state["failed"] = True
            try:
                window.destroy()
            except Exception:
                pass
            return
        window.load_url(state["url"])

    window = None
    native_ok = False
    try:
        import webview

        # pywebview's WebView2 backend silently cancels every download (no dialog, no error,
        # nothing) unless this is turned on -- it's what made "Guardar"/"Descargar" on artifacts
        # look like it did nothing. With it enabled, WebView2 shows a native SaveFileDialog for
        # every download the page triggers (the <a download> / blob-URL clicks in console.html).
        webview.settings["ALLOW_DOWNLOADS"] = True

        # html=, not url=: no server exists yet at this point, and waiting for one before the
        # window can show anything is the entire problem this splash exists to fix (django.setup,
        # migrations and the URL-resolver warm-up below routinely take several seconds with
        # nothing on screen). boot() replaces this with the real app via load_url() once ready.
        window = webview.create_window(
            "RAGPoC — Knowledge Studio", html=_SPLASH_HTML, width=1400, height=900, min_size=(960, 640)
        )
        # Blocks until every window is closed. The backend actually initializes here (not at
        # create_window() above) — a broken WebView2/pythonnet CLR raises from this call, before
        # boot ever runs, landing in the except below with the server never having been started.
        webview.start(boot)
        native_ok = True
    except Exception as e:
        # No native webview backend available (e.g. the WebView2 runtime really is missing).
        # Log the whole traceback, not just str(e): every past instance of this has been a
        # CLR/pythonnet load failure whose one-line message named a symptom rather than a
        # cause, and this log file is the only diagnostic a user can send back.
        print(f"No se pudo abrir la ventana nativa ({e}); usando una ventana del navegador…")
        print(traceback.format_exc())

    if native_ok:
        if state.get("failed"):
            return  # already reported via _show_startup_failure() inside boot()
        server = state.get("server")
        server_thread = state.get("server_thread")
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=5)
        return

    # --- No native window at all: the original, splash-less flow, unchanged in spirit. ---
    if not state.get("url"):
        # boot() never ran (the backend failed before webview.start() ever invoked it, the
        # common case for a missing WebView2 runtime) -- so nothing has been booted yet.
        _run_boot(state, on_progress=print)
    # else: boot() already fully succeeded and started a real server before the backend gave
    # out from under it after load_url() (native_ok stays False if webview.start() itself still
    # raised afterwards) -- reuse that server instead of starting a second one against the same
    # sqlite file.

    url = state["url"]
    server = state.get("server")
    server_thread = state.get("server_thread")

    if server_thread is None:
        # Already-running-instance case: nothing of our own to serve, so there's no server
        # thread to hold the process open below -- just show a window onto it and exit.
        if not open_browser_app_window(url):
            print("Sin navegador compatible; abriendo una pestaña en el navegador predeterminado…")
            webbrowser.open(url)
        return

    if not open_browser_app_window(url):
        print("Sin navegador compatible; abriendo una pestaña en el navegador predeterminado…")
        webbrowser.open(url)
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
        _show_startup_failure(traceback.format_exc())

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
