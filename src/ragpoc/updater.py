from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from packaging.version import Version

from ragpoc import __version__
from ragpoc.http import SSL_CONTEXT, new_async_client
from ragpoc.http import get as http_get

GITHUB_REPO = "AngelDann/RAGPoC-App-Code"
_RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_ALLOWED_DOWNLOAD_HOSTS = {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}


class UpdateError(RuntimeError):
    pass


class UpdateState:
    IDLE = "idle"
    CHECKING = "checking"
    UPDATE_AVAILABLE = "update_available"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    READY_TO_INSTALL = "ready_to_install"
    APPLYING = "applying"
    ERROR = "error"


def get_current_platform() -> str:
    """Detects the current operating system / runtime environment:
    - 'android': Android OS (Chaquopy, BeeWare, Termux, or Android runtime markers)
    - 'windows': Windows desktop (win32)
    - 'macos': macOS (darwin)
    - 'linux': Generic Linux desktop / server
    """
    if (
        hasattr(sys, "getandroidapilevel")
        or "ANDROID_ROOT" in os.environ
        or "ANDROID_BOOTLOGO" in os.environ
        or "TERMUX_VERSION" in os.environ
    ):
        return "android"
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return "unknown"


def select_release_asset(assets: list[dict], target_os: str) -> dict | None:
    """Finds the most appropriate downloadable release asset for the target OS."""
    patterns = {
        "windows": ["-windows.zip", ".zip"],
        "android": ["-android.apk", ".apk"],
        "linux": ["-linux.tar.gz", ".tar.gz", ".appimage", ".deb"],
        "macos": ["-macos.dmg", ".dmg", "-darwin.zip"],
    }
    candidates = patterns.get(target_os, [f"-{target_os}"])
    for ext in candidates:
        for a in assets:
            name = (a.get("name") or "").lower()
            if name.endswith(ext):
                return a
    # Fallback: if windows requested, also check for loose .exe if no zip
    if target_os == "windows":
        for a in assets:
            if (a.get("name") or "").lower().endswith(".exe") and not a.get("name", "").lower().startswith("ragpoc-setup"):
                return a
    return None


class UpdateManager:
    """Manages the lifecycle of background downloading, staging, and deferred execution of app updates."""

    def __init__(self) -> None:
        self.state: str = UpdateState.IDLE
        self.target_version: str | None = None
        self.download_url: str | None = None
        self.asset_name: str | None = None
        self.release_notes: str = ""
        self.release_url: str | None = None
        self.platform: str = get_current_platform()
        self.bytes_downloaded: int = 0
        self.total_bytes: int = 0
        self.progress_percent: float = 0.0
        self.error_message: str | None = None
        self.staging_dir: Path | None = None
        self.zip_path: Path | None = None
        self._download_task: asyncio.Task | None = None
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def get_status(self) -> dict:
        return {
            "current_version": __version__,
            "state": self.state,
            "target_version": self.target_version or __version__,
            "download_url": self.download_url,
            "asset_name": self.asset_name,
            "platform": self.platform,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "progress_percent": round(self.progress_percent, 1),
            "ready_to_install": self.state == UpdateState.READY_TO_INSTALL,
            "error_message": self.error_message,
        }

    async def check(self, target_os: str | None = None) -> dict:
        current_os = target_os or get_current_platform()
        self.platform = current_os
        info = await check_for_update(target_os=current_os)
        if info.get("update_available"):
            self.target_version = info.get("latest_version")
            self.download_url = info.get("download_url")
            self.asset_name = info.get("asset_name")
            self.release_notes = info.get("release_notes", "")
            self.release_url = info.get("release_url")
            if self.state not in (UpdateState.DOWNLOADING, UpdateState.EXTRACTING, UpdateState.READY_TO_INSTALL):
                self.state = UpdateState.UPDATE_AVAILABLE
        else:
            if self.state not in (UpdateState.DOWNLOADING, UpdateState.EXTRACTING, UpdateState.READY_TO_INSTALL):
                self.state = UpdateState.IDLE
        info["state"] = self.state
        info["ready_to_install"] = self.state == UpdateState.READY_TO_INSTALL
        info["progress_percent"] = round(self.progress_percent, 1)
        info["bytes_downloaded"] = self.bytes_downloaded
        info["total_bytes"] = self.total_bytes
        return info

    async def start_download(self, download_url: str | None = None, target_version: str | None = None) -> dict:
        url = download_url or self.download_url
        if not url:
            raise UpdateError("download_url es requerido para iniciar la descarga.")
        if urlparse(url).hostname not in _ALLOWED_DOWNLOAD_HOSTS:
            raise UpdateError("download_url debe apuntar a un asset de GitHub Releases.")

        ver = target_version or self.target_version or "latest"

        async with self._get_lock():
            if self.state in (UpdateState.DOWNLOADING, UpdateState.EXTRACTING):
                return self.get_status()
            if self.state == UpdateState.READY_TO_INSTALL and self.staging_dir and self.staging_dir.exists():
                return self.get_status()

            self.state = UpdateState.DOWNLOADING
            self.download_url = url
            self.target_version = ver
            self.bytes_downloaded = 0
            self.total_bytes = 0
            self.progress_percent = 0.0
            self.error_message = None

            loop = asyncio.get_running_loop()
            self._download_task = loop.create_task(self._run_download_and_stage(url, ver))

        return self.get_status()

    async def _run_download_and_stage(self, download_url: str, version: str) -> None:
        tmp_dir = Path(tempfile.gettempdir())
        pid = os.getpid()
        zip_path = tmp_dir / f"ragpoc_update_staged_{pid}.zip"
        staging_dir = tmp_dir / f"ragpoc_update_staging_{pid}"
        self.zip_path = zip_path
        self.staging_dir = staging_dir

        current_os = get_current_platform()
        try:
            if current_os == "android":
                apk_path = tmp_dir / "ragpoc_update.apk"
                async with new_async_client(timeout=300) as client:
                    async with client.stream("GET", download_url, follow_redirects=True) as response:
                        response.raise_for_status()
                        total = int(response.headers.get("content-length", 0))
                        self.total_bytes = total
                        with apk_path.open("wb") as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                                self.bytes_downloaded += len(chunk)
                                if total > 0:
                                    self.progress_percent = (self.bytes_downloaded / total) * 100.0
                self.state = UpdateState.READY_TO_INSTALL
                self.progress_percent = 100.0
                return

            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir.mkdir(parents=True, exist_ok=True)

            async with new_async_client(timeout=300) as client:
                async with client.stream("GET", download_url, follow_redirects=True) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    self.total_bytes = total
                    with zip_path.open("wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            self.bytes_downloaded += len(chunk)
                            if total > 0:
                                self.progress_percent = (self.bytes_downloaded / total) * 100.0

            self.state = UpdateState.EXTRACTING
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(staging_dir)

            self.state = UpdateState.READY_TO_INSTALL
            self.progress_percent = 100.0
        except asyncio.CancelledError:
            self._cleanup_download_files()
            self.state = UpdateState.UPDATE_AVAILABLE
            raise
        except Exception as e:
            self._cleanup_download_files()
            self.state = UpdateState.ERROR
            self.error_message = str(e)

    def _cleanup_download_files(self) -> None:
        if self.staging_dir and self.staging_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)
        if self.zip_path and self.zip_path.exists():
            try:
                self.zip_path.unlink()
            except OSError:
                pass

    async def cancel_download(self) -> dict:
        async with self._get_lock():
            if self._download_task and not self._download_task.done():
                self._download_task.cancel()
                try:
                    await self._download_task
                except asyncio.CancelledError:
                    pass
            self._cleanup_download_files()
            self.state = UpdateState.UPDATE_AVAILABLE if self.target_version else UpdateState.IDLE
            self.bytes_downloaded = 0
            self.total_bytes = 0
            self.progress_percent = 0.0
            self.error_message = None
        return self.get_status()

    async def apply_staged_update(self) -> None:
        current_os = get_current_platform()
        if current_os == "android":
            return

        if not getattr(sys, "frozen", False):
            raise UpdateError("El auto-actualizador solo funciona en la build compilada (.exe).")

        if self.state != UpdateState.READY_TO_INSTALL or not self.staging_dir or not self.staging_dir.exists():
            raise UpdateError("No hay ninguna actualización descargada lista para instalar.")

        current_exe = Path(sys.executable).resolve()
        current_dir = current_exe.parent
        pid = os.getpid()

        script_path = _write_updater_script(
            pid=pid,
            current_exe=current_exe,
            staging_exe=self.staging_dir / current_exe.name,
            current_internal=current_dir / "_internal",
            staging_internal=self.staging_dir / "_internal",
            staging_dir=self.staging_dir,
            zip_path=self.zip_path or Path(tempfile.gettempdir()) / f"ragpoc_update_staged_{pid}.zip",
        )

        self.state = UpdateState.APPLYING

        subprocess.Popen(
            ["cmd", "/c", str(script_path)],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        threading.Timer(1.5, lambda: os._exit(0)).start()


updater_manager = UpdateManager()


async def check_for_update(target_os: str | None = None) -> dict:
    """Queries GitHub Releases for the latest published version. Works whether or not the app
    is frozen (only apply_update() requires a frozen build), so it degrades gracefully when run
    from source -- update_available will just never fire since __version__ tracks HEAD there.

    Async, not the sync ragpoc.http.get used elsewhere in this module: this fires unconditionally
    on every desktop app startup (see console.html's checkForUpdate()), racing the DOM-ready
    workspace load. A sync view here would tie up Django's single thread-sensitive worker thread
    for the whole GitHub round-trip (up to the 10s timeout on a slow/blocked connection) --
    since ASGI serializes *every* sync view onto that one thread, the workspace tree/page fetch
    would queue up behind it, and the window would sit unresponsive until it finished."""
    current_os = target_os or get_current_platform()
    async with new_async_client() as client:
        response = await client.get(_RELEASES_API, timeout=10, headers={"Accept": "application/vnd.github+json"})
    response.raise_for_status()
    data = response.json()
    latest_tag = (data.get("tag_name") or "").lstrip("v")
    
    asset = select_release_asset(data.get("assets", []), current_os)

    available = bool(latest_tag) and Version(latest_tag) > Version(__version__)
    return {
        "current_version": __version__,
        "latest_version": latest_tag or __version__,
        "update_available": available,
        "platform": current_os,
        "asset_name": asset["name"] if asset else None,
        "download_url": asset["browser_download_url"] if asset else None,
        "release_notes": data.get("body") or "",
        "release_url": data.get("html_url"),
    }


async def apply_update(download_url: str) -> None:
    """Downloads the new build and initiates the platform-specific update process.
    If the update is already staged (READY_TO_INSTALL), it applies the staged update directly.
    Otherwise downloads to staging and initiates the detached batch swap script.
    """
    if updater_manager.state == UpdateState.READY_TO_INSTALL and updater_manager.staging_dir and updater_manager.staging_dir.exists():
        await updater_manager.apply_staged_update()
        return

    current_os = get_current_platform()
    if urlparse(download_url).hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise UpdateError("download_url debe apuntar a un asset de GitHub Releases.")

    if current_os == "android":
        tmp_dir = Path(tempfile.gettempdir())
        apk_path = tmp_dir / "ragpoc_update.apk"
        async with new_async_client(timeout=180) as client:
            async with client.stream("GET", download_url, follow_redirects=True) as response:
                response.raise_for_status()
                with apk_path.open("wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
        return

    if not getattr(sys, "frozen", False):
        raise UpdateError("El auto-actualizador solo funciona en la build compilada (.exe).")

    current_exe = Path(sys.executable).resolve()
    current_dir = current_exe.parent
    tmp_dir = Path(tempfile.gettempdir())
    pid = os.getpid()
    # PID-scoped, not fixed names: a prior update attempt's detached swap script survives this
    # process (that's the point -- it waits out our exit before touching files) and can still be
    # sitting in its own wait loop, e.g. if the app was force-closed mid-update and relaunched. A
    # second apply_update() reusing the same fixed temp paths would overwrite that other script's
    # zip/staging/.bat out from under it mid-read -- observed in testing as a silent, unlogged
    # failure (the running cmd interpreter doesn't lock the .bat file it's reading). Scoping every
    # temp path to our own pid makes concurrent/leftover attempts independent instead of colliding.
    zip_path = tmp_dir / f"ragpoc_update_{pid}.zip"
    staging_dir = tmp_dir / f"ragpoc_update_staging_{pid}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    async with new_async_client(timeout=180) as client:
        async with client.stream("GET", download_url, follow_redirects=True) as response:
            response.raise_for_status()
            with zip_path.open("wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging_dir)

    script_path = _write_updater_script(
        pid=pid,
        current_exe=current_exe,
        staging_exe=staging_dir / current_exe.name,
        current_internal=current_dir / "_internal",
        staging_internal=staging_dir / "_internal",
        staging_dir=staging_dir,
        zip_path=zip_path,
    )
    # CREATE_NO_WINDOW, not DETACHED_PROCESS: detached gives cmd.exe no console at all, and cmd
    # runs every `|` pipeline stage by re-launching itself, which fails without one -- cmd died
    # on the swap script's very first pipeline and nothing after it ever ran (silently, since the
    # failure log line was itself past that point). CREATE_NO_WINDOW still shows no window but
    # gives the child its own console, and an explicitly-detached parent isn't needed for it to
    # outlive us. The DEVNULL handles matter too: a windowed build's own std handles are invalid,
    # and children inherit them by default.
    subprocess.Popen(
        ["cmd", "/c", str(script_path)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    # Give the current HTTP response time to flush back to the browser/webview before this
    # process (and the uvicorn thread serving it) disappears out from under the connection.
    threading.Timer(1.5, lambda: os._exit(0)).start()


def _write_updater_script(
    pid: int,
    current_exe: Path,
    staging_exe: Path,
    current_internal: Path,
    staging_internal: Path,
    staging_dir: Path,
    zip_path: Path,
) -> Path:
    # Two parts to every swap: the bulk of the build (_internal\, dozens of DLLs/data files) and
    # the exe itself. They need different techniques.
    #
    # _internal\ is synced with `robocopy /MIR`: by the time this runs, the old process (whose
    # pid we waited out below) has already exited, so nothing has these DLLs open the way a
    # running process holds its own .exe -- a plain in-place overwrite is fine, no rename-first
    # dance needed. robocopy's /R and /W give it its own built-in retry against a transient
    # lock/scan (e.g. antivirus briefly touching a freshly-written DLL), so no hand-rolled retry
    # loop is needed here either. Its exit codes do NOT follow normal cmd conventions though --
    # 0-7 all mean success (0 = nothing changed, 1 = files copied, etc.), only >=8 is a real
    # failure, hence `if %errorlevel% GEQ 8` rather than `if errorlevel 1`.
    #
    # The exe still needs the rename-first pattern (the pattern mature Windows auto-updaters use,
    # e.g. Squirrel.Windows): Windows allows *renaming* a running process's own exe even while
    # still mapped/executing (unlike overwriting it) -- but that only matters while it's still
    # running, and by this point in the script it no longer is. The rename-first swap is kept
    # anyway since it's already proven reliable and gives a rollback path (old_backup) for free.
    # A single unretried move here previously lost races against Windows/antivirus locks
    # silently (nothing checked its exit code); both moves below have their own retry loop for
    # the same reason robocopy gets one -- new_exe just landed from the internet and is unsigned
    # (no Authenticode signing in release.yml), which is exactly what Defender/SmartScreen scans
    # on-write. 60 attempts (~60s), not 15 (~15s): live testing showed a freshly-exited process's
    # own exe staying locked well past 15s on a real machine often enough that most real update
    # attempts failed silently this way -- 60s matches the pid-wait loop's own budget above and
    # gave every attempt in that same testing a clean, unassisted swap. The renamed-away old exe
    # is deleted best-effort; if that step fails because of a lingering handle, main() sweeps up
    # any leftover *.exe.old on the next app start (see cleanup_stale_update_files), once that
    # lock is long gone.
    #
    # Two commands are deliberately avoided throughout the script because they need a console,
    # which a background-launched updater can easily end up without (see apply_update):
    #   - `a | b` pipelines: cmd runs each stage by re-launching itself, and dies outright if it
    #     can't. The old `tasklist | find` wait loop killed the script on its first line, which is
    #     why every swap silently no-op'd and left a stray copy behind -- redirect to a temp file
    #     and `find` in that file instead.
    #   - `timeout`: exits immediately with "input redirection is not supported" when stdin isn't
    #     a console, turning every retry pause into a busy spin -- `ping -n 2 127.0.0.1` sleeps
    #     ~1s with no such requirement.
    # Both hand-rolled retry loops re-expand %attempts% correctly despite being inside
    # parenthesised blocks: `goto` makes cmd re-read and re-parse from the label each pass, so
    # the value is never stale.
    log_path = current_exe.with_name("ragpoc.log")
    old_backup = current_exe.with_suffix(current_exe.suffix + ".old")
    # pid-scoped, like the zip/staging paths in apply_update() -- see the comment there.
    pid_probe = Path(tempfile.gettempdir()) / f"ragpoc_update_pid_{pid}.txt"
    script_path = Path(tempfile.gettempdir()) / f"ragpoc_update_{pid}.bat"
    script_path.write_text(
        "@echo off\r\n"
        # Logged before anything can go wrong, so a swap that dies early leaves a trace instead of
        # looking like the updater was never invoked at all.
        f'echo [%date% %time%] Update: swap script started, waiting for pid {pid} to exit >> "{log_path}"\r\n'
        "set attempts=0\r\n"
        ":wait\r\n"
        f'tasklist /nh /fo csv /fi "PID eq {pid}" > "{pid_probe}" 2>nul\r\n'
        f'find "{pid}" "{pid_probe}" >nul 2>&1\r\n'
        "if not errorlevel 1 (\r\n"
        "  set /a attempts+=1\r\n"
        "  if %attempts% LSS 60 (\r\n"
        "    ping -n 2 127.0.0.1 >nul 2>&1\r\n"
        "    goto wait\r\n"
        "  )\r\n"
        f'  echo [%date% %time%] Update failed: pid {pid} still running after 60s, aborting swap >> "{log_path}"\r\n'
        "  goto cleanup\r\n"
        ")\r\n"
        f'del /f /q "{pid_probe}" >nul 2>&1\r\n'
        f'if exist "{old_backup}" del /f /q "{old_backup}" >nul 2>&1\r\n'
        f'robocopy "{staging_internal}" "{current_internal}" /MIR /R:15 /W:1 /NFL /NDL /NJH /NJS /NP >nul 2>&1\r\n'
        "set robo_rc=%errorlevel%\r\n"
        "if %robo_rc% GEQ 8 (\r\n"
        f'  echo [%date% %time%] Update failed: robocopy could not sync _internal, exit=%robo_rc% >> "{log_path}"\r\n'
        "  goto relaunch\r\n"
        ")\r\n"
        # No literal parentheses in this text (or in any echo that follows a multi-line `if (...)`
        # block, throughout this script): cmd.exe's block parser counts parens to find where an
        # `if (...)` ends, and unescaped `(`/`)` in plain (unquoted) echo text -- even on an
        # unrelated later line -- can desync that count and silently corrupt everything parsed
        # after it. Confirmed by direct testing: this single line, immediately after the robocopy
        # if-block above, was enough on its own to make the entire rest of the script (both exe
        # moves, the relaunch) execute with no trace in the log and no error anywhere -- cmd just
        # silently skipped straight to relaunching the *old*, unswapped exe.
        f'echo [%date% %time%] Update: robocopy synced _internal, exit=%robo_rc% >> "{log_path}"\r\n'
        "set attempts=0\r\n"
        ":rename_retry\r\n"
        # >> log, not >nul: testing found `move` redirected to nul unreliable specifically when
        # the parent cmd.exe's own stdout/stderr are themselves DEVNULL (as they are here -- see
        # apply_update's Popen call) -- a nested nul-device redirect under an already-nul parent
        # handle intermittently made the move fail (or the whole script stall) with no error
        # surfaced anywhere, which is exactly the silent no-op this comment used to warn about
        # further up. Redirecting to a real file sidesteps that nul-handle interaction entirely
        # and is strictly more useful (move's only output is "N file(s) moved." or the real error).
        f'move /y "{current_exe}" "{old_backup}" >> "{log_path}" 2>&1\r\n'
        "if errorlevel 1 (\r\n"
        "  set /a attempts+=1\r\n"
        "  if %attempts% LSS 60 (\r\n"
        "    ping -n 2 127.0.0.1 >nul 2>&1\r\n"
        "    goto rename_retry\r\n"
        "  )\r\n"
        f'  echo [%date% %time%] Update failed: could not rename "{current_exe}" out of the way after 60 attempts >> "{log_path}"\r\n'
        "  goto relaunch\r\n"
        ")\r\n"
        f'echo [%date% %time%] Update: renamed current exe out of the way to {old_backup.name}, %attempts% attempts >> "{log_path}"\r\n'
        f'for %%A in ("{staging_exe}") do set new_exe_size=%%~zA\r\n'
        "set attempts=0\r\n"
        ":install_retry\r\n"
        f'move /y "{staging_exe}" "{current_exe}" >> "{log_path}" 2>&1\r\n'
        "if errorlevel 1 (\r\n"
        "  set /a attempts+=1\r\n"
        "  if %attempts% LSS 60 (\r\n"
        "    ping -n 2 127.0.0.1 >nul 2>&1\r\n"
        "    goto install_retry\r\n"
        "  )\r\n"
        f'  echo [%date% %time%] Update failed: could not move new exe into place after 60 attempts, restoring previous version >> "{log_path}"\r\n'
        f'  move /y "{old_backup}" "{current_exe}" >> "{log_path}" 2>&1\r\n'
        "  goto relaunch\r\n"
        ")\r\n"
        f'del /f /q "{old_backup}" >nul 2>&1\r\n'
        # Logs the ACTUAL resulting file size, not just "the move command reported success" -- a
        # move can report success while something else (AV quarantine, a sync client) touches the
        # file microseconds later, and that would otherwise look identical to a clean install in
        # this log. %%~zA reads the size Windows currently has on disk for the file, checked right
        # after the move so any such interference shows up as a mismatch against the source size
        # captured just before the move (new_exe_size, above).
        f'for %%A in ("{current_exe}") do set installed_exe_size=%%~zA\r\n'
        f'echo [%date% %time%] Update: new build installed, staged size=%new_exe_size% installed size=%installed_exe_size% >> "{log_path}"\r\n'
        ":relaunch\r\n"
        # A brief pause before the first launch of the exe that was JUST written to disk. An
        # AV/EDR scan (or any other transient handle on the freshly-written files) overlapping
        # the very first launch is exactly the kind of race this pause, and the onedir switch
        # itself (no more self-extraction step at all), both guard against.
        "ping -n 4 127.0.0.1 >nul 2>&1\r\n"
        f'start "" "{current_exe}"\r\n'
        ":cleanup\r\n"
        f'rd /s /q "{staging_dir}" >nul 2>&1\r\n'
        f'del /f /q "{zip_path}" >nul 2>&1\r\n'
        f'del /f /q "{pid_probe}" >nul 2>&1\r\n'
        f'del "%~f0"\r\n',
        encoding="utf-8",
    )
    return script_path


def cleanup_stale_update_files() -> None:
    """Best-effort sweep for *.exe.old leftovers and stale temp staging files from an update whose
    final cleanup step lost a lock race (see _write_updater_script) -- called on every startup of
    the frozen build, by which point whatever process held the file locked has long since exited.
    Never raises: this is opportunistic housekeeping, not something that should ever block startup.
    Safe to run unconditionally: it only fires after the app is already up and running again, i.e.
    never while an update is actually in flight."""
    if not getattr(sys, "frozen", False):
        return
    try:
        current_exe = Path(sys.executable).resolve()
        for stale in current_exe.parent.glob("*.exe.old"):
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass

    try:
        tmp_dir = Path(tempfile.gettempdir())
        for stale_staging in tmp_dir.glob("ragpoc_update_staging_*"):
            if stale_staging.is_dir():
                shutil.rmtree(stale_staging, ignore_errors=True)
        for stale_zip in tmp_dir.glob("ragpoc_update_*.zip"):
            if stale_zip.is_file():
                try:
                    stale_zip.unlink()
                except OSError:
                    pass
    except OSError:
        pass


def unblock_downloaded_install() -> None:
    """A release zip downloaded via a browser and extracted with Windows Explorer gets every
    extracted file tagged with the NTFS "Mark of the Web" (a hidden Zone.Identifier=Internet
    stream). .NET Framework then refuses to load pythonnet's Python.Runtime.dll out of that
    install folder -- pywebview treats that as "no native backend available" and falls back to
    opening the OS browser instead of the desktop window. Stripping the zone tag from every
    file fixes it without requiring anyone to know to right-click the zip and choose Unblock.

    Second line of defense, not the primary fix: ragpoc.clr_host ships a host config that
    makes the CLR load those assemblies whatever zone they carry, which works even where this
    sweep cannot (a read-only or network install folder, a file re-tagged after startup). This
    stays because it is nearly free once it has run, and it also clears the tag off everything
    else in the folder rather than just unblocking the one DLL the CLR happens to need.

    Only the initial manual "download from GitHub + extract with Explorer" install is ever
    tagged this way -- the in-app updater writes fresh files via zipfile.extractall(), which
    never sets this stream -- so the probe below short-circuits on every later launch."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    try:
        import pythonnet
    except ImportError:
        return  # no CLR-backed webview in this build, so no zone tag can break the window
    # Probe the DLL whose zone tag actually breaks the window, *not* the exe: Windows deletes
    # the exe's own Zone.Identifier as soon as the user clicks through SmartScreen's "Run
    # anyway" on first launch, so an exe-based probe reports "nothing to do" on precisely the
    # runs where the rest of the folder is still tagged -- which is how this went unnoticed.
    probe = Path(pythonnet.__file__).resolve().parent / "runtime" / "Python.Runtime.dll"
    try:
        with open(f"{probe}:Zone.Identifier", "rb"):
            pass
    except OSError:
        return  # not blocked -- nothing to do
    try:
        for path in Path(sys.executable).resolve().parent.rglob("*"):
            if not path.is_file():
                continue
            try:
                os.remove(f"{path}:Zone.Identifier")
            except OSError:
                pass
    except OSError:
        pass
