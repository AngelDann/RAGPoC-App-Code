"""The .NET Framework host configuration that lets the desktop window survive a zip download.

pywebview's Windows backend renders through WebView2, which it drives from managed code:
`import webview` ends up at `import clr`, which makes pythonnet start a .NET Framework CLR
inside our process and load its own managed assembly, `pythonnet/runtime/Python.Runtime.dll`.

That load is the fragile step. Windows tags every file extracted from a downloaded zip with
the "Mark of the Web" (an NTFS Zone.Identifier=3 stream), and .NET Framework treats an
assembly carrying that tag as coming from a *remote source*: since .NET 4.0 it refuses to
load it unless the host opts in via the `loadFromRemoteSources` switch. clr_loader surfaces
that refusal as a bare `RuntimeError: Failed to resolve Python.Runtime.Loader.Initialize
from ...\\Python.Runtime.dll`, pywebview reports "no backend available", and the launcher's
fallback quietly opens a browser tab instead of the desktop window -- the exact symptom
users of the release zip hit, on a machine where nothing is actually missing.

The switch is a host-process setting, not something a library can flip at runtime, and the
CLR reads it from "<full path of the running exe>.config" when it starts. Shipping that file
next to RAGPoC.exe therefore fixes the whole failure class at its source: the CLR loads the
assembly regardless of what zone tags the install folder carries, so it no longer matters
whether the user installed via the installer, extracted the zip, or copied the folder off a
network share. See ragpoc.updater.unblock_downloaded_install for the second line of defense.
"""

from __future__ import annotations

import sys
from pathlib import Path

CLR_HOST_CONFIG_XML = """<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <startup>
    <supportedRuntime version="v4.0" sku=".NETFramework,Version=v4.0" />
  </startup>
  <runtime>
    <loadFromRemoteSources enabled="true" />
  </runtime>
</configuration>
"""


def config_path_for(exe: Path) -> Path:
    """The .NET Framework looks for the host's config at the executable's *full* path plus
    ".config" -- i.e. RAGPoC.exe.config, keeping the .exe, not RAGPoC.config."""
    return exe.with_name(f"{exe.name}.config")


def ensure_clr_host_config() -> None:
    """Writes the host config next to the running executable if it isn't already there.

    ragpoc.spec ships this file with every build, so on a healthy install this is a single
    stat() and nothing else. It exists for the installs that predate that build step: they
    self-update by extracting a release zip over themselves, and this runs early enough in
    startup (before anything imports webview, and therefore before the CLR starts and reads
    the file) that such an install gets its window back on the very first launch after
    updating rather than needing a second restart.

    Never raises. A read-only install folder means we cannot write the file, but the app
    still works -- it just falls back to the browser window path in desktop_launcher.
    """
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    try:
        config = config_path_for(Path(sys.executable).resolve())
        if not config.exists():
            config.write_text(CLR_HOST_CONFIG_XML, encoding="utf-8")
    except OSError:
        pass
