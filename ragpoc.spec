# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

block_cipher = None

ROOT = Path.cwd()
APP_NAME = 'RAGPoC'

# collect_submodules() below imports our own packages to enumerate their
# submodules; they live under src/, so it has to be on sys.path before
# those calls run (pathex on Analysis() only takes effect later).
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

datas = [
    (str(ROOT / 'src/ragpoc/templates'), 'ragpoc/templates'),
    (str(ROOT / '.env.example'), '.'),
    (str(ROOT / 'assets'), 'assets'),
]

# Include sqlite-vec extension binaries if present
import sqlite_vec
sqlite_vec_dir = Path(sqlite_vec.__file__).parent
datas.append((str(sqlite_vec_dir), 'sqlite_vec'))

# Some pydantic_ai dependencies read their own package version via
# importlib.metadata.version(...) at import time (not lazily), which raises
# PackageNotFoundError under PyInstaller because frozen builds don't ship
# .dist-info metadata by default — this breaks every request (any view that
# imports knowledge.views -> pydantic_ai). copy_metadata bundles just the
# dist-info so the lookup succeeds.
from PyInstaller.utils.hooks import copy_metadata
for _pkg in ('genai_prices', 'pydantic_ai_slim', 'mcp'):
    datas += copy_metadata(_pkg)

# Collect every submodule of our own packages: Django resolves middleware,
# apps, management commands, etc. by dotted string (settings.py, app
# registry), which PyInstaller's static import scan can't trace — an
# explicit hand-maintained list silently drops entries (e.g. it previously
# missed knowledge.middleware, breaking the compiled app at startup).
from PyInstaller.utils.hooks import collect_submodules
own_packages = (
    collect_submodules('knowledge')
    + collect_submodules('ragpoc')
    + collect_submodules('ragpoc_django')
)

hiddenimports = own_packages + [
    'django',
    'django.core.management',
    'django.core.management.commands.migrate',
    'django.core.management.commands.showmigrations',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'sqlite_vec',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'pydantic',
    'pydantic_ai',
    'pydantic_ai.models.openai',
    'openai',
    'ddgs',
    'truststore',
    'asgiref',
    'webview',
    'webview.platforms.winforms',
    'webview.platforms.edgechromium',
    'clr_loader',
]

a = Analysis(
    ['desktop_launcher.py'],
    pathex=[str(ROOT), str(ROOT / 'src')],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --onedir, not --onefile: a onefile build self-extracts to a fresh %TEMP%\_MEI<pid>
# directory and LoadLibrary()s python313.dll from there on *every* launch, not just
# after an update -- that extract-then-load step is an inherent race window (a
# transient lock/scan on the just-written files can make it fail with "Failed to
# load Python DLL ... module not found") that a onedir build doesn't have at all:
# the exe loads its DLLs straight from its own install folder, no extraction step,
# no _MEI, no per-launch race. exclude_binaries=True hands a.binaries/a.zipfiles/
# a.datas to COLLECT below instead of embedding them in the exe itself.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # upx=False deliberately. UPX rewrites the PE headers of everything it packs, which
    # corrupts managed .NET assemblies -- Python.Runtime.dll and Microsoft.Web.WebView2.Core.dll
    # among them -- so the CLR fails to load them and the desktop window degrades to a browser
    # tab, the same symptom (and same hard-to-read error) as the Mark-of-the-Web problem
    # ragpoc/clr_host.py fixes. Nothing packs today only because the CI runner has no upx on
    # PATH; that is luck, not a guarantee, and it also spares us the AV false positives packed
    # executables attract. Size cost is small next to a bundle this size.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / 'assets' / 'ragpoc.ico'),
)

# Produces dist/RAGPoC/RAGPoC.exe + dist/RAGPoC/_internal/... . release.yml zips the
# *contents* of dist/RAGPoC/ (not the folder itself) so it extracts as RAGPoC.exe +
# _internal\ directly inside the install folder -- BASE_DIR (src/ragpoc/config.py)
# is the exe's own parent directory, so this keeps data/, .env and ragpoc.log
# resolving exactly where existing installs already have them.
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,  # see the EXE() call above
    upx_exclude=[],
    name=APP_NAME,
)

# pywebview renders through WebView2, which it drives from managed code: importing it starts a
# .NET Framework CLR in-process via pythonnet. That CLR reads its host configuration from
# "<full path of the running exe>.config" at startup, and without the loadFromRemoteSources
# switch in that file it refuses to load pythonnet's own Python.Runtime.dll whenever the
# install folder carries Windows' Mark of the Web -- which every folder extracted from a
# downloaded zip does -- leaving the user with a browser tab instead of the desktop window.
# See ragpoc/clr_host.py for the full story.
#
# Written here rather than passed through datas because PyInstaller routes datas into
# _internal/, and .NET only ever looks for this file directly beside the executable. COLLECT
# has already populated dist/RAGPoC/ by the time the spec reaches this line, so both the
# release zip and the Inno Setup installer (which copy dist/RAGPoC/*) pick it up.
from ragpoc.clr_host import CLR_HOST_CONFIG_XML

(Path(DISTPATH) / APP_NAME / f'{APP_NAME}.exe.config').write_text(CLR_HOST_CONFIG_XML, encoding='utf-8')
