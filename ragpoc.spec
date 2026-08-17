# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

block_cipher = None

ROOT = Path.cwd()

# collect_submodules() below imports our own packages to enumerate their
# submodules; they live under src/, so it has to be on sys.path before
# those calls run (pathex on Analysis() only takes effect later).
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

datas = [
    (str(ROOT / 'src/ragpoc/templates'), 'ragpoc/templates'),
    (str(ROOT / '.env.example'), '.'),
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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RAGPoC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
