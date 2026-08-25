import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from django.test import RequestFactory

from knowledge.views import check_update_view
from ragpoc.updater import (
    check_for_update,
    get_current_platform,
    select_release_asset,
)


def test_get_current_platform_windows():
    with patch.object(sys, "platform", "win32"), patch.dict(os.environ, {}, clear=True):
        assert get_current_platform() == "windows"


def test_get_current_platform_android_env():
    with patch.dict(os.environ, {"ANDROID_ROOT": "/system"}):
        assert get_current_platform() == "android"


def test_get_current_platform_termux():
    with patch.dict(os.environ, {"TERMUX_VERSION": "0.118"}):
        assert get_current_platform() == "android"


def test_get_current_platform_linux():
    with patch.object(sys, "platform", "linux"), patch.dict(os.environ, {}, clear=True):
        assert get_current_platform() == "linux"


def test_get_current_platform_macos():
    with patch.object(sys, "platform", "darwin"), patch.dict(os.environ, {}, clear=True):
        assert get_current_platform() == "macos"


def test_select_release_asset_windows():
    assets = [
        {"name": "RAGPoC-Setup.exe", "browser_download_url": "https://example.com/setup.exe"},
        {"name": "RAGPoC-windows.zip", "browser_download_url": "https://example.com/win.zip"},
        {"name": "RAGPoC-android.apk", "browser_download_url": "https://example.com/app.apk"},
        {"name": "SHA256SUMS.txt", "browser_download_url": "https://example.com/sha.txt"},
    ]
    matched = select_release_asset(assets, "windows")
    assert matched is not None
    assert matched["name"] == "RAGPoC-windows.zip"


def test_select_release_asset_android():
    assets = [
        {"name": "RAGPoC-windows.zip", "browser_download_url": "https://example.com/win.zip"},
        {"name": "RAGPoC-v2.0.5-android.apk", "browser_download_url": "https://example.com/app.apk"},
        {"name": "SHA256SUMS.txt", "browser_download_url": "https://example.com/sha.txt"},
    ]
    matched = select_release_asset(assets, "android")
    assert matched is not None
    assert matched["name"] == "RAGPoC-v2.0.5-android.apk"


def test_select_release_asset_fallback():
    assets = [
        {"name": "SHA256SUMS.txt", "browser_download_url": "https://example.com/sha.txt"},
    ]
    matched = select_release_asset(assets, "android")
    assert matched is None


@pytest.mark.asyncio
async def test_check_for_update_with_platform_override():
    fake_release = {
        "tag_name": "v99.0.0",
        "body": "Test release notes",
        "html_url": "https://github.com/AngelDann/RAGPoC-App-Code/releases/tag/v99.0.0",
        "assets": [
            {"name": "RAGPoC-windows.zip", "browser_download_url": "https://example.com/win.zip"},
            {"name": "RAGPoC-android.apk", "browser_download_url": "https://example.com/android.apk"},
        ],
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: fake_release
    mock_response.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("ragpoc.updater.new_async_client", return_value=mock_client):
        # Check Android target
        info_android = await check_for_update(target_os="android")
        assert info_android["update_available"] is True
        assert info_android["platform"] == "android"
        assert info_android["asset_name"] == "RAGPoC-android.apk"
        assert info_android["download_url"] == "https://example.com/android.apk"

        # Check Windows target
        info_win = await check_for_update(target_os="windows")
        assert info_win["update_available"] is True
        assert info_win["platform"] == "windows"
        assert info_win["asset_name"] == "RAGPoC-windows.zip"


@pytest.mark.asyncio
async def test_check_update_view_with_query_param():
    fake_release = {
        "tag_name": "v99.0.0",
        "body": "Android release",
        "html_url": "https://github.com/AngelDann/RAGPoC-App-Code/releases/tag/v99.0.0",
        "assets": [
            {"name": "RAGPoC-android.apk", "browser_download_url": "https://example.com/android.apk"},
        ],
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: fake_release
    mock_response.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    factory = RequestFactory()
    request = factory.get("/api/update/check?platform=android")

    with patch("ragpoc.updater.new_async_client", return_value=mock_client):
        response = await check_update_view(request)
        assert response.status_code == 200
        import json
        data = json.loads(response.content)
        assert data["platform"] == "android"
        assert data["asset_name"] == "RAGPoC-android.apk"


@pytest.mark.asyncio
async def test_update_manager_background_download_and_staging(tmp_path, monkeypatch):
    import io
    import zipfile
    from ragpoc.updater import UpdateManager, UpdateState

    manager = UpdateManager()
    assert manager.state == UpdateState.IDLE

    # Mock platform as windows
    monkeypatch.setattr("ragpoc.updater.get_current_platform", lambda: "windows")

    # Create dummy zip payload
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("RAGPoC.exe", "fake binary")
        zf.writestr("_internal/marker.txt", "marker")
    zip_bytes = zip_buf.getvalue()

    class _FakeStreamResponse:
        headers = {"content-length": str(len(zip_bytes))}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        def raise_for_status(self):
            pass
        async def aiter_bytes(self):
            chunk_size = len(zip_bytes) // 2 or 1
            yield zip_bytes[:chunk_size]
            yield zip_bytes[chunk_size:]

    class _FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        def stream(self, *a, **kw):
            return _FakeStreamResponse()

    monkeypatch.setattr("ragpoc.updater.new_async_client", lambda *a, **kw: _FakeClient())

    status = await manager.start_download(
        download_url="https://github.com/AngelDann/RAGPoC-App-Code/releases/download/v9.0.0/RAGPoC-windows.zip",
        target_version="9.0.0",
    )
    assert status["state"] in (UpdateState.DOWNLOADING, UpdateState.READY_TO_INSTALL)

    # Wait for the background task to complete
    if manager._download_task:
        await manager._download_task

    assert manager.state == UpdateState.READY_TO_INSTALL
    assert manager.progress_percent == 100.0
    assert manager.staging_dir is not None
    assert manager.staging_dir.exists()
    assert (manager.staging_dir / "RAGPoC.exe").exists()
    assert (manager.staging_dir / "_internal" / "marker.txt").exists()

    # Clean up
    manager._cleanup_download_files()


@pytest.mark.asyncio
async def test_update_manager_cancel_download(monkeypatch):
    import asyncio
    from ragpoc.updater import UpdateManager, UpdateState

    manager = UpdateManager()
    monkeypatch.setattr("ragpoc.updater.get_current_platform", lambda: "windows")

    class _HangingStreamResponse:
        headers = {"content-length": "1000000"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        def raise_for_status(self):
            pass
        async def aiter_bytes(self):
            yield b"part1"
            await asyncio.sleep(10)
            yield b"part2"

    class _FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        def stream(self, *a, **kw):
            return _HangingStreamResponse()

    monkeypatch.setattr("ragpoc.updater.new_async_client", lambda *a, **kw: _FakeClient())

    await manager.start_download(
        download_url="https://github.com/AngelDann/RAGPoC-App-Code/releases/download/v9.0.0/RAGPoC-windows.zip",
        target_version="9.0.0",
    )
    assert manager.state == UpdateState.DOWNLOADING

    cancel_res = await manager.cancel_download()
    assert cancel_res["state"] == UpdateState.UPDATE_AVAILABLE
    assert manager.progress_percent == 0.0


@pytest.mark.asyncio
async def test_update_views_integration(monkeypatch):
    import json
    from knowledge.views import (
        cancel_download_update_view,
        start_download_update_view,
        update_status_view,
    )
    from ragpoc.updater import updater_manager

    factory = RequestFactory()

    # Test status endpoint
    req_status = factory.get("/api/updates/status")
    res_status = update_status_view(req_status)
    assert res_status.status_code == 200
    data_status = json.loads(res_status.content)
    assert "state" in data_status
    assert "ready_to_install" in data_status

    # Test start download endpoint validation
    req_dl_invalid = factory.post(
        "/api/updates/download",
        data=json.dumps({"download_url": "https://malicious.com/app.zip"}),
        content_type="application/json",
    )
    res_dl_invalid = await start_download_update_view(req_dl_invalid)
    assert res_dl_invalid.status_code == 400

    # Test cancel endpoint
    req_cancel = factory.post("/api/updates/cancel")
    res_cancel = await cancel_download_update_view(req_cancel)
    assert res_cancel.status_code == 200

