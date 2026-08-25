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
