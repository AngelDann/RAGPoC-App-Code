"""Download every model the local agent needs, with progress and verification.

    python scripts/download_local_models.py              # all
    python scripts/download_local_models.py --model qwen3-vl-2b
    python scripts/download_local_models.py --list

Each download is verified against the model's required files before reporting
success, so a repo layout that no longer matches the manifest fails loudly
instead of looking like an empty install.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpoc.chat_models import CHAT_MODELS, ChatModelDownloader, list_chat_models  # noqa: E402
from ragpoc.config import Settings  # noqa: E402

MB = 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="key of a single model to download")
    parser.add_argument("--list", action="store_true", help="list available models and exit")
    parser.add_argument("--data-dir", default="", help="override the data directory")
    args = parser.parse_args()

    if args.list:
        for entry in list_chat_models():
            print(f"{entry['key']:<16} {entry['size_mb']:>8.1f} MB  {entry['modalities']}")
            print(f"{'':<16} {entry['description']}")
        return 0

    settings = Settings(_env_file=None, data_dir=args.data_dir or "data", allowed_upload_dir="data")
    keys = [args.model] if args.model else sorted(CHAT_MODELS)
    unknown = [k for k in keys if k not in CHAT_MODELS]
    if unknown:
        print(f"unknown model(s): {unknown}", file=sys.stderr)
        return 2

    failures = 0
    for key in keys:
        downloader = ChatModelDownloader(settings, key)
        print(f"\n=== {key} ({downloader.manifest.size_bytes / MB:.1f} MB) -> {downloader.install_dir}")
        status = downloader.download_blocking()
        print(f"    {status.state}: {status.message}")
        if status.state != "ready":
            failures += 1
            if status.error:
                print(f"    error: {status.error}", file=sys.stderr)
        else:
            print(f"    verified: {downloader.manifest.required}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
