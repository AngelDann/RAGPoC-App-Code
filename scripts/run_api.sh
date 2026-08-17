#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec .venv/bin/uvicorn ragpoc.api:create_app --factory --host 127.0.0.1 --port 8000
