#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -m ragpoc.mcp_server

# This MCP process uses stdio only. It does not bind a network port.
