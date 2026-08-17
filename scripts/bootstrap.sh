#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt -e .
mkdir -p data/uploads data/renders data/derived/video_frames
: > data/.gitkeep
: > data/uploads/.gitkeep
: > data/renders/.gitkeep
: > data/derived/.gitkeep
: > data/derived/video_frames/.gitkeep

.venv/bin/python -m pytest -q
printf '\nBootstrap completed. Copy .env.example to .env and add OPENROUTER_API_KEY for live embeddings.\n'
