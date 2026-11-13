#!/bin/bash
echo "🌿 Starting MangroveNet FastAPI Web Interface..."
source .venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
