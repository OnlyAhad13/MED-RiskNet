#!/bin/bash
# Server startup script for MED-RiskNET API

set -e

echo "Starting MED-RiskNET API Server..."

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
else
    echo "Running locally"
    
    # Activate conda environment if available
    if command -v conda &> /dev/null; then
        echo "Activating conda environment..."
        conda activate ml_global || true
    fi
fi

# Set environment variables
export PYTHONPATH=$(pwd)
export PYTHONUNBUFFERED=1

# Default configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
RELOAD="${RELOAD:-false}"

# Build uvicorn command
UVICORN_CMD="uvicorn deployment.app:app --host $HOST --port $PORT --workers $WORKERS"

if [ "$RELOAD" = "true" ]; then
    UVICORN_CMD="$UVICORN_CMD --reload"
fi

# Start server
echo "Starting server on http://$HOST:$PORT"
echo "Workers: $WORKERS"
echo "Hot reload: $RELOAD"
echo ""

exec $UVICORN_CMD
