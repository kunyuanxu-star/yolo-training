#!/usr/bin/env bash
set -e

# YOLO Training Platform — local mode startup script
# Usage: bash start-local.sh
# Stop:  Ctrl+C

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- prerequisite checks ----
check_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo -e "${RED}Error: '$1' not found. Please install it first.${NC}"
        exit 1
    }
}

check_cmd python3
check_cmd node
check_cmd npm

# Ensure .env exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo -e "${YELLOW}Creating .env from .env.example ...${NC}"
        cp .env.example .env
    else
        echo -e "${RED}Error: .env not found and no .env.example to copy.${NC}"
        exit 1
    fi
fi

# Ensure storage directory exists
mkdir -p storage

# ---- Python deps check ----
if ! python3 -c "import fastapi, uvicorn, sqlalchemy" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies ...${NC}"
    pip install -r requirements.txt
fi

# ---- Frontend deps check ----
if [ ! -d frontend/node_modules ]; then
    echo -e "${YELLOW}Installing frontend dependencies ...${NC}"
    (cd frontend && npm install)
fi

# ---- graceful shutdown ----
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down ...${NC}"
    kill 0 2>/dev/null
    wait 2>/dev/null
    echo -e "${GREEN}Stopped.${NC}"
}
trap cleanup SIGINT SIGTERM EXIT

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ---- start backend ----
echo -e "${GREEN}Starting backend on http://localhost:8000 ...${NC}"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | while IFS= read -r line; do
    echo -e "${CYAN}[backend]${NC} $line"
done &
BACKEND_PID=$!

sleep 2

# ---- start frontend ----
echo -e "${GREEN}Starting frontend on http://localhost:3000 ...${NC}"
(cd frontend && npm run dev 2>&1 | while IFS= read -r line; do
    echo -e "${CYAN}[frontend]${NC} $line"
done) &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Web UI:   http://localhost:3000${NC}"
echo -e "${GREEN}  API Docs: http://localhost:8000/docs${NC}"
echo -e "${GREEN}  Health:   http://localhost:8000/health${NC}"
echo -e "${GREEN}  Press Ctrl+C to stop${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

wait
