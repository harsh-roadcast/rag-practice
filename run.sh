#!/bin/bash

# --- Color Definitions for prettier output ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- 1. Cleanup Function ---
# This runs when you press Ctrl+C
cleanup() {
    echo -e "\n${RED}🛑 Shutting down services...${NC}"
    
    # Kill all background jobs (Celery & Uvicorn) started by this script
    kill $(jobs -p) 2>/dev/null
    
    echo -e "${GREEN}✅ All python processes stopped.${NC}"
    echo -e "${BLUE}ℹ️  Note: Docker containers are still running. Run 'docker-compose down' to stop them.${NC}"
}

# Register the cleanup function to run on SIGINT (Ctrl+C)
trap cleanup SIGINT

# --- 2. Start Docker Infrastructure ---
echo -e "${BLUE}🐳 Starting Docker containers (Elasticsearch & Redis)...${NC}"
sudo docker-compose up -d

# Wait a moment for ports to open (prevents connection errors on startup)
echo -e "${BLUE}⏳ Waiting 5 seconds for services to initialize...${NC}"
sleep 5

# --- 3. Start Celery Worker (Background) ---
echo -e "${GREEN}👷 Starting Celery Worker...${NC}"
poetry run celery -A app.services.ingest.celery_app worker --loglevel=info &
CELERY_PID=$!

# --- 4. Start FastAPI Server (Background) ---
echo -e "${GREEN}🌐 Starting FastAPI Server...${NC}"
poetry run uvicorn app.main:app --reload &
UVICORN_PID=$!

# --- 5. Keep Alive ---
echo -e "${GREEN}🚀 RAG Stack is running!${NC}"
echo -e "   - API: http://localhost:8000/docs"
echo -e "   - ES:  http://localhost:9200"
echo -e "   - Press ${RED}Ctrl+C${NC} to stop."

# Wait for both processes. If one crashes, the script exits.
wait $CELERY_PID $UVICORN_PID