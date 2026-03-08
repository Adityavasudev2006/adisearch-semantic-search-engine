#!/bin/bash

echo "Starting FastAPI backend..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Flask frontend..."
python web/app.py &

wait