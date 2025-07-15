#!/bin/bash
# This script starts the necessary services for the application.

set -e

DB_PATH="/app/chroma_data"
DB_HOST="0.0.0.0" # Listen on all interfaces within the container
DB_PORT="8000"

# 1. Start the ChromaDB server in the background.
# It will use the data from the mounted volume at /app/chroma_data if provided.
echo "Starting ChromaDB server..."
chroma run --path "$DB_PATH" --host "$DB_HOST" --port "$DB_PORT" &

# Wait a moment for the server to initialize.
sleep 5

# 2. Launch the Streamlit application.
# It will be available on the port exposed by the Docker container.
echo "Launching Streamlit application..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.runOnSave=false 