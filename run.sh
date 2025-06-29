#!/bin/bash
#
# This script starts the production-ready Uvicorn API service.
# It assumes that the Triton Docker container is already running separately.

echo "--- Starting Jina V3 Embedder Uvicorn Service ---"

# The number of worker processes to spawn.
# A good starting point is (2 * number_of_cpu_cores) + 1.
# We are using 4 as a sensible default. Adjust as needed for your machine.
WORKERS=4

# The port you specified for the application.
PORT=24434

echo "-> Starting Uvicorn with $WORKERS workers on port $PORT..."

# Start the FastAPI application using Uvicorn.
# --workers: Spawns multiple processes to handle more requests concurrently.
# --port: Sets the custom port.
# The systemd service file will ensure 'uvicorn' is in the PATH.
uvicorn service:app --host 0.0.0.0 --port $PORT --workers $WORKERS