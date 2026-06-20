#!/bin/bash

# Navigate to the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo " Launching local NumPy Federated Simulation"
echo "=============================================="

# Detect if running under WSL/Linux
IS_LINUX=false
if [[ "$(uname)" == "Linux" ]]; then
    IS_LINUX=true
fi

# Detect Python executable
if [ "$IS_LINUX" = true ]; then
    if [ -f "./.venv/bin/python" ]; then
        PYTHON_EXEC="./.venv/bin/python"
    elif command -v python3 &>/dev/null; then
        PYTHON_EXEC="python3"
    else
        PYTHON_EXEC="python"
    fi
else
    # Windows-like environments (Git Bash, etc.)
    if [ -f "./.venv/Scripts/python.exe" ]; then
        PYTHON_EXEC="./.venv/Scripts/python.exe"
    else
        PYTHON_EXEC="python"
    fi
fi

echo "Using Python executable: $PYTHON_EXEC"

# 1. Start Server in background
echo "Starting central server (MIN_CLIENTS=3, NUM_ROUNDS=5)..."
export MIN_CLIENTS=3
export NUM_ROUNDS=5
$PYTHON_EXEC apps/server/src/main.py &
SERVER_PID=$!

sleep 2

# 2. Start Clients in background
echo "Starting 3 client nodes (Hospitals 1, 2, and 3)..."
export SERVER_URL="http://localhost:8080"

export HOSPITAL_NAME="hospital_1"
export DATA_PATH="./data/hospitals/hospital_1"
$PYTHON_EXEC apps/client/src/main.py &
CLIENT1_PID=$!

export HOSPITAL_NAME="hospital_2"
export DATA_PATH="./data/hospitals/hospital_2"
$PYTHON_EXEC apps/client/src/main.py &
CLIENT2_PID=$!

export HOSPITAL_NAME="hospital_3"
export DATA_PATH="./data/hospitals/hospital_3"
$PYTHON_EXEC apps/client/src/main.py &
CLIENT3_PID=$!

echo ""
echo "Simulation running in background."
echo "PIDs: Server=$SERVER_PID, Hospital 1=$CLIENT1_PID, Hospital 2=$CLIENT2_PID, Hospital 3=$CLIENT3_PID"
echo ""
echo "To stop the simulation, run:"
echo "kill $SERVER_PID $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID"
