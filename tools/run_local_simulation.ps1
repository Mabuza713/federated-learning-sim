# Set location to the parent of the script directory (project root)
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$PROJECT_ROOT = (Get-Item .).FullName

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Launching local NumPy Federated Simulation" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 1. Start Server
Write-Host "Starting central server (MIN_CLIENTS=3, NUM_ROUNDS=5)..." -ForegroundColor Yellow
Start-Process cmd.exe -ArgumentList "/c `"title Federated Server&&set MIN_CLIENTS=3&&set NUM_ROUNDS=5&&.\.venv\Scripts\python.exe apps/server/src/main.py`""
Start-Sleep -Seconds 2

# 2. Start Clients
Write-Host "Starting 3 client nodes (Hospitals 1, 2, and 3)..." -ForegroundColor Yellow

$H1_DATA = Join-Path $PROJECT_ROOT "data\hospitals\hospital_1"
$H2_DATA = Join-Path $PROJECT_ROOT "data\hospitals\hospital_2"
$H3_DATA = Join-Path $PROJECT_ROOT "data\hospitals\hospital_3"

Start-Process cmd.exe -ArgumentList "/c `"title Hospital 1&&set SERVER_URL=http://localhost:8080&&set HOSPITAL_NAME=hospital_1&&set DATA_PATH=$H1_DATA&&.\.venv\Scripts\python.exe apps/client/src/main.py`""
Start-Process cmd.exe -ArgumentList "/c `"title Hospital 2&&set SERVER_URL=http://localhost:8080&&set HOSPITAL_NAME=hospital_2&&set DATA_PATH=$H2_DATA&&.\.venv\Scripts\python.exe apps/client/src/main.py`""
Start-Process cmd.exe -ArgumentList "/c `"title Hospital 3&&set SERVER_URL=http://localhost:8080&&set HOSPITAL_NAME=hospital_3&&set DATA_PATH=$H3_DATA&&.\.venv\Scripts\python.exe apps/client/src/main.py`""

Write-Host "`nSimulation launched successfully!" -ForegroundColor Green
Write-Host "Server and clients are running in separate command windows." -ForegroundColor Green
Write-Host "Simply close the opened terminal windows to stop the simulation." -ForegroundColor Cyan
