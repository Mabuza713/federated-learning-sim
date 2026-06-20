#!/bin/bash

echo "Zabijanie lokalnych procesów Python dla symulacji (WSL/Linux)..."

# Zabicie procesów serwera i klientów na podstawie nazwy pliku
pkill -f "apps/server/src/main.py"
pkill -f "apps/client/src/main.py"
pkill -f "main.py"

echo "Procesy lokalne zostały zabite!"
