#!/bin/bash

# Ustawienie katalogu roboczego na główny katalog projektu
cd "$(dirname "$0")/.."

echo "Zabijanie (force kill) zasobów Kubernetes dla symulacji uczenia federacyjnego..."

# Wyłączenie oczekiwania (force delete z grace-period=0)
kubectl delete deployment federated-sim-server federated-sim-client-1 federated-sim-client-2 federated-sim-client-3 --ignore-not-found --grace-period=0 --force
kubectl delete service federated-sim-server --ignore-not-found --grace-period=0 --force
kubectl delete configmap federated-sim-config --ignore-not-found --grace-period=0 --force

# Usunięcie wszelkich wiszących lub utkniętych podów po etykiecie app
kubectl delete pods -l "app=federated-sim-server" --ignore-not-found --grace-period=0 --force
kubectl delete pods -l "app=federated-sim-client-1" --ignore-not-found --grace-period=0 --force
kubectl delete pods -l "app=federated-sim-client-2" --ignore-not-found --grace-period=0 --force
kubectl delete pods -l "app=federated-sim-client-3" --ignore-not-found --grace-period=0 --force

echo "Wszystkie powiązane kontenery i zasoby K8s zostały natychmiastowo zabite!"
