# Ustawienie katalogu roboczego na główny katalog projektu (rodzic folderu tools)
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

Write-Host "Czyszczenie zasobów Kubernetes dla Symulacji..." -ForegroundColor Yellow

kubectl delete -f k8s/overlays/local-simulation/client-3.yaml --ignore-not-found
kubectl delete -f k8s/overlays/local-simulation/client-2.yaml --ignore-not-found
kubectl delete -f k8s/overlays/local-simulation/client-1.yaml --ignore-not-found
kubectl delete -f k8s/base/server-service.yaml --ignore-not-found
kubectl delete -f k8s/base/configmap.yaml --ignore-not-found

Write-Host "Czyszczenie zakończone!" -ForegroundColor Green
