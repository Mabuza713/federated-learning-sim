# Ustawienie katalogu roboczego na główny katalog projektu (rodzic folderu tools)
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Federated Learning Simulator K8s Deployer" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 1. Przygotowanie danych
Write-Host "`n[1/4] Przygotowanie podziału danych..." -ForegroundColor Yellow
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "Uruchamianie data_scrapper.py..." -ForegroundColor Gray
    python data/data_scrapper.py
    Write-Host "Uruchamianie data_splitter.py..." -ForegroundColor Gray
    python data/data_splitter.py
} else {
    Write-Error "Python nie jest zainstalowany lub nie ma go w zmiennej środowiskowej PATH! Uruchom skrypty przygotowania danych ręcznie."
    Exit 1
}

# 2. Budowanie obrazów Docker
Write-Host "`n[2/4] Budowanie obrazów Docker..." -ForegroundColor Yellow
Write-Host "Budowanie federated-sim-server:latest..." -ForegroundColor Gray
docker build -t federated-sim-server:latest ./apps/server
Write-Host "Budowanie federated-sim-client:latest..." -ForegroundColor Gray
docker build -t federated-sim-client:latest ./apps/client

# Uwaga dla Minikube/kind
Write-Host "`n* Uwaga: Jeśli używasz Minikube, upewnij się, że przed uruchomieniem tego skryptu wywołałeś: 'minikube docker-env | Invoke-Expression'" -ForegroundColor DarkYellow
Write-Host "* Uwaga: Jeśli używasz kind, załaduj obrazy do klastra: 'kind load docker-image ...'" -ForegroundColor DarkYellow

# 3. Wdrożenie manifestów Kubernetes
Write-Host "`n[3/4] Wdrażanie do Kubernetes..." -ForegroundColor Yellow

$PROJECT_ROOT = (Get-Item .).FullName
# Zamiana pojedynczych ukośników wstecznych na podwójne ukośniki dla formatu YAML w systemie Windows
$ESCAPED_ROOT = $PROJECT_ROOT -replace '\\', '\\\\'

kubectl apply -f k8s/base/configmap.yaml

# Dynamiczne wstrzyknięcie bezwzględnej ścieżki
(Get-Content k8s/base/server-service.yaml) -replace '<PROJECT_ROOT>', $ESCAPED_ROOT | kubectl apply -f -
(Get-Content k8s/overlays/local-simulation/client-1.yaml) -replace '<PROJECT_ROOT>', $ESCAPED_ROOT | kubectl apply -f -
(Get-Content k8s/overlays/local-simulation/client-2.yaml) -replace '<PROJECT_ROOT>', $ESCAPED_ROOT | kubectl apply -f -
(Get-Content k8s/overlays/local-simulation/client-3.yaml) -replace '<PROJECT_ROOT>', $ESCAPED_ROOT | kubectl apply -f -

# 4. Sprawdzenie statusu
Write-Host "`n[4/4] Sprawdzanie statusu wdrożenia..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
kubectl get deployments
kubectl get pods

Write-Host "`nSymulacja wdrożona pomyślnie!" -ForegroundColor Green
Write-Host "Aby śledzić logi serwera:    kubectl logs -f -l app=federated-sim-server" -ForegroundColor Cyan
Write-Host "Aby śledzić logi klienta 1:  kubectl logs -f -l app=federated-sim-client-1" -ForegroundColor Cyan
