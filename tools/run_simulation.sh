#!/bin/bash
set -e

# Ustawienie katalogu roboczego na główny katalog projektu (rodzic folderu tools)
cd "$(dirname "$0")/.."

# Kolory do wypisywania tekstu
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN} Federated Learning Simulator K8s Deployer    ${NC}"
echo -e "${CYAN}==============================================${NC}"

# 1. Przygotowanie danych
echo -e "\n${YELLOW}[1/4] Przygotowanie podziału danych...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Błąd: Python nie jest zainstalowany lub nie ma go w zmiennej PATH!"
    exit 1
fi

echo "Uruchamianie data_scrapper.py..."
$PYTHON_CMD data/data_scrapper.py
echo "Uruchamianie data_splitter.py..."
$PYTHON_CMD data/data_splitter.py

# 2. Budowanie obrazów Docker
echo -e "\n${YELLOW}[2/4] Budowanie obrazów Docker...${NC}"
echo "Budowanie federated-sim-server:latest..."
docker build -t federated-sim-server:latest ./apps/server
echo "Budowanie federated-sim-client:latest..."
docker build -t federated-sim-client:latest ./apps/client

# Uwaga dla Minikube/kind
echo -e "\n* Uwaga: Jeśli używasz Minikube, upewnij się, że wywołałeś: 'eval \$(minikube docker-env)' przed tym skryptem."
echo -e "* Uwaga: Jeśli używasz kind, załaduj obrazy: 'kind load docker-image ...'"

# 3. Wdrożenie manifestów Kubernetes
echo -e "\n${YELLOW}[3/4] Wdrażanie do Kubernetes...${NC}"

# Pobranie bezwzględnej ścieżki projektu
PROJECT_ROOT="$(pwd)"

kubectl apply -f k8s/base/configmap.yaml

# Dynamiczne wstrzyknięcie bezwzględnej ścieżki za pomocą sed
sed "s|<PROJECT_ROOT>|${PROJECT_ROOT}|g" k8s/base/server-service.yaml | kubectl apply -f -
sed "s|<PROJECT_ROOT>|${PROJECT_ROOT}|g" k8s/overlays/local-simulation/client-1.yaml | kubectl apply -f -
sed "s|<PROJECT_ROOT>|${PROJECT_ROOT}|g" k8s/overlays/local-simulation/client-2.yaml | kubectl apply -f -
sed "s|<PROJECT_ROOT>|${PROJECT_ROOT}|g" k8s/overlays/local-simulation/client-3.yaml | kubectl apply -f -

# 4. Sprawdzenie statusu
echo -e "\n${YELLOW}[4/4] Sprawdzanie statusu wdrożenia...${NC}"
sleep 3
kubectl get deployments
kubectl get pods

echo -e "\n${GREEN}Symulacja wdrożona pomyślnie!${NC}"
echo -e "Aby śledzić logi serwera:    ${CYAN}kubectl logs -f -l app=federated-sim-server${NC}"
echo -e "Aby śledzić logi klienta 1:  ${CYAN}kubectl logs -f -l app=federated-sim-client-1${NC}"
