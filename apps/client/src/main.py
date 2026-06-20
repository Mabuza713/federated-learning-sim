import os
import time
import requests
from data_loader import get_dataloader
from train import SimpleCNN, CrossEntropyLoss, serialize_weights, deserialize_weights, train_local

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8080")
HOSPITAL_NAME = os.getenv("HOSPITAL_NAME", "hospital_1")
DATA_PATH = os.getenv("DATA_PATH", "/data")
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
DP_NOISE_SCALE = float(os.getenv("DP_NOISE_SCALE", "0.0"))

print(f"Initializing client: {HOSPITAL_NAME}")
print(f"Server URL: {SERVER_URL}")
print(f"Data Path: {DATA_PATH}")
print(f"Differential Privacy Noise Scale: {DP_NOISE_SCALE}")

model = SimpleCNN()
criterion = CrossEntropyLoss()

def main():
    local_round = 0
    dataloader = get_dataloader(DATA_PATH, BATCH_SIZE, HOSPITAL_NAME)
    
    while True:
        try:
            # Check server status first to see if simulation finished
            status_resp = requests.get(f"{SERVER_URL}/status")
            if status_resp.status_code == 200:
                status = status_resp.json()
                if local_round >= status["total_rounds"]:
                    print(f"[{HOSPITAL_NAME}] Reached total rounds ({status['total_rounds']}). Training completed. Sleeping...")
                    time.sleep(60)
                    continue
            
            # Fetch global model weights
            resp = requests.get(f"{SERVER_URL}/model")
            if resp.status_code != 200:
                print(f"[{HOSPITAL_NAME}] Failed to fetch model from server. Retrying in 5s...")
                time.sleep(5)
                continue
                
            data = resp.json()
            server_round = data["round"]
            
            if server_round >= local_round:
                print(f"\n[{HOSPITAL_NAME}] Syncing with round {server_round} (Local round is {local_round}).")
                # Synchronize weights
                deserialize_weights(model, data["weights"])
                local_round = server_round
                
                # Read training mode parameters from server
                epochs_to_run = data.get("epochs", LOCAL_EPOCHS)
                mode_to_run = data.get("mode", "FedAvg")
                print(f"[{HOSPITAL_NAME}] Mode: {mode_to_run}, Epochs to run: {epochs_to_run}")
                
                # Perform local training
                train_local(model, dataloader, criterion, epochs_to_run, LEARNING_RATE, HOSPITAL_NAME, DP_NOISE_SCALE)
                
                # Submit weights
                print(f"[{HOSPITAL_NAME}] Uploading weights for round {local_round} (sample size: {len(dataloader.dataset)})...")
                payload = {
                    "client_id": HOSPITAL_NAME,
                    "round": local_round,
                    "weights": serialize_weights(model.state_dict()),
                    "sample_size": len(dataloader.dataset)
                }
                
                submit_resp = requests.post(f"{SERVER_URL}/submit", json=payload)
                if submit_resp.status_code == 200:
                    res = submit_resp.json()
                    print(f"[{HOSPITAL_NAME}] Submitted weights successfully! Server response: {res}")
                    local_round += 1
                else:
                    print(f"[{HOSPITAL_NAME}] Submission failed: {submit_resp.text}. Retrying in 5s...")
                    time.sleep(5)
            else:
                # server_round < local_round: server is still aggregating the current round, wait
                print(f"[{HOSPITAL_NAME}] Waiting for server to update round (Local: {local_round}, Server: {server_round})...")
                time.sleep(5)
                
        except requests.exceptions.RequestException as e:
            print(f"[{HOSPITAL_NAME}] Server communication error: {e}. Retrying in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    main()
