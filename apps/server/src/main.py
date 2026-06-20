import os
import random
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from aggregation import federated_averaging, serialize_weights, deserialize_weights, calculate_weight_divergence

# Hyperparameters / settings
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "3"))
TOTAL_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
VAL_DATA_PATH = os.getenv("VAL_DATA_PATH", "./data/hospitals/server_val")

# Adaptive mode settings
DIVERGENCE_THRESHOLD = float(os.getenv("DIVERGENCE_THRESHOLD", "0.1"))
DEFAULT_LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "3"))

app = FastAPI(title="Federated Learning Server")

class SimpleCNN:
    """
    Model container matching the weights structure of the client model.
    """
    def __init__(self):
        # Match client's NumPy model shape (Input: 12288, Hidden: 32, Output: 2)
        # Randomly choose an initialization method for the weights at startup
        init_method = np.random.choice(["he_normal", "xavier_normal", "random_normal"])
        
        if init_method == "he_normal":
            std1 = np.sqrt(2.0 / 12288)
            std2 = np.sqrt(2.0 / 32)
        elif init_method == "xavier_normal":
            std1 = np.sqrt(2.0 / (12288 + 32))
            std2 = np.sqrt(2.0 / (32 + 2))
        else: # random_normal
            std1 = 0.01
            std2 = 0.01
            
        self.W1 = np.random.randn(12288, 32).astype(np.float32) * std1
        self.W2 = np.random.randn(32, 2).astype(np.float32) * std2
        
        # Also randomize biases slightly instead of always setting to zero
        self.b1 = (np.random.randn(32) * 0.01).astype(np.float32)
        self.b2 = (np.random.randn(2) * 0.01).astype(np.float32)
        
        print(f"[Model Initialization] Initialized weights using method: {init_method} (std1: {std1:.6f}, std2: {std2:.6f})")

    def forward(self, X: np.ndarray) -> np.ndarray:
        B = X.shape[0]
        # Flatten input: (B, 12288)
        X_flat = X.reshape(B, -1)
        Z1 = np.dot(X_flat, self.W1) + self.b1
        A1 = np.maximum(0, Z1) # ReLU
        Z2 = np.dot(A1, self.W2) + self.b2
        return Z2

    def state_dict(self) -> dict:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.W1 = state_dict["W1"].copy()
        self.b1 = state_dict["b1"].copy()
        self.W2 = state_dict["W2"].copy()
        self.b2 = state_dict["b2"].copy()

# Simple NumPy Dataset & DataLoader for Server Validation
class NumPyDataset:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_path, cls_name)
            for f in os.listdir(cls_dir):
                f_path = os.path.join(cls_dir, f)
                if os.path.isfile(f_path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((f_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

class NumPyDataLoader:
    def __init__(self, dataset: NumPyDataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_inputs = []
            batch_labels = []

            for idx in batch_indices:
                img_path, label = self.dataset.samples[idx]
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        img = img.resize((64, 64))
                        img_arr = np.array(img, dtype=np.float32) / 255.0
                        img_arr = (img_arr - 0.5) / 0.5
                        img_arr = np.transpose(img_arr, (2, 0, 1))
                        batch_inputs.append(img_arr)
                        batch_labels.append(label)
                except Exception:
                    continue

            if len(batch_inputs) > 0:
                yield np.array(batch_inputs, dtype=np.float32), np.array(batch_labels, dtype=np.int64)

def evaluate_model(model: SimpleCNN, dataloader: NumPyDataLoader) -> tuple:
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        logits = model.forward(inputs)
        
        # Softmax & Cross-Entropy Loss
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        B = logits.shape[0]
        loss = -np.mean(np.log(probs[np.arange(B), labels] + 1e-15))
        total_loss += loss * B

        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == labels)
        total += B

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy

# Server State
current_round = 0
global_model = SimpleCNN()
client_updates = {} # maps client_id -> state_dict dict
validation_history = [] # list of dicts: {"round": int, "loss": float, "accuracy": float}
val_dataloader = None

# Adaptive mode state variables
next_mode = "FedAvg"
next_epochs = DEFAULT_LOCAL_EPOCHS
current_divergence = 0.0
current_mode = "FedAvg"
current_epochs = DEFAULT_LOCAL_EPOCHS

def init_val_dataloader():
    global val_dataloader
    if os.path.exists(VAL_DATA_PATH):
        try:
            dataset = NumPyDataset(VAL_DATA_PATH)
            if len(dataset) > 0:
                val_dataloader = NumPyDataLoader(dataset, batch_size=32, shuffle=False)
                print(f"[Server Initialization] Loaded validation dataset from {VAL_DATA_PATH} with {len(dataset)} images.")
            else:
                print(f"[Server Initialization] Validation directory {VAL_DATA_PATH} is empty. Validation disabled.")
        except Exception as e:
            print(f"[Server Initialization] Error loading validation dataset: {e}. Validation disabled.")
    else:
        print(f"[Server Initialization] Validation directory {VAL_DATA_PATH} not found. Validation disabled.")

def run_validation():
    global current_round, global_model, validation_history, val_dataloader, current_divergence, current_mode, current_epochs
    if val_dataloader is not None:
        loss, acc = evaluate_model(global_model, val_dataloader)
        validation_history.append({
            "round": current_round,
            "loss": float(loss),
            "accuracy": float(acc),
            "divergence": float(current_divergence),
            "mode": current_mode,
            "epochs": current_epochs
        })
        print(f"[Server Validation] Round {current_round} (mode: {current_mode}, epochs: {current_epochs}, divergence: {current_divergence:.4f}) - Loss: {loss:.4f} - Accuracy: {acc:.2f}%")

class SubmitRequest(BaseModel):
    client_id: str
    round: int
    weights: str

@app.get("/status")
def get_status():
    return {
        "current_round": current_round,
        "total_rounds": TOTAL_ROUNDS,
        "min_clients": MIN_CLIENTS,
        "submitted_clients": list(client_updates.keys()),
        "submission_count": len(client_updates),
        "validation_history": validation_history,
        "next_mode": next_mode,
        "next_epochs": next_epochs,
        "divergence_threshold": DIVERGENCE_THRESHOLD
    }

@app.get("/model")
def get_model():
    weights_b64 = serialize_weights(global_model.state_dict())
    return {
        "round": current_round,
        "weights": weights_b64,
        "mode": next_mode,
        "epochs": next_epochs
    }

@app.post("/submit")
def submit_weights(req: SubmitRequest):
    global current_round, global_model, client_updates, next_mode, next_epochs, current_mode, current_epochs, current_divergence

    if req.round != current_round:
        raise HTTPException(
            status_code=400,
            detail=f"Submit round {req.round} does not match server current round {current_round}"
        )

    if req.client_id in client_updates:
        return {"status": "ignored", "detail": "Already submitted for this round"}

    try:
        weights = deserialize_weights(req.weights)
        client_updates[req.client_id] = weights
        print(f"Received update from client {req.client_id} for round {current_round}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to deserialize weights: {str(e)}")

    if len(client_updates) >= MIN_CLIENTS:
        print(f"Collected updates from {len(client_updates)} clients.")
        
        # Calculate divergence
        try:
            current_divergence = calculate_weight_divergence(list(client_updates.values()))
            print(f"[Adaptive Selection] Calculated weight divergence for round {current_round}: {current_divergence:.4f}")
        except Exception as e:
            current_divergence = 0.0
            print(f"[Adaptive Selection] Error calculating divergence: {e}")

        # Update current round's run parameters (the mode/epochs that produced this state)
        current_mode = next_mode
        current_epochs = next_epochs

        # Make decision for NEXT round
        if current_divergence > DIVERGENCE_THRESHOLD:
            next_mode = "FedSGD"
            next_epochs = 1
            print(f"[Adaptive Selection] Divergence ({current_divergence:.4f}) > Threshold ({DIVERGENCE_THRESHOLD:.4f}). Next round mode: FedSGD (1 local epoch).")
        else:
            next_mode = "FedAvg"
            next_epochs = DEFAULT_LOCAL_EPOCHS
            print(f"[Adaptive Selection] Divergence ({current_divergence:.4f}) <= Threshold ({DIVERGENCE_THRESHOLD:.4f}). Next round mode: FedAvg ({next_epochs} local epochs).")

        # Perform aggregation
        print("Performing Federated Averaging...")
        averaged_weights = federated_averaging(list(client_updates.values()))
        if averaged_weights is not None:
            global_model.load_state_dict(averaged_weights)
            current_round += 1
            client_updates = {}
            print(f"Round updated to {current_round}")
            run_validation()
        else:
            print("Error during federated averaging")

    return {"status": "success", "current_round": current_round}

# Initialize data and perform validation round 0 before server starts
@app.on_event("startup")
def startup_event():
    init_val_dataloader()
    run_validation()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

