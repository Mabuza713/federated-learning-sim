import io
import base64
import pickle
import numpy as np

class SimpleCNN:
    """
    Simplified MLP model replacing PyTorch's SimpleCNN.
    We keep the class name for compatibility with client and server main scripts.
    """
    def __init__(self):
        # Input: flattened image of shape (B, 3 * 64 * 64) -> (B, 12288)
        # Hidden: 32 neurons
        # Output: 2 neurons (binary classification)
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
        
        # Momentum velocity buffers
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X shape: (B, 3, 64, 64)
        B = X.shape[0]
        # Flatten input: (B, 12288)
        X_flat = X.reshape(B, -1)
        
        self.last_X = X_flat
        self.Z1 = np.dot(X_flat, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1) # ReLU activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

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


class CrossEntropyLoss:
    """
    NumPy implementation of CrossEntropyLoss.
    """
    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        # Numerically stable softmax
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        B = logits.shape[0]
        loss = -np.mean(np.log(self.probs[np.arange(B), targets] + 1e-15))
        return loss

    def backward(self, targets: np.ndarray) -> np.ndarray:
        # Gradient of loss w.r.t logits (dZ2)
        B = self.probs.shape[0]
        dZ2 = self.probs.copy()
        dZ2[np.arange(B), targets] -= 1.0
        dZ2 = dZ2 / B # Average over batch
        return dZ2


def serialize_weights(model_state: dict) -> str:
    """
    Serializes a dict of NumPy weights to a base64 encoded string.
    """
    buffer = io.BytesIO()
    pickle.dump(model_state, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def deserialize_weights(model: SimpleCNN, weights_str: str) -> None:
    """
    Deserializes a base64 encoded string into a SimpleCNN model.
    """
    buffer = io.BytesIO(base64.b64decode(weights_str.encode("utf-8")))
    buffer.seek(0)
    state_dict = pickle.load(buffer)
    model.load_state_dict(state_dict)


def train_local(
    model: SimpleCNN,
    dataloader,
    criterion: CrossEntropyLoss,
    local_epochs: int,
    learning_rate: float,
    hospital_name: str,
    dp_noise_scale: float = 0.0
) -> None:
    """
    Trains the model locally using the given dataloader in NumPy.
    """
    model.train()
    momentum = 0.9
    
    print(f"[{hospital_name}] Starting local training for {local_epochs} epochs (NumPy)...")
    if dp_noise_scale > 0.0:
        print(f"[{hospital_name}] Differential Privacy enabled. Noise scale: {dp_noise_scale}")
        
    for epoch in range(local_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            dZ2 = criterion.backward(labels)
            
            # Gradients for layer 2
            dW2 = np.dot(model.A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0)
            
            # Gradients for layer 1 (ReLU backprop)
            dA1 = np.dot(dZ2, model.W2.T)
            dZ1 = dA1 * (model.Z1 > 0)
            
            dW1 = np.dot(model.last_X.T, dZ1)
            db1 = np.sum(dZ1, axis=0)
            
            # Add Gaussian noise for Differential Privacy (DP)
            if dp_noise_scale > 0.0:
                dW2 += np.random.normal(0.0, dp_noise_scale, size=dW2.shape)
                db2 += np.random.normal(0.0, dp_noise_scale, size=db2.shape)
                dW1 += np.random.normal(0.0, dp_noise_scale, size=dW1.shape)
                db1 += np.random.normal(0.0, dp_noise_scale, size=db1.shape)
            
            # Update weights using SGD with momentum
            model.v_W2 = momentum * model.v_W2 + dW2
            model.W2 -= learning_rate * model.v_W2
            
            model.v_b2 = momentum * model.v_b2 + db2
            model.b2 -= learning_rate * model.v_b2
            
            model.v_W1 = momentum * model.v_W1 + dW1
            model.W1 -= learning_rate * model.v_W1
            
            model.v_b1 = momentum * model.v_b1 + db1
            model.b1 -= learning_rate * model.v_b1
            
            running_loss += loss * inputs.shape[0]
            predicted = np.argmax(outputs, axis=1)
            total += labels.shape[0]
            correct += np.sum(predicted == labels)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"[{hospital_name}] Epoch {epoch+1}/{local_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
