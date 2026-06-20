import io
import base64
import pickle
import numpy as np

def serialize_weights(model_state: dict) -> str:
    """
    Serializes a PyTorch-compatible/NumPy model state_dict to a base64 encoded string.
    """
    buffer = io.BytesIO()
    pickle.dump(model_state, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def deserialize_weights(weights_str: str) -> dict:
    """
    Deserializes a base64 encoded string into a model state_dict of NumPy arrays.
    """
    buffer = io.BytesIO(base64.b64decode(weights_str.encode("utf-8")))
    buffer.seek(0)
    return pickle.load(buffer)

def federated_averaging(weights_list: list) -> dict:
    """
    Performs federated averaging (FedAvg) over a list of client state_dicts (NumPy arrays).
    """
    if not weights_list:
        return None
    global_state_dict = {}
    keys = weights_list[0].keys()
    for key in keys:
        arrays = [w[key] for w in weights_list]
        global_state_dict[key] = np.mean(arrays, axis=0).astype(np.float32)
    return global_state_dict

def calculate_weight_divergence(weights_list: list) -> float:
    """
    Computes the average relative L2 distance (divergence) between the client weights
    and their mean (aggregated weights).
    """
    if not weights_list or len(weights_list) < 2:
        return 0.0
    
    # First, compute the mean weights (what the new global model will be)
    mean_weights = federated_averaging(weights_list)
    if mean_weights is None:
        return 0.0
        
    total_rel_div = 0.0
    num_keys = len(mean_weights)
    
    for key in mean_weights.keys():
        mean_arr = mean_weights[key]
        mean_norm = np.linalg.norm(mean_arr)
        
        # Calculate mean L2 norm of the difference for each client
        client_diffs = []
        for w in weights_list:
            diff_norm = np.linalg.norm(w[key] - mean_arr)
            client_diffs.append(diff_norm)
            
        avg_diff = np.mean(client_diffs)
        # Relative divergence for this parameter key
        rel_div = avg_diff / (mean_norm + 1e-8)
        total_rel_div += rel_div
        
    # Return average relative divergence across all parameter keys
    return float(total_rel_div / num_keys)
