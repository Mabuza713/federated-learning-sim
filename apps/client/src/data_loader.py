import os
import time
import random
from PIL import Image
import numpy as np

class NumPyDataset:
    def __init__(self, data_path: str):
        self.data_path = data_path
        # Sort classes alphabetically: e.g. ['NORMAL', 'PNEUMONIA']
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
    def __init__(self, dataset: NumPyDataset, batch_size: int, shuffle: bool = True):
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
                        # Normalize to [-1, 1]: (x - 0.5) / 0.5
                        img_arr = np.array(img, dtype=np.float32) / 255.0
                        img_arr = (img_arr - 0.5) / 0.5
                        # Shape: (3, 64, 64)
                        img_arr = np.transpose(img_arr, (2, 0, 1))
                        batch_inputs.append(img_arr)
                        batch_labels.append(label)
                except Exception:
                    continue
            
            if len(batch_inputs) > 0:
                yield np.array(batch_inputs, dtype=np.float32), np.array(batch_labels, dtype=np.int64)

def get_dataloader(data_path: str, batch_size: int, hospital_name: str) -> NumPyDataLoader:
    """
    Creates and returns a NumPyDataLoader for the client's dataset.
    If the dataset is missing or empty, it will wait and retry periodically.
    """
    while True:
        if not os.path.exists(data_path):
            print(f"[{hospital_name}] Directory {data_path} does not exist. Waiting 10s...")
            time.sleep(10)
            continue
        try:
            dataset = NumPyDataset(data_path)
            if len(dataset) == 0:
                print(f"[{hospital_name}] No images found in {data_path}. Waiting 10s...")
                time.sleep(10)
                continue
            
            dataloader = NumPyDataLoader(dataset, batch_size=batch_size, shuffle=True)
            print(f"[{hospital_name}] Successfully loaded dataset with {len(dataset)} images (Classes: {dataset.classes}).")
            return dataloader
        except Exception as e:
            print(f"[{hospital_name}] Error loading dataset: {e}. Waiting 10s...")
            time.sleep(10)
