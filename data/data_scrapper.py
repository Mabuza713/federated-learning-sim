import os
import shutil
import kagglehub

DATA_DIR = os.path.join(os.path.dirname(__file__), "tuberculosis_dataset")


def download_tb_dataset():
    """Pobiera dataset Tuberculosis (TB) Chest X-ray z platformy Kaggle

    i zapisuje go lokalnie w folderze data/tuberculosis_dataset.
    """
    if os.path.exists(DATA_DIR) and any(os.scandir(DATA_DIR)):
        print(f"Dataset już istnieje w lokalizacji: {DATA_DIR}")
        return DATA_DIR

    print("Rozpoczynanie pobierania datasetu z Kaggle...")
    cache_path = kagglehub.dataset_download(
        "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
    )
    print(f"Pobrano do cache: {cache_path}")

    print(f"Zapisywanie plików do lokalizacji docelowej: {DATA_DIR}...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Kopiujemy zawartość z cache do folderu projektu
    for item in os.listdir(cache_path):
        src_path = os.path.join(cache_path, item)
        dst_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    print("Dataset został pomyślnie zapisany w projekcie.")
    return DATA_DIR


if __name__ == "__main__":
    download_tb_dataset()
