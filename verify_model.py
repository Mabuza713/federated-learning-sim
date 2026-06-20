import os
import sys
import requests

# Dodanie folderu src klienta do ścieżki wyszukiwania modułów, aby móc zaimportować train.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "apps", "client", "src")))

try:
    import numpy as np
    from train import SimpleCNN, deserialize_weights, CrossEntropyLoss
    from data_loader import get_dataloader
except ImportError as e:
    print(f"Błąd importu: Nie udało się zaimportować wymaganych modułów. Upewnij się, że struktura projektu jest poprawna. Szczegóły: {e}")
    sys.exit(1)

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8080")

def main():
    print("=== Sprawdzanie ostatecznego modelu na serwerze ===")
    print(f"Adres serwera: {SERVER_URL}\n")

    # 1. Pobranie statusu serwera
    try:
        status_resp = requests.get(f"{SERVER_URL}/status")
    except requests.exceptions.ConnectionError:
        print(f"Błąd połączenia: Serwer pod adresem {SERVER_URL} nie jest uruchomiony.")
        print("Upewnij się, że serwer/symulacja jest włączona przed uruchomieniem tego skryptu.")
        sys.exit(1)

    if status_resp.status_code == 200:
        status = status_resp.json()
        print("[STATUS SERWERA]")
        print(f"  - Aktualna runda na serwerze: {status.get('current_round')}")
        print(f"  - Wszystkich rund w symulacji: {status.get('total_rounds')}")
        print(f"  - Wymagana liczba klientów: {status.get('min_clients')}")
        print(f"  - Zgłoszeni klienci w tej rundzie: {', '.join(status.get('submitted_clients', [])) or 'Brak'}")
        
        history = status.get('validation_history', [])
        if history:
            print("  - Historia walidacji na serwerze:")
            for entry in history:
                print(f"    * Runda {entry['round']}: Strata = {entry['loss']:.4f}, Dokładność = {entry['accuracy']:.2f}%")
        print()
    else:
        print(f"Błąd przy pobieraniu statusu serwera (Kod: {status_resp.status_code}): {status_resp.text}")

    # 2. Pobranie modelu globalnego
    print("[POBIERANIE MODELU]")
    model_resp = requests.get(f"{SERVER_URL}/model")
    if model_resp.status_code == 200:
        data = model_resp.json()
        model_round = data.get("round")
        print(f"  - Pobrano wagi modelu z rundy: {model_round}")

        # Inicjalizacja klasy modelu SimpleCNN
        model = SimpleCNN()
        
        # Sprawdzenie wag przed załadowaniem (początkowa losowa inicjalizacja)
        initial_w2_mean = model.W2.mean()

        # Deserializacja wag i załadowanie ich do modelu
        try:
            deserialize_weights(model, data.get("weights"))
            print("  - Pomyślnie załadowano i zdeserializowano wagi modelu!")
            
            # Weryfikacja zmiany wag
            new_w2_mean = model.W2.mean()
            if initial_w2_mean == new_w2_mean:
                print("  - Ostrzeżenie: Wagi modelu nie zmieniły się po załadowaniu. Czy serwer został zresetowany?")
            else:
                print("  - Weryfikacja: Wagi uległy zmianie (zostały pomyślnie zsynchronizowane z serwerem).")

            print("\n[SZCZEGÓŁY MODELU]")
            print(f"  - Kształt wag warstwy 1 (W1): {model.W1.shape}")
            print(f"  - Kształt wag warstwy 2 (W2): {model.W2.shape}")
            print(f"  - Średnia wag warstwy W1: {model.W1.mean():.6f}")
            print(f"  - Średnia wag warstwy W2: {model.W2.mean():.6f}")
            print(f"  - Przykładowe obciążenie b2 (bias): {model.b2}")

            # Lokalna weryfikacja na danych walidacyjnych/testowych serwera
            val_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "hospitals", "server_val"))
            if os.path.exists(val_path):
                print("\n[LOKALNA WALIDACJA NA DANYCH TESTOWYCH]")
                try:
                    # Wyłączenie trybu treningowego
                    model.eval()
                    
                    # Załadowanie danych
                    val_loader = get_dataloader(val_path, batch_size=16, hospital_name="Weryfikacja")
                    
                    criterion = CrossEntropyLoss()
                    total_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for inputs, labels in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        total_loss += loss * inputs.shape[0]
                        preds = np.argmax(outputs, axis=1)
                        correct += np.sum(preds == labels)
                        total += labels.shape[0]
                        
                    avg_loss = total_loss / total if total > 0 else 0.0
                    accuracy = 100.0 * correct / total if total > 0 else 0.0
                    
                    print(f"  - Lokalizacja danych testowych: {val_path}")
                    print(f"  - Liczba obrazów testowych: {total}")
                    print(f"  - Strata walidacji (Loss): {avg_loss:.4f}")
                    print(f"  - Dokładność walidacji (Accuracy): {accuracy:.2f}%")
                except Exception as eval_err:
                    print(f"  - Błąd podczas lokalnej walidacji: {eval_err}")
            else:
                print(f"\n[LOKALNA WALIDACJA] Nie znaleziono katalogu z danymi testowymi pod adresem: {val_path}")

        except Exception as e:
            print(f"  - Błąd podczas deserializacji wag: {e}")
    else:
        print(f"Błąd przy pobieraniu modelu z serwera (Kod: {model_resp.status_code}): {model_resp.text}")

if __name__ == "__main__":
    main()
