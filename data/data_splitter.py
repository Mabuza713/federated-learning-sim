import os
import random
import shutil


def find_image_dirs(base_dir):
    tb_dir = None
    normal_dir = None

    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.lower() in ["tuberculosis", "tb"]:
                tb_dir = os.path.join(root, d)
            elif d.lower() == "normal":
                normal_dir = os.path.join(root, d)

    return tb_dir, normal_dir


def split_data():
    base_dir = os.path.join(os.path.dirname(__file__), "tuberculosis_dataset")
    if not os.path.exists(base_dir):
        print(
            f"Błąd: Katalog {base_dir} nie istnieje. Uruchom najpierw data_scrapper.py"
        )
        return

    tb_dir, normal_dir = find_image_dirs(base_dir)

    if not tb_dir or not normal_dir:
        print(
            "Błąd: Nie znaleziono folderów z obrazami Tuberculosis/Normal w pobranym zestawie."
        )
        return

    valid_exts = (".png", ".jpg", ".jpeg")
    tb_images = [
        os.path.join(tb_dir, f)
        for f in os.listdir(tb_dir)
        if f.lower().endswith(valid_exts)
    ]
    normal_images = [
        os.path.join(normal_dir, f)
        for f in os.listdir(normal_dir)
        if f.lower().endswith(valid_exts)
    ]

    print(f"Znaleziono obrazów chorych (Tuberculosis): {len(tb_images)}")
    print(f"Znaleziono obrazów zdrowych (Normal): {len(normal_images)}")

    # Ustawienie ziarna losowości dla powtarzalności wyników
    random.seed(42)
    random.shuffle(tb_images)
    random.shuffle(normal_images)

    # Serwer: Zbiór walidacyjny serwera (np. po 50 obrazów)
    val_tb_count = min(50, len(tb_images) // 10)
    val_normal_count = val_tb_count

    val_tb = tb_images[:val_tb_count]
    val_normal = normal_images[:val_normal_count]

    # Pozostałe obrazy do podziału dla szpitali
    remaining_all_tb = tb_images[val_tb_count:]
    remaining_all_normal = normal_images[val_normal_count:]

    # Szpital 3: Split 50% / 50% (Zrównoważony)
    # Przypisujemy po 200 obrazów Normal i Tuberculosis z puli pozostałych
    h3_tb_count = min(200, len(remaining_all_tb) // 3)
    h3_normal_count = h3_tb_count

    h3_tb = remaining_all_tb[:h3_tb_count]
    h3_normal = remaining_all_normal[:h3_normal_count]

    # Pozostałe obrazy dla h1 i h2
    remaining_tb = remaining_all_tb[h3_tb_count:]
    remaining_normal = remaining_all_normal[h3_normal_count:]

    # Szpital 1: Prawie same dane chorych (~90% Tuberculosis, 10% Normal)
    # Przypisujemy 90% z pozostałych chorych do szpitala 1
    h1_tb_count = int(len(remaining_tb) * 0.9)
    # Aby zachować proporcje ~90% chorych, zdrowych powinno być 1/9 liczby chorych
    h1_normal_count = h1_tb_count // 9

    h1_tb = remaining_tb[:h1_tb_count]
    h1_normal = remaining_normal[:h1_normal_count]

    # Szpital 2: Prawie same dane zdrowych (wszystkie pozostałe obrazy)
    h2_tb = remaining_tb[h1_tb_count:]
    h2_normal = remaining_normal[h1_normal_count:]

    splits = {
        "hospital_1": {
            "tb": h1_tb,
            "normal": h1_normal,
            "desc": "Prawie same dane chorych (~90% Tuberculosis)",
        },
        "hospital_2": {
            "tb": h2_tb,
            "normal": h2_normal,
            "desc": "Prawie same dane zdrowych (~98.5% Normal)",
        },
        "hospital_3": {
            "tb": h3_tb,
            "normal": h3_normal,
            "desc": "Zrównoważony split (50% Tuberculosis / 50% Normal)",
        },
        "server_val": {
            "tb": val_tb,
            "normal": val_normal,
            "desc": "Zbiór walidacyjny serwera (50% Tuberculosis / 50% Normal)",
        },
    }

    hospitals_dir = os.path.join(os.path.dirname(__file__), "hospitals")
    os.makedirs(hospitals_dir, exist_ok=True)

    print("\nRozpoczynanie dzielenia i kopiowania plików do folderów szpitali...")
    for hosp_name, data in splits.items():
        hosp_dir = os.path.join(hospitals_dir, hosp_name)
        if os.path.exists(hosp_dir):
            shutil.rmtree(hosp_dir)
        os.makedirs(hosp_dir)

        # Tworzenie podfolderów klas
        os.makedirs(os.path.join(hosp_dir, "Tuberculosis"))
        os.makedirs(os.path.join(hosp_dir, "Normal"))

        # Kopiowanie obrazów chorych
        for path in data["tb"]:
            shutil.copy2(
                path, os.path.join(hosp_dir, "Tuberculosis", os.path.basename(path))
            )

        # Kopiowanie obrazów zdrowych
        for path in data["normal"]:
            shutil.copy2(path, os.path.join(hosp_dir, "Normal", os.path.basename(path)))

        tb_len = len(data["tb"])
        normal_len = len(data["normal"])
        total = tb_len + normal_len
        tb_pct = (tb_len / total * 100) if total > 0 else 0
        normal_pct = (normal_len / total * 100) if total > 0 else 0

        print(f"\n[{hosp_name}] - {data['desc']}:")
        print(f"  Razem: {total} obrazów")
        print(f"  Chorzy (Tuberculosis): {tb_len} ({tb_pct:.2f}%)")
        print(f"  Zdrowi (Normal): {normal_len} ({normal_pct:.2f}%)")


if __name__ == "__main__":
    split_data()
