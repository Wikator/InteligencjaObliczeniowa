import os
import shutil
import random

# Ścieżka do katalogu z obrazami
photos_dir = 'photos'

# Ścieżki do katalogów treningowych i walidacyjnych
train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# Utwórz katalogi treningowe i walidacyjne, jeśli nie istnieją
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Pobierz listę plików w katalogu photos
photo_files = os.listdir(photos_dir)

# Zmieszaj pliki
random.shuffle(photo_files)

# Oblicz liczbę plików do treningu i walidacji w stosunku 70:30
num_train = int(0.7 * len(photo_files))
num_val = len(photo_files) - num_train

# Przekopiuj pliki do katalogu treningowego
for i in range(num_train):
    src = os.path.join(photos_dir, photo_files[i])
    dst = os.path.join(train_dir, photo_files[i])
    shutil.copyfile(src, dst)

# Przekopiuj pliki do katalogu walidacyjnego
for i in range(num_train, num_train + num_val):
    src = os.path.join(photos_dir, photo_files[i])
    dst = os.path.join(val_dir, photo_files[i])
    shutil.copyfile(src, dst)

print("Podział danych na zbiory treningowy i walidacyjny zakończony pomyślnie.")
