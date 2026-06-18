# Wykrywanie spambotów na Twitterze (Deep Learning & Digital DNA)

Projekt służy do klasyfikacji i detekcji zaawansowanych kont zautomatyzowanych (Social Spambots) na platformie Twitter. System łączy analizę szeregów czasowych, semantykę tekstu (BERT) oraz sekwencje zachowań użytkownika ("Digital DNA" - warianty wpisów ACT: Action, Conversation, Retweet) przy użyciu sieci LSTM.

## Szybki start (Instrukcja krok po kroku)

### 1. Klonowanie i przygotowanie struktury danych
Pobierz repozytorium i upewnij się, że pobrana z [Google Drive](https://drive.google.com/drive/folders/1Yno8gUorhEHxfyTSZgIoGFrOCrQCSvIC?usp=sharing) paczka danych `datasets_full.zip` znajduje się bezpośrednio w głównym katalogu projektu.

Rozpakuj archiwum ZIP za pomocą terminala:
```bash
unzip datasets_full.zip
```

Upewnij się, że po rozpakowaniu powstał katalog o nazwie `datasets_full.csv/` zawierający pliki kont (np. `genuine_accounts.csv/tweets.csv`, `social_spambots_1.csv/tweets.csv` itd.).

### 2. Tworzenie i aktywacja środowiska wirtualnego

```bash
# Utworzenie środowiska o nazwie tf_env
python3 -m venv tf_env

# Aktywacja (Linux / WSL / macOS)
source tf_env/bin/activate

# Aktywacja (Windows PowerShell)
.\tf_env\Scripts\Activate.ps1
```

### 3. Instalacja bibliotek

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Uruchomienie potoku badań ablacyjnych

```bash
python main.py
```

## Przebieg działania programu

1. **Wczytanie danych:** Ładowane są surowe metadane oraz pliki tekstowe.
2. **Preprocessing:** Dane są czyszczone, sortowane chronologicznie i normalizowane.
3. **Koder BERT:** Przy pierwszym uruchomieniu pobierany jest model językowy `all-MiniLM-L6-v2`. Przetworzone teksty zapisują się do pliku podręcznego `bert_encoded_tweets.pkl` (2.38 GB), dzięki czemu kolejne uruchomienia skryptu trwają zaledwie sekundy.
4. **Pętla eksperymentów (Ablation Studies):** Skrypt automatycznie, w jednej sekwencji uruchamia 6 różnych konfiguracji modeli (Naiwny Bayes, Tylko Tekst, Tylko Czas, Tylko DNA [ACT], Hybryda 1xLSTM, Hybryda 2xLSTM).
5. **Balansowanie klas:** Wagi klas obliczane są dynamicznie na podstawie faktycznych proporcji zbioru, eliminując problem nadreprezentacji botów.
6. **Wykresy i wyniki:** Podsumowująca tabelka celności (Accuracy) drukowana jest w konsoli, a wykresy zbieżności funkcji straty i dokładności automatycznie lądują w wygenerowanym folderze `charts/`.