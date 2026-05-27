import pandas as pd
import os


def merge_specific_cresci_datasets(output_dir, target_folders):
    all_tweets = []
    all_users = []

    print("Rozpoczynam przetwarzanie folderów (users.csv oraz tweets.csv)...\n")

    for folder_info in target_folders:
        target_folder = folder_info["path"]
        label = folder_info["label"]

        if not os.path.exists(target_folder):
            print(f"  [BŁĄD] Folder nie istnieje: {target_folder}")
            continue

        print(f"Przetwarzanie folderu: {target_folder} (Etykieta: {label})")

        users_file = os.path.join(target_folder, 'users.csv')
        if os.path.exists(users_file):
            try:
                try:
                    df_users = pd.read_csv(users_file, low_memory=False, on_bad_lines='skip', encoding='utf-8')
                except UnicodeDecodeError:
                    df_users = pd.read_csv(users_file, low_memory=False, on_bad_lines='skip', encoding='latin-1')

                if 'id' in df_users.columns:
                    df_users.rename(columns={'id': 'user_id'}, inplace=True)

                df_labels = df_users[['user_id']].copy()
                df_labels['label'] = label
                all_users.append(df_labels)
                print(f"  [OK] Wczytano plik: {users_file}")
            except Exception as e:
                print(f"  [BŁĄD] Wczytywania {users_file}: {e}")
        else:
            print(f"  [BŁĄD] Nie znaleziono pliku {users_file}")

        tweets_file = os.path.join(target_folder, 'tweets.csv')

        if os.path.exists(tweets_file):
            try:
                try:
                    df_tweets = pd.read_csv(tweets_file, low_memory=False, on_bad_lines='skip', encoding='utf-8')
                except UnicodeDecodeError:
                    df_tweets = pd.read_csv(tweets_file, low_memory=False, on_bad_lines='skip', encoding='latin-1')

                if 'user_id' not in df_tweets.columns and 'author_id' in df_tweets.columns:
                    df_tweets.rename(columns={'author_id': 'user_id'}, inplace=True)
                all_tweets.append(df_tweets)
                print(f"  [OK] Wczytano plik: {tweets_file}")
            except Exception as e:
                print(f"  [BŁĄD] Wczytywania {tweets_file}: {e}")
        else:
            print(f"  [BŁĄD] Nie znaleziono pliku {tweets_file}")

    print("\nŁączenie i zapisywanie ramek danych...")
    os.makedirs(output_dir, exist_ok=True)

    if all_users:
        final_users = pd.concat(all_users, ignore_index=True)
        final_users.drop_duplicates(subset=['user_id'], inplace=True)
        labels_path = os.path.join(output_dir, 'users_labels.csv')
        final_users.to_csv(labels_path, index=False)
        print(f"Zapisano etykiety: {labels_path}")

    if all_tweets:
        final_tweets = pd.concat(all_tweets, ignore_index=True)
        tweets_path = os.path.join(output_dir, 'tweets_merged.csv')
        final_tweets.to_csv(tweets_path, index=False)
        print(f"Zapisano tweety: {tweets_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "src" else script_dir
    katalog_docelowy = os.path.join(project_root, "data", "raw")

    target_folders = [
        {"path": os.path.join(katalog_docelowy, "cresci-2017", "traditional_spambots_1.csv"), "label": 1},
        {"path": os.path.join(katalog_docelowy, "cresci-2017", "genuine_accounts.csv"), "label": 0}
    ]

    merge_specific_cresci_datasets(katalog_docelowy, target_folders)