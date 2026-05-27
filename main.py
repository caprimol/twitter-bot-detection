import os
from src.pipeline.orchestrator import MVPBotDetectionPipeline


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    TWEETS_CSV_PATH = os.path.join(project_root, "data", "raw", "tweets_merged.csv")
    LABELS_CSV_PATH = os.path.join(project_root, "data", "raw", "users_labels.csv")

    if not os.path.exists(TWEETS_CSV_PATH) or not os.path.exists(LABELS_CSV_PATH):
        print(f"\n[BŁĄD] Nie znaleziono plików w {os.path.join(project_root, 'data', 'raw')}!")
        return

    pipeline = MVPBotDetectionPipeline(max_seq_len=100)
    pipeline.run_pipeline(tweets_path=TWEETS_CSV_PATH, labels_path=LABELS_CSV_PATH)


if __name__ == "__main__":
    main()