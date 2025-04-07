import psycopg2
import json
from tqdm import tqdm
from collections import defaultdict
import statistics
import random

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "WellComeGroup",
    "user": "WellComeGroup",
    "password": "WellComeGroup"
}

TRAIN_FILE = "tours_train.jsonl"
TEST_FILE = "tours_test.jsonl"
TRAIN_RATIO = 0.8  # 80% на обучение, 20% на тест

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def fetch_data():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, title, description_long FROM synthetic.tours WHERE deleted = FALSE")
    tours = cur.fetchall()
    cur.execute("SELECT tour_id, score, text FROM synthetic.feedback")
    feedback = cur.fetchall()
    cur.close()
    conn.close()
    return tours, feedback

def process_data(tours, feedback):
    tour_map = {t[0]: {"id": t[0], "title": t[1], "description_long": t[2]} for t in tours}
    feedback_map = defaultdict(list)

    for tour_id, score, text in feedback:
        feedback_map[tour_id].append({
            "score": score,
            "text": text.strip() if text else "",
            "text_length": len(text.strip().split()) if text else 0,
            "has_text": bool(text and text.strip())
        })

    result = []
    for tour_id, tour_info in tqdm(tour_map.items(), desc="Processing tours"):
        reviews = feedback_map.get(tour_id, [])
        if not reviews:
            continue

        avg_score = statistics.mean([r["score"] for r in reviews])
        text_reviews = [r for r in reviews if r["has_text"]]
        text_lengths = [r["text_length"] for r in text_reviews]

        tour_entry = {
            "id": tour_info["id"],
            "title": tour_info["title"],
            "description_long": tour_info["description_long"],
            "avg_score": round(avg_score, 3),
            "review_count": len(reviews),
            "text_review_count": len(text_reviews),
            "text_review_ratio": round(len(text_reviews) / len(reviews), 3),
            "avg_text_length": round(statistics.mean(text_lengths), 2) if text_lengths else 0,
            "reviews": reviews
        }

        result.append(tour_entry)

    return result

def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def split_and_save(data):
    random.shuffle(data)
    train_size = int(len(data) * TRAIN_RATIO)
    save_jsonl(data[:train_size], TRAIN_FILE)
    save_jsonl(data[train_size:], TEST_FILE)
    print(f"✅ Saved {train_size} train and {len(data) - train_size} test samples")

if __name__ == "__main__":
    tours, feedback = fetch_data()
    data = process_data(tours, feedback)
    split_and_save(data)
