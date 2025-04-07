import json
import itertools
import random
from tqdm import tqdm

include_review_weights = True
include_tour_weight = True
review_weight_chance = 1.0
tour_weight_chance = 1.0


def compute_review_weight(review):
    base = review["score"] / 5.0
    return round(base * (1 + review["text_length"] / 20), 3) if review["has_text"] else round(base * 0.6, 3)

def compute_tour_weight(tour):
    count_weight = min(tour["review_count"] / 500, 1.0)
    avg_score_weight = tour["avg_score"] / 5.0
    return round((count_weight * 0.6 + avg_score_weight * 0.4), 3)

def format_review(review):
    entry = {
        "score": review["score"],
        "text": review["text"],
        "text_length": review["text_length"],
        "has_text": review["has_text"]
    }
    if include_review_weights and random.random() < review_weight_chance:
        entry["weight"] = compute_review_weight(review)
    return entry

def format_tour(tour):
    entry = {
        "id": tour["id"],
        "title": tour["title"],
        "description_long": tour["description_long"],
        "avg_score": tour["avg_score"],
        "review_count": tour["review_count"],
        "text_review_count": tour["text_review_count"],
        "text_review_ratio": tour["text_review_ratio"],
        "avg_text_length": tour["avg_text_length"],
        "reviews": [format_review(r) for r in tour["reviews"]]
    }
    if include_tour_weight and random.random() < tour_weight_chance:
        entry["weight"] = compute_tour_weight(tour)
    return entry

def generate_pairs(data):
    pairs = []
    for a, b in tqdm(itertools.combinations(data, 2), desc="Generating pairs"):
        if a["review_count"] < 2 or b["review_count"] < 2:
            continue
        label = 1 if compute_tour_weight(a) > compute_tour_weight(b) else 0
        pairs.append({
            "tour_a": format_tour(a),
            "tour_b": format_tour(b),
            "label": label
        })
    return pairs

def convert(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    pairs = generate_pairs(data)
    with open(output_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(pairs)} pairs to {output_file}")

if __name__ == "__main__":
    convert("tours_train.jsonl", "pairwise_train.jsonl")
    convert("tours_test.jsonl", "pairwise_test.jsonl")
