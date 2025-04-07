import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import json
from typing import List, Dict

import torch
import numpy as np
from scipy.special import expit  # sigmoid

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)

# ✅ Константы
# MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "DeepPavlov/rubert-base-cased"
TRAIN_PATH = "pairwise_train.jsonl"
TEST_PATH = "pairwise_test.jsonl"
OUTPUT_DIR = "./pairwise_lora_output"
MAX_LENGTH = 512

# ✅ Конфиг LoRA
PEFT_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# ✅ Формирование текста тура
def tour_to_text(tour: Dict) -> str:
    parts = [
        f"Название тура: {tour['title']}",
        f"Описание: {tour['description_long']}",
        f"Средняя оценка: {tour['avg_score']}, Кол-во отзывов: {tour['review_count']}, "
        f"Доля текстовых: {tour['text_review_ratio']}, Средняя длина: {tour['avg_text_length']}",
        f"Вес тура: {tour.get('weight', 1.0)}",
        "Отзывы:"
    ]
    for review in tour["reviews"]:
        text = review["text"] if review["has_text"] else "[Без текста]"
        line = f"- Оценка: {review['score']}. Текст: {text} (Вес: {review.get('weight', 1.0)})"
        parts.append(line)
    return "\n".join(parts)

# ✅ Загрузка обучающих данных
def load_pairwise_data(file_path: str) -> List[Dict]:
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text_a = tour_to_text(item["tour_a"])
            text_b = tour_to_text(item["tour_b"])
            input_text = f"Тур A:\n{text_a}\n\nТур B:\n{text_b}\n\nКакой тур лучше?"
            samples.append({"text": input_text, "labels": float(item["label"])})
    return samples

# ✅ Токенизация
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

# ✅ Создание датасетов
train_data = Dataset.from_list(load_pairwise_data(TRAIN_PATH)).map(tokenize_fn, batched=False)
test_data = Dataset.from_list(load_pairwise_data(TEST_PATH)).map(tokenize_fn, batched=False)

train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ✅ Загрузка модели и PEFT
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,  # один логит
    torch_dtype=torch.float32
)
model.gradient_checkpointing_enable()
model = get_peft_model(model, PEFT_CONFIG)

# ✅ Метрика
def compute_metrics(p: EvalPrediction):
    logits = p.predictions.reshape(-1)
    labels = p.label_ids.astype(int)

    probs = expit(logits)  # sigmoid
    preds = (probs > 0.5).astype(int)

    acc = np.mean(preds == labels)
    return {
        "accuracy": acc,
        # При необходимости можно добавить другие метрики:
        # "precision": precision_score(labels, preds),
        # "recall": recall_score(labels, preds),
        # "f1": f1_score(labels, preds),
    }

# ✅ Аргументы тренировки
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none",  # без wandb и т.п.
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data.shuffle(seed=42),
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ✅ Старт обучения
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Обучение завершено и модель сохранена.")
