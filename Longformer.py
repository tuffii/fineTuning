import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.simplefilter("ignore", category=FutureWarning)

import json
from typing import List, Dict
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    LongformerConfig
)
import torch

# Конфигурация
MODEL_NAME = "allenai/longformer-base-4096"  # Модель с поддержкой 4096 токенов
DATA_PATH = "pairwise_dataset.jsonl"
OUTPUT_DIR = "./longformer_lora_output"
MAX_LENGTH = 3072  # Можно увеличить до 4096 при наличии GPU 24GB+

# Настройки PEFT
PEFT_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none"
)

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = MAX_LENGTH

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# Подготовка данных
raw_data = load_pairwise_data(DATA_PATH)
dataset = Dataset.from_list(raw_data)
dataset = dataset.map(tokenize_fn, batched=False)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Загрузка модели с адаптацией под длинный контекст
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    attention_window=512
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, PEFT_CONFIG)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="no",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_total_limit=1,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none",
    optim="adafactor"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset.shuffle(seed=42)
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Обучение Longformer завершено. Модель сохранена в:", OUTPUT_DIR)