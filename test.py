import warnings
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map

# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="for .*: copying from a non-meta parameter.*")

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Настройки ROCm
torch.cuda.empty_cache()
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

model_name = "t-tech/T-lite-it-1.0"

# Инициализируем модель без загрузки весов
with init_empty_weights():
    model_tmp = AutoModelForCausalLM.from_pretrained(model_name)

# Распределение памяти: максимум на GPU, минимум на CPU
max_memory = {
    0: "14.5GiB",  # максимум на GPU
    "cpu": "2GiB"  # минимально возможное на CPU
}

# Распределение слоев по устройствам
device_map = infer_auto_device_map(
    model_tmp,
    max_memory=max_memory,
    no_split_module_classes=["GPT2Block", "LlamaDecoderLayer"]
)

# Загрузка модели с учетом device_map
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch.bfloat16
)

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Генерация текста
inputs = tokenizer("La capitale dell'Italia è", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.7,
        do_sample=True,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Результат
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {decoded_output}")

# Использование памяти
print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
