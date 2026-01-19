import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# --- Mapeo de clases completas ---
label_map = {
    "EF_DIR": 0,
    "EF_DIR_P": 1,
    "EF_DM": 2,
    "EF_DM_P": 3,
    "EF_INT": 4,
    "EF_INT_P": 5,
    "EF_INC": 6,
    "EF_INC_P": 7,
    "EF_NRM": 8,
    "EF_NRM_P": 9,
    "EF_POT": 10,
    "EF_POT_P": 11
}
index_to_label = {v: k for k, v in label_map.items()}

# --- Cargar y filtrar dataset ---
with open("acaia_feedback_epistemic_dataset.json", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

def is_effective(label):
    return label.startswith("<EF")

def get_full_class(label):
    clean = label.replace("<", "").replace(">", "").replace(" ", "").split(",")
    if "EF" not in clean:
        return None
    sub = None
    if "DIR" in clean:
        sub = "DIR"
    elif "DM" in clean:
        sub = "DM"
    elif "INT" in clean:
        sub = "INT"
    elif "INC" in clean:
        sub = "INC"
    elif "NRM" in clean:
        sub = "NRM"
    elif "POT" in clean:
        sub = "POT"
    else:
        return None
    suffix = "_P" if "P" in clean else ""
    return f"EF_{sub}{suffix}"

# Filtrar y asignar clases
filtered = [
    (entry["text"], label_map[get_full_class(entry["label"])])
    for entry in raw_data
    if is_effective(entry["label"]) and get_full_class(entry["label"]) in label_map
]

texts = [t for t, l in filtered]
labels = [l for t, l in filtered]

# --- Tokenización ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

# --- Dataset personalizado ---
class Effective12ClassDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

dataset = Effective12ClassDataset(encodings, labels)

# --- Modelo ---
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=12
)

# --- Entrenamiento ---
training_args = TrainingArguments(
    output_dir="./bert-effective-12",
    per_device_train_batch_size=8,
    num_train_epochs=4,
    logging_dir="./logs_effective12",
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# --- Guardar modelo entrenado ---
model.save_pretrained("./bert-effective-stance")
tokenizer.save_pretrained("./bert-effective-stance")
