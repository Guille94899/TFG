import os
import json
import re
import numpy as np
import torch

from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

from torch.nn import BCEWithLogitsLoss



# CONFIGURACION

set_seed(42)

DATA_PATH = "train_ef.json"

MODEL_CHECKPOINT = "microsoft/deberta-v3-base"
RESULTS_DIR = "./results_ep"
SAVE_DIR = "deberta_ep_final"

MAX_LENGTH = 512
THRESH = 0.5

# HIPERPARÁMETROS ÓPTIMOS (Optuna)
LEARNING_RATE = 3.0358233863984693e-05
NUM_EPOCHS = 15
TRAIN_BATCH = 8
EVAL_BATCH = 8
WEIGHT_DECAY = 0.0028376159480514975
WARMUP_RATIO = 0.007962962268492847

# Parámetros estables
MAX_GRAD_NORM = 1.0
LR_SCHEDULER = "linear"

# HELPERS

def ensure_labels(entry):
    labs = entry.get("labels", entry.get("label", []))
    if isinstance(labs, str):
        labs = [labs]
    return [l.strip() for l in labs if l and isinstance(l, str)]


def norm_label(x: str) -> str:
    x = x.strip().strip("<>").strip()
    x = x.replace(", ", "_").replace(",", "_")
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"_+", "_", x)
    x = re.sub(r"_+$", "", x)

    mapping = {"EP_ME_P": "EP_MEP_P"}
    return mapping.get(x, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



# Cargar datos

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

texts = [d["text"] for d in raw]
labels_raw = [ensure_labels(d) for d in raw]
labels_norm = [[norm_label(x) for x in lst] for lst in labels_raw]

# Binarizador multilabel
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels_norm).astype("float32")

num_labels = len(mlb.classes_)
id2label = {i: c for i, c in enumerate(mlb.classes_)}
label2id = {c: i for i, c in enumerate(mlb.classes_)}

print("\nEtiquetas del modelo:")
print(list(mlb.classes_))

dataset = Dataset.from_dict({
    "text": texts,
    "labels": [list(row) for row in Y],
})



# Tokenización + split SOLO train/val (90/10)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
    enc["labels"] = batch["labels"]
    return enc

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

split = tokenized.train_test_split(test_size=0.10, seed=42)
train_ds = split["train"].with_format("torch")
val_ds = split["test"].with_format("torch")

data_collator = DataCollatorWithPadding(tokenizer)



# Trainer multilabel

class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = BCEWithLogitsLoss()
        labels = labels.float()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    probs = sigmoid(logits)
    preds = (probs >= THRESH).astype(int)

    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )

    return {
        "f1_macro": float(f1_ma),
        "f1_micro": float(f1_mi),
        "precision_macro": float(p_ma),
        "recall_macro": float(r_ma),
    }


# TrainingArguments

use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
use_fp16 = use_cuda and not use_bf16

training_args = TrainingArguments(
    output_dir=RESULTS_DIR,

    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=EVAL_BATCH,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER,
    max_grad_norm=MAX_GRAD_NORM,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,

    metric_for_best_model="f1_macro",
    greater_is_better=True,

    logging_steps=50,
    report_to="none",

    fp16=bool(use_fp16),
    bf16=bool(use_bf16),
)


# Modelo + entrenamiento

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
)

trainer = BCETrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Guardar modelo FINAL

os.makedirs(SAVE_DIR, exist_ok=True)

trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

with open(os.path.join(SAVE_DIR, "label_encoder.json"), "w", encoding="utf-8") as f:
    json.dump(list(mlb.classes_), f, indent=2, ensure_ascii=False)

print(f"\n✅ Modelo FINAL guardado en: {SAVE_DIR}")
