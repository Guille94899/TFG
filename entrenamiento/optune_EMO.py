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



# CONFIG

set_seed(42)

TRAIN_PATH = "train_emo.json"
TEST_PATH = "test_emo.json"

MODEL_CHECKPOINT = "roberta-base"

RESULTS_DIR = "./results_ef_optuna"
SAVE_DIR = "emo_model_roberta_optuna_best"

MAX_LENGTH = 512
PRED_THRESHOLD = 0.5

N_TRIALS = 20  
VAL_SPLIT = 0.2


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
    mapping = {"EP_ME_P": "EP_MEP_P"}  # tu mapping
    return mapping.get(x, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_json_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    texts = [d["text"] for d in raw]
    labels_raw = [ensure_labels(d) for d in raw]
    labels_norm = [[norm_label(x) for x in lst] for lst in labels_raw]
    return texts, labels_norm



# Load TRAIN

train_texts, train_labels_norm = load_json_dataset(TRAIN_PATH)


# Binarizer + pos_weight (solo con TRAIN)

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_labels_norm).astype("float32")

num_labels = len(mlb.classes_)
id2label = {i: c for i, c in enumerate(mlb.classes_)}
label2id = {c: i for i, c in enumerate(mlb.classes_)}

print("\nEtiquetas del modelo:")
print(mlb.classes_)

n_samples = Y_train.shape[0]
pos = Y_train.sum(axis=0)
neg = n_samples - pos
pos = np.clip(pos, 1.0, None)
pos_weight = torch.tensor(neg / pos, dtype=torch.float32)

print("\npos_weight =", pos_weight)

train_ds = Dataset.from_dict({"text": train_texts, "labels": [list(row) for row in Y_train]})


#  Tokenizer + split (train/val)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
    enc["labels"] = np.array(batch["labels"], dtype="float32")
    return enc

tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text"])
split = tokenized_train.train_test_split(test_size=VAL_SPLIT, seed=42)
train_split = split["train"]
val_split = split["test"]

data_collator = DataCollatorWithPadding(tokenizer)

# Custom Trainer 

class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    probs = sigmoid(logits)
    preds = (probs >= PRED_THRESHOLD).astype(int)

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



# TrainingArguments base + model_init (necesario para Optuna)

use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
use_fp16 = use_cuda and not use_bf16

base_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    eval_strategy="epoch",
    save_strategy="no",          
    logging_steps=100,
    report_to="none",
    fp16=bool(use_fp16),
    bf16=bool(use_bf16),
)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

trainer = WeightedBCETrainer(
    pos_weight=pos_weight,
    model_init=model_init,
    args=base_args,
    train_dataset=train_split,
    eval_dataset=val_split,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# Optuna search

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 6e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 4, 16),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
    }

def compute_objective(metrics):
    return metrics["eval_f1_macro"]

best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    n_trials=N_TRIALS,
    hp_space=hp_space,
    compute_objective=compute_objective,
)

print("\n=== BEST RUN ===")
print(best_run)


# Final training with best hyperparams

final_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    fp16=bool(use_fp16),
    bf16=bool(use_bf16),
    **best_run.hyperparameters,
)

final_trainer = WeightedBCETrainer(
    pos_weight=pos_weight,
    model=model_init(),
    args=final_args,
    train_dataset=train_split,
    eval_dataset=val_split,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

final_trainer.train()

#  Final TEST evaluation (solo al final)

if os.path.exists(TEST_PATH):
    print("\n--- EVALUACIÓN FINAL EN TEST ---")
    test_texts, test_labels_norm = load_json_dataset(TEST_PATH)
    Y_test = mlb.transform(test_labels_norm).astype("float32")
    test_ds = Dataset.from_dict({"text": test_texts, "labels": [list(row) for row in Y_test]})
    tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    test_metrics = final_trainer.evaluate(tokenized_test)
    print("\nMétricas en TEST:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
else:
    print(f"\n[AVISO] No encuentro '{TEST_PATH}'. Me salto la evaluación en test.")


#  Save

os.makedirs(SAVE_DIR, exist_ok=True)
final_trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

with open(os.path.join(SAVE_DIR, "label_encoder.json"), "w", encoding="utf-8") as f:
    json.dump(list(mlb.classes_), f, indent=2, ensure_ascii=False)

with open(os.path.join(SAVE_DIR, "best_hyperparameters.json"), "w", encoding="utf-8") as f:
    json.dump(best_run.hyperparameters, f, indent=2, ensure_ascii=False)

print(f"\n✅ Modelo guardado en: {SAVE_DIR}")
print(f"✅ HP guardados en: {os.path.join(SAVE_DIR, 'best_hyperparameters.json')}")