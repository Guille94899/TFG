import os
import json
import re
import random
import numpy as np
import torch

from datasets import Dataset
from seqeval.metrics import f1_score as seq_f1

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

set_seed(42)
random.seed(42)


# CONFIG

DATA_PATH = "data_ready_emo.cleaned.json"  
MODEL_CHECKPOINT = "roberta-base"
OUTPUT_DIR = "roberta_emo_bio"

MAX_LENGTH = 256
NUM_EPOCHS = 16
LR = 2e-5
NEG_RATIO = 0.4 

TAG_RE = re.compile(r"<<(EMO_[A-Z]+(?:_[A-Z]+)?)>>")
BOM = "\ufeff"

LEAD_PUNCT_RE = re.compile(r"^[^\w']+")
TRAIL_PUNCT_RE = re.compile(r"[^\w']+$")

# ----------------------------
# Utils
# ----------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace(BOM, "")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def detach_tags(text: str) -> str:
    """
    Asegura que <<EMO_...>> quede separado por espacios aunque venga pegado a ) . “ ” etc.
    (Esto evita que el tag se pierda por tokenización.)
    """
    if not text:
        return text
    text = TAG_RE.sub(r" <<\1>> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_all_tags(output_text: str) -> str:
    return re.sub(r"\s*<<EMO_[A-Z]+(?:_[A-Z]+)?>>", "", output_text)

def words_split(text: str):
    return text.split()

def norm_token(tok: str) -> str:
    tok = LEAD_PUNCT_RE.sub("", tok)
    tok = TRAIL_PUNCT_RE.sub("", tok)
    return tok.strip()

# ----------------------------
# Span extraction
# ----------------------------
def spans_from_output(output_text: str):
    """
    Extrae spans como lista de (span_words, tag) desde output_text (con tags).
    Regla:
    - Si el tag está dentro de paréntesis => span = contenido desde '(' hasta antes del tag (limpiado)
    - Si no => span = palabra anterior (limpiada)
    """
    spans = []
    out_words = words_split(output_text)

    for i, w in enumerate(out_words):
        m = TAG_RE.search(w)  
        if not m:
            continue
        tag = m.group(1)

        open_idx = None
        for j in range(i - 1, max(-1, i - 40), -1):
            if "(" in out_words[j]:
                open_idx = j
                break

        close_near = False
        for k in range(i, min(len(out_words), i + 8)):
            if ")" in out_words[k]:
                close_near = True
                break

        if open_idx is not None and close_near:
            span_words_raw = out_words[open_idx:i]
            span_words = [norm_token(x.replace("(", "").replace(")", "")) for x in span_words_raw]
            span_words = [x for x in span_words if x]
            if span_words:
                spans.append((span_words, tag))
                continue

        if i > 0:
            prev = norm_token(out_words[i - 1].replace("(", "").replace(")", ""))
            if prev:
                spans.append(([prev], tag))

    return spans

def build_bio_labels(input_text: str, output_text: str):
    words = words_split(input_text)
    labels = ["O"] * len(words)

    spans = spans_from_output(output_text)

    words_norm = [norm_token(w.replace("(", "").replace(")", "")) for w in words]  

    used = [False] * len(words)

    for span_words, tag in spans:
        L = len(span_words)
        if L == 0:
            continue

        for i in range(0, len(words) - L + 1):
            if words_norm[i:i+L] == span_words and not any(used[i:i+L]):
                labels[i] = f"B-{tag}"
                for j in range(i+1, i+L):
                    labels[j] = f"I-{tag}"
                for j in range(i, i+L):
                    used[j] = True
                break

    return words, labels


# Load 

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

raw2 = []
for ex in raw:
    inp = clean_text(ex.get("input_text", ""))
    out = clean_text(ex.get("output_text", ""))


    out = detach_tags(out)  

    if not inp or not out:
        continue

    raw2.append({"input_text": inp, "output_text": out})

    if random.random() < NEG_RATIO:
        raw2.append({
            "input_text": inp,
            "output_text": clean_text(strip_all_tags(out))
        })

raw = raw2
print("Total ejemplos (con negativos):", len(raw))


# Build BIO dataset

examples = []
label_set = {"O"}

for ex in raw:
    words, labs = build_bio_labels(ex["input_text"], ex["output_text"])
    if not words:
        continue
    for l in labs:
        label_set.add(l)
    examples.append({"words": words, "bio": labs})

label_list = sorted(label_set)
label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for l,i in label2id.items()}

print("Num labels BIO:", len(label_list))
print("Sample labels:", label_list[:25])

dataset = Dataset.from_list(examples).train_test_split(test_size=0.1, seed=42)


# Tokenize + align (RoBERTa + pretokenized)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT,
    use_fast=True,
    add_prefix_space=True,  
)

def tokenize_align(batch):
    tok = tokenizer(
        batch["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    aligned = []
    for i in range(len(batch["words"])):
        word_ids = tok.word_ids(batch_index=i)
        wlabs = batch["bio"][i]
        prev = None
        lab_ids = []
        for w in word_ids:
            if w is None:
                lab_ids.append(-100)
            else:
                if w != prev:
                    lab_ids.append(label2id[wlabs[w]])
                else:
                    lab_ids.append(-100)
            prev = w
        aligned.append(lab_ids)

    tok["labels"] = aligned
    return tok

tokds = dataset.map(tokenize_align, batched=True, remove_columns=dataset["train"].column_names)


#  Model

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

collator = DataCollatorForTokenClassification(tokenizer)


# Entity-F1 

EVAL_WORDS = dataset["test"]["words"]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_seqs = []
    pred_seqs = []

    for words, lab_ids, pred_ids in zip(EVAL_WORDS, labels, preds):
        enc = tokenizer([words], is_split_into_words=True, truncation=True, max_length=MAX_LENGTH)
        word_ids = enc.word_ids(0)

        true_w = ["O"] * len(words)
        pred_w = ["O"] * len(words)
        seen = set()

        for t, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            li = lab_ids[t]
            pi = pred_ids[t]
            if li != -100:
                true_w[wid] = id2label[int(li)]
            pred_w[wid] = id2label[int(pi)]

        true_seqs.append(true_w)
        pred_seqs.append(pred_w)

    f1 = seq_f1(true_seqs, pred_seqs)
    return {"entity_f1": float(f1)}

#  Train (best by entity_f1)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    learning_rate=LR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,

    logging_steps=50,
    report_to="none",

    load_best_model_at_end=True,
    metric_for_best_model="entity_f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokds["train"],
    eval_dataset=tokds["test"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Mejor modelo (por entity_f1) guardado en:", OUTPUT_DIR)

# el mejor modelo del trainer en inferencia
model = trainer.model
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# 7) Inferencia: insertar tags
# ----------------------------
def predict_word_labels(sentence: str):
    words = words_split(sentence)
    enc = tokenizer([words], is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=MAX_LENGTH)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits[0]

    pred_ids = logits.argmax(-1).detach().cpu().tolist()

    word_ids = tokenizer([words], is_split_into_words=True,
                         truncation=True, max_length=MAX_LENGTH).word_ids(0)

    pred_word = ["O"] * len(words)
    seen = set()
    for tid, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        pred_word[wid] = model.config.id2label[int(pred_ids[tid])]
    return words, pred_word

def insert_tags(words, labs):
    out = []
    i = 0
    while i < len(words):
        lab = labs[i]
        out.append(words[i])

        if lab.startswith("B-"):
            tag = lab[2:]
            j = i + 1
            while j < len(words) and labs[j] == f"I-{tag}":
                out.append(words[j])
                j += 1
            out.append(f"<<{tag}>>")
            i = j
        else:
            i += 1

    return " ".join(out)

def annotate(sentence: str):
    w, l = predict_word_labels(sentence)
    return insert_tags(w, l)


