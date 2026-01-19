from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import torch
import json
import os

# 1. Cargar todos los JSON procesados
def load_all_cleaned_jsons(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file)) as f:
                data.extend(json.load(f))
    return data

raw_data = load_all_cleaned_jsons("./ready_ep")

# 2. Codificar etiquetas
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform([item["label"] for item in raw_data])

# 3. Crear dataset Hugging Face
dataset = Dataset.from_dict({
    "text": [item["text"] for item in raw_data],
    "label": labels
})

# 4. Tokenizar
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize)

# 5. Separar en train/test
split = tokenized_dataset.train_test_split(test_size=0.2)

# 6. Cargar modelo
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

# 7. Entrenamiento

training_args = TrainingArguments(
    output_dir="./bert-epistemic-stance",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    logging_dir="./logs_ep",
    logging_steps=10,
    evaluation_strategy="epoch", 
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    tokenizer=tokenizer,
)

trainer.train()
