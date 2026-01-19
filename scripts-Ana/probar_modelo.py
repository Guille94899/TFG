from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# Lista de etiquetas en orden
label_list = [
    "EP", "EF", "P",
    "FV", "CGA", "EM", "IIE", "IIR", "APH", "MEP",
    "DIR", "DM", "INT", "INC", "NRM", "POT"
]

# Cargar modelo y tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list),
    problem_type="multi_label_classification"
)
model.eval()

# Texto de ejemplo (puedes cambiarlo por cualquier otro)
text = "I think we are looking for more clarity from the Government about their expectations of local authorities."

# Tokenización
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

# Mostrar etiquetas con probabilidad mayor a 0.5
threshold = 0.5
print("\nEtiquetas predichas (por encima del umbral de 0.5):")
for label, prob in zip(label_list, probs):
    if prob > threshold:
        print(f"{label}: {prob:.2f}")
