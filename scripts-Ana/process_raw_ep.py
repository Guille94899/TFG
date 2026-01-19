import os
import json
import re

VALID_LABELS = {
    "EP_FV", "EP_FV_P", "EP_CGA", "EP_CGA_P", "EP_EM", 
    "EP_IIE", "EP_IIE_P", "EP_IIR", "EP_APH", "EP_APH_P", "EP_MEP_P"
}

def normalize_label(raw_label):
    if not raw_label.startswith("<") or not raw_label.endswith(">"):
        return None
    label = raw_label.strip("<>").replace(", ", "_")
    return label if label in VALID_LABELS else None

def process_json_file(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    cleaned = []
    for entry in data:
        normalized_label = normalize_label(entry.get("label", ""))
        if normalized_label:
            cleaned.append({
                "text": entry["text"],
                "marker": entry["marker"],
                "label": normalized_label
            })

    with open(output_path, 'w') as f:
        json.dump(cleaned, f, indent=2)

# Ejemplo: procesar todos los JSON en un directorio
input_dir = "./raw_ep"
output_dir = "./ready_ep"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".json"):
        process_json_file(
            os.path.join(input_dir, fname),
            os.path.join(output_dir, fname)
        )
