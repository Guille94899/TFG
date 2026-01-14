import os
import json
import re
import random
import argparse
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize


# Limpiar texto (quita etiquetas <EP, ...>, <EF, ...>, <EMO,...>)

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Extraer etiquetas normalizadas

def extract_labels(sentence: str, prefix: str):
    pattern = fr"<\s*{prefix}\s*,\s*([^>]+)>"
    matches = re.findall(pattern, sentence)

    labels = []
    for m in matches:
        m = m.strip().replace("-", "_")

        m = re.sub(r"^([A-Za-z]{2,3})\s+P$", r"\1, P", m)
        m = re.sub(r",\s*$", "", m)

        parts = [p.strip().upper() for p in re.split(r"\s*,\s*", m) if p.strip()]
        if not parts:
            continue

        tipo = parts[0]

        if prefix in {"EP", "EF"} and len(parts) >= 2 and parts[1] == "P":
            labels.append(f"{prefix}_{tipo}_P")
            continue

        POLS = {"POS", "NEG", "NEU", "MIX"}

        if len(parts) >= 3 and parts[2] in POLS:
            labels.append(f"{prefix}_{tipo}_{parts[2]}")
        elif len(parts) >= 2 and parts[1] in POLS:
            labels.append(f"{prefix}_{tipo}_{parts[1]}")
        else:
            labels.append(f"{prefix}_{tipo}")

    return labels


# SUBMUESTREO DE NEGATIVAS

def balance_dataset(entries, none_label, max_neg_ratio=1.5, seed=42):
    random.seed(seed)

    positives = [e for e in entries if none_label not in e["labels"]]
    negatives = [e for e in entries if none_label in e["labels"]]

    max_negs = int(len(positives) * max_neg_ratio)
    if len(negatives) > max_negs:
        negatives = random.sample(negatives, max_negs)

    balanced = positives + negatives
    random.shuffle(balanced)
    return balanced



# CONCATENAR TODOS LOS .txt DE UNA CARPETA

def concat_txt_folder(folder_path, output_path):
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".txt"))
    if not files:
        raise ValueError(f"No se han encontrado .txt en la carpeta: {folder_path}")

    with open(output_path, "w", encoding="utf-8") as outfile:
        for fname in files:
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as infile:
                contenido = infile.read().strip()
                if contenido:
                    outfile.write(contenido + "\n\n")

    print(f"✔ input concatenado generado en: {output_path}")
    return output_path



# MAIN

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, default="txt")
    parser.add_argument("--output_ep", type=str, default="output_ep.json")
    parser.add_argument("--output_ef", type=str, default="output_ef.json")
    parser.add_argument("--output_emo", type=str, default="output_emo.json")
    parser.add_argument("--train_ep", type=str, default="train_ep.json")
    parser.add_argument("--train_ef", type=str, default="train_ef.json")
    parser.add_argument("--train_emo", type=str, default="train_emo.json")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_limit", type=int, default=20)

    args = parser.parse_args()

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, args.folder)

    temp_input = os.path.join(base_dir, "_input_concatenado.txt")
    input_path = concat_txt_folder(folder_path, temp_input)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = sent_tokenize(text)

    ep_entries, ef_entries, emo_entries = [], [], []
    train_ep_entries, train_ef_entries, train_emo_entries = [], [], []

    unmatched_emo_counter = Counter()
    unmatched_examples = []

    for sent in sentences:
        sent_clean = clean_text(sent)

        ep_labels = extract_labels(sent, "EP")
        ef_labels = extract_labels(sent, "EF")
        emo_labels = extract_labels(sent, "EMO")

        if ep_labels:
            ep_entries.append({"text": sent_clean, "labels": ep_labels})
        if ef_labels:
            ef_entries.append({"text": sent_clean, "labels": ef_labels})
        if emo_labels:
            emo_entries.append({"text": sent_clean, "labels": emo_labels})

        ep_or_none = ep_labels if ep_labels else ["NO_EP"]
        ef_or_none = ef_labels if ef_labels else ["NO_EF"]
        emo_or_none = emo_labels if emo_labels else ["NO_EMO"]

        train_ep_entries.append({"text": sent_clean, "labels": ep_or_none})
        train_ef_entries.append({"text": sent_clean, "labels": ef_or_none})
        train_emo_entries.append({"text": sent_clean, "labels": emo_or_none})

        if args.debug:
            raw_emo_tags = re.findall(r"<\s*EMO\s*,\s*[^>]+>", sent)
            if raw_emo_tags and not emo_labels:
                for t in raw_emo_tags:
                    unmatched_emo_counter[t] += 1
                if len(unmatched_examples) < args.debug_limit:
                    unmatched_examples.append(sent.strip())

    # Balanceo FINAL (aquí es donde decides cuántos NO_* entran de verdad)
    balanced_train_ep = balance_dataset(train_ep_entries, "NO_EP")
    balanced_train_ef = balance_dataset(train_ef_entries, "NO_EF")
    balanced_train_emo = balance_dataset(train_emo_entries, "NO_EMO")

    # Conteos SOLO sobre el dataset balanceado (esto corrige NO_EP/NO_EF/NO_EMO)
    ep_label_counts_train = Counter(lab for e in balanced_train_ep for lab in e["labels"])
    ef_label_counts_train = Counter(lab for e in balanced_train_ef for lab in e["labels"])
    emo_label_counts_train = Counter(lab for e in balanced_train_emo for lab in e["labels"])

    # Guardar datasets
    for path, data in [
        (args.output_ep, ep_entries),
        (args.output_ef, ef_entries),
        (args.output_emo, emo_entries),
        (args.train_ep, balanced_train_ep),
        (args.train_ef, balanced_train_ef),
        (args.train_emo, balanced_train_emo),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Guardar conteos TRAIN (balanceados)
    with open("label_counts_ep_train.json", "w", encoding="utf-8") as f:
        json.dump(ep_label_counts_train, f, indent=2, ensure_ascii=False)
    with open("label_counts_ef_train.json", "w", encoding="utf-8") as f:
        json.dump(ef_label_counts_train, f, indent=2, ensure_ascii=False)
    with open("label_counts_emo_train.json", "w", encoding="utf-8") as f:
        json.dump(emo_label_counts_train, f, indent=2, ensure_ascii=False)
        
    print("✔ Dataset generado correctamente.")
    print(f"- EP puras: {len(ep_entries)}")
    print(f"- EF puras: {len(ef_entries)}")
    print(f"- EMO puras: {len(emo_entries)}")

    print(f"- NO_EP en train final: {ep_label_counts_train.get('NO_EP', 0)}")
    print(f"- NO_EF en train final: {ef_label_counts_train.get('NO_EF', 0)}")
    print(f"- NO_EMO en train final: {emo_label_counts_train.get('NO_EMO', 0)}")

    print("Conteos TRAIN balanceados guardados en label_counts_*_train.json")

    if args.debug and unmatched_emo_counter:
        print("\n[DEBUG EMO] Tags EMO problemáticos:")
        for tag, c in unmatched_emo_counter.most_common(20):
            print(f"  {c}x {tag}")


if __name__ == "__main__":
    main()
