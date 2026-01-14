import os
import re
import json
import nltk
from nltk.tokenize import sent_tokenize

TXT_FOLDER = "txts"
DATA_EP = "data_ep.json"
DATA_EF = "data_ef.json"
DATA_EMO = "data_emo.json"

TAG_EP_PATTERN = re.compile(r"<EP\s*,\s*[^>]+>")
TAG_EF_PATTERN = re.compile(r"<EF\s*,\s*[^>]+>")
TAG_EMO_PATTERN = re.compile(r"<EMO\s*,\s*[^>]+>")

REMOVE_CURLY_PATTERN = re.compile(r"\{[^{}]+\}")
REMOVE_TAGS = re.compile(r"<[^>]+>")

def ensure_sentence_end(s: str) -> str:
    s = s.rstrip()
    if not s:
        return s
    if s.endswith(('.', '!', '?')):
        return s
    if s.endswith((';', ':', ',')) or s.endswith('…'):
        return s[:-1].rstrip() + '.'
    if s[-1] in ')]}"\'':
        if len(s) > 1 and s[-2] in '.!?':
            return s
        if len(s) > 1 and s[-2] in ';:,':  # ; : ,
            return s[:-2].rstrip() + s[-1]
        return s[:-1].rstrip() + '.' + s[-1]
    return s + '.'

def clean_text(text):
    text = REMOVE_CURLY_PATTERN.sub("", text)
    text = REMOVE_TAGS.sub("", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    sentences = sent_tokenize(content)

    ep_items = []
    ef_items = []
    emo_items = []

    for sentence in sentences:
        has_ep = bool(TAG_EP_PATTERN.search(sentence))
        has_ef = bool(TAG_EF_PATTERN.search(sentence))
        has_emo = bool(TAG_EMO_PATTERN.search(sentence))

        if not (has_ep or has_ef or has_emo):
            continue

        # Texto con etiquetas (para output del T5)
        output_full = ensure_sentence_end(sentence.strip())
        # Texto limpio sin etiquetas (input del T5)
        input_full = ensure_sentence_end(clean_text(output_full))

        item = {
            "input_text": input_full,
            "output_text": output_full
        }

        if has_ep:
            ep_items.append(item)
        if has_ef:
            ef_items.append(item)
        if has_emo:
            emo_items.append(item)

    return ep_items, ef_items, emo_items

def main():
    nltk.download("punkt", quiet=True)

    all_ep = []
    all_ef = []
    all_emo = []

    for filename in os.listdir(TXT_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(TXT_FOLDER, filename)
            print(f"Procesando: {filepath}")
            ep, ef, emo = process_text_file(filepath)
            all_ep.extend(ep)
            all_ef.extend(ef)
            all_emo.extend(emo)

    with open(DATA_EP, "w", encoding="utf-8") as f:
        json.dump(all_ep, f, indent=4, ensure_ascii=False)

    with open(DATA_EF, "w", encoding="utf-8") as f:
        json.dump(all_ef, f, indent=4, ensure_ascii=False)

    with open(DATA_EMO, "w", encoding="utf-8") as f:
        json.dump(all_emo, f, indent=4, ensure_ascii=False)

    print(f"\nGuardado {len(all_ep)} items en {DATA_EP}")
    print(f"Guardado {len(all_ef)} items en {DATA_EF}")
    print(f"Guardado {len(all_emo)} items en {DATA_EMO}")

if __name__ == "__main__":
    main()
