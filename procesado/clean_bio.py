import json, re

DATA_IN  = "data_ready_emo.json"
DATA_OUT = "data_ready_emo.cleaned.json"

TAG_RE = re.compile(r"<<(EP_[A-Z]+(?:_[A-Z]+)?)>>")
BOM = "\ufeff"

def clean_basic(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace(BOM, "")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def detach_tags(text: str) -> str:
    """
    Garantiza que <<EP_...>> quede separado por espacios,
    incluso si está pegado a puntuación: <<...>>)  o  .<<...>>
    """
    if not text:
        return text

    #  Espacios alrededor del tag (preserva el tag exacto)
    text = TAG_RE.sub(r" <<\1>> ", text)

    #  Normaliza espacios
    text = re.sub(r"\s+", " ", text).strip()
    return text

with open(DATA_IN, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []
for ex in data:
    inp = clean_basic(ex.get("input_text", ""))
    out = clean_basic(ex.get("output_text", ""))

    # Limpieza clave
    out = detach_tags(out)

    if inp and out:
        cleaned.append({"input_text": inp, "output_text": out})

with open(DATA_OUT, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print("OK. Guardado:", DATA_OUT, "Ejemplos:", len(cleaned))
