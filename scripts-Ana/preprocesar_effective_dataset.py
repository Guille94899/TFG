import json

# Archivo de entrada y salida
input_file = "effective_stance_dataset_first.json"
output_file = "effective_preprocesado.json"

def convertir_etiqueta(etiqueta_raw):
    etiqueta_raw = etiqueta_raw.replace("<", "").replace(">", "").replace(" ", "")
    partes = etiqueta_raw.split(",")
    if "EF" not in partes:
        return None
    sub = None
    for tipo in ["DIR", "DM", "INT", "INC", "NRM", "POT"]:
        if tipo in partes:
            sub = tipo
            break
    if sub is None:
        return None
    return f"EF_{sub}_P" if "P" in partes else f"EF_{sub}"

preprocesados = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        texto = entry["text"]
        marcador = entry["marker"]
        etiqueta = convertir_etiqueta(entry["label"])

        if etiqueta is None:
            continue

        input_text = f"{texto} [MARKER] {marcador}"
        preprocesados.append({
            "input": input_text,
            "label": etiqueta
        })

# Guardar como JSONL
with open(output_file, "w", encoding="utf-8") as out:
    for item in preprocesados:
        json.dump(item, out)
        out.write("\n")

print(f"{len(preprocesados)} ejemplos guardados en {output_file}")
