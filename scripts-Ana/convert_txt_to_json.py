import re
import json

def extract_annotations(text, label_type):
    pattern = re.compile(r'(.*?)(<(' + label_type + r'.*?)>)(.*?)', re.DOTALL)
    entries = []

    for match in re.finditer(pattern, text):
        full_text = match.group(0).replace('\n', ' ').strip()
        annotation = match.group(3)
        # Marker is the word immediately before the annotation, if available
        pre_match_text = match.group(1).strip()
        marker = ""
        if pre_match_text:
            words = pre_match_text.split()
            marker = words[-1] if words else ""
        entries.append({
            "text": full_text.replace(f"<{annotation}>", "").strip(),
            "marker": marker,
            "label": f"<{annotation}>"
        })
    return entries

def main():
    # Cargar texto del archivo
    with open("input.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # Extraer anotaciones
    ef_entries = extract_annotations(text, "EF")
    ep_entries = extract_annotations(text, "EP")

    # Guardar JSON
    with open("output_ef.json", "w", encoding="utf-8") as f:
        json.dump(ef_entries, f, indent=2, ensure_ascii=False)

    with open("output_ep.json", "w", encoding="utf-8") as f:
        json.dump(ep_entries, f, indent=2, ensure_ascii=False)

    print("Conversión completada. Archivos generados: output_ef.json y output_ep.json")

if __name__ == "__main__":
    main()
