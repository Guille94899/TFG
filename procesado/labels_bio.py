import re
import json




MAIN_LABEL = "EMO"   

INPUT_FILE = f"data_{MAIN_LABEL.lower()}.json"
OUTPUT_FILE = f"data_ready_{MAIN_LABEL.lower()}.json"

OTHER_LABEL = "EF" if MAIN_LABEL == "EP" else "EP"



main_tag_pattern = re.compile(
    rf"<\s*{MAIN_LABEL}\s*,\s*([^,>\s]+)\s*(?:,\s*([^>\s]+))?\s*>"
)

other_tag_pattern = re.compile(
    rf"<\s*{OTHER_LABEL}\s*,\s*[^>]+>"
)


# FUNCIONES

def replace_main_tags(text: str) -> str:
    def repl(match):
        part1 = match.group(1).strip()
        part2 = match.group(2).strip() if match.group(2) else None
        return f"{MAIN_LABEL}_{part1}_{part2}" if part2 else f"{MAIN_LABEL}_{part1}"
    return main_tag_pattern.sub(repl, text)

def remove_other_tags(text: str) -> str:
    return other_tag_pattern.sub("", text)



def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_data = []

    for item in data:
        text = item.get("output_text", "")

        if main_tag_pattern.search(text) or other_tag_pattern.search(text):
            # Reemplaza etiquetas del tipo principal (EP o EF)
            new_text = replace_main_tags(text)
            # Elimina etiquetas del otro tipo
            new_text = remove_other_tags(new_text)
            # Limpieza final
            new_text = re.sub(r"\s+", " ", new_text).strip()

            item["output_text"] = new_text
            updated_data.append(item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=4, ensure_ascii=False)

    print(
        f"Archivo guardado como {OUTPUT_FILE} "
        f"con {len(updated_data)} ítems procesados para {MAIN_LABEL}."
    )

if __name__ == "__main__":
    main()
