import json
from config import MAX_HISTORY

def format_history(history):
    if not history:
        return ""

    lines = []
    for msg in history[-MAX_HISTORY:]:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        lines.append(f"{role}: {msg['content']}")

    return "HISTORIAL:\n" + "\n".join(lines) + "\n\n"


def parse_json_response(text: str):
    try:
        data = json.loads(text)

        if not isinstance(data, dict):
            raise ValueError("No es un dict")

        if "answer" not in data:
            raise ValueError("Falta 'answer'")

        # Eliminar confidence si viene en la respuesta (campo obsoleto)
        data.pop("confidence", None)

        # sources
        sources = data.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        data["sources"] = sources

        return data

    except Exception as e:
        print(f"[JSON PARSE ERROR] {e}\nRaw response:\n{text}")
        return None
