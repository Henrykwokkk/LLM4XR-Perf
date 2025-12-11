import json
import re
from typing import List, Dict

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s, count=1)
        s = re.sub(r"\n```$", "", s, count=1)
    return s.strip()

def parse_llm_content(content: str) -> List[Dict]:
    text = strip_code_fences(content)

    def validate_and_fix(obj: Dict) -> Dict:
        if not isinstance(obj, dict):
            raise ValueError("Parsed item is not an object")
        if "path" not in obj or "line" not in obj:
            raise ValueError("Missing required keys 'path' and/or 'line'")
        obj["path"] = str(obj["path"])
        if isinstance(obj["line"], str) and obj["line"].isdigit():
            obj["line"] = int(obj["line"])
        if not isinstance(obj["line"], int):
            raise ValueError("Key 'line' must be an integer")
        return obj

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [validate_and_fix(parsed)]
        elif isinstance(parsed, list):
            return [validate_and_fix(x) for x in parsed]
    except json.JSONDecodeError:
        pass

    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            results.append(validate_and_fix(obj))
        except json.JSONDecodeError:
            continue

    if not results:
        raise ValueError("Content is not valid JSON/NDJSON")
    return results

def save_as_json_array(items: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def save_as_ndjson(items: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")