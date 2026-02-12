import json
import re
from typing import List, Dict, Optional

def strip_code_fences(s: str) -> str:
    # Remove surrounding ```json ... ``` or ``` ... ``` fences.
    s = s.strip()
    # If it starts with ```, remove the first and last fences.
    if s.startswith("```"):
        # Remove the first line of ``` or ```json.
        s = re.sub(r"^```json\n", "", s, count=1)
        s = re.sub(r"^```\n", "", s, count=1)
        # Remove trailing ```.
        s = re.sub(r"\n```$", "", s, count=1)
        s = re.sub(r"```$", "", s, count=1)
    return s.strip()

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract a JSON object from text that may include reasoning.
    Prefer a dict that contains 'defect_lines'.
    """
    # Strategy 1: Find all possible JSON objects and prefer those containing "defect_lines".
    # Use a stack to match braces.
    candidates = []
    stack = []
    start_pos = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_pos = i
            stack.append(i)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_pos != -1:
                    # Found a complete JSON object.
                    json_str = text[start_pos:i+1]
                    # Check if it contains "defect_lines" or looks like the target format.
                    if '"defect_lines"' in json_str or ('"path"' in json_str and '"line"' in json_str):
                        try:
                            parsed = json.loads(json_str)
                            # Validate that it matches the target format.
                            if isinstance(parsed, dict):
                                # Prefer ones containing defect_lines.
                                priority = 2 if 'defect_lines' in parsed else (1 if ('path' in parsed and 'line' in parsed) else 0)
                                if priority > 0:
                                    candidates.append((json_str, len(json_str), priority))
                        except json.JSONDecodeError:
                            pass
                    start_pos = -1
    
    # Prefer ones containing defect_lines, then choose the longest.
    if candidates:
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return candidates[0][0]
    
    # Strategy 2: Try line-by-line parsing to find JSON lines.
    for line in text.splitlines():
        line = line.strip()
        if not line or not (line.startswith('{') and line.endswith('}')):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and ('defect_lines' in obj or ('path' in obj and 'line' in obj)):
                return line
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Use regex to find a JSON fragment containing "defect_lines", then expand.
    # Find the position of "defect_lines".
    defect_lines_pos = text.find('"defect_lines"')
    if defect_lines_pos != -1:
        # Search backward for the nearest {
        json_start = text.rfind('{', 0, defect_lines_pos)
        if json_start != -1:
            # Starting from json_start, match braces to find the full JSON.
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end != -1:
                json_str = text[json_start:json_end]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        return json_str
                except json.JSONDecodeError:
                    pass
    
    return None

def parse_llm_content(content: str) -> List[Dict]:
    """
    Parse LLM output into a list of objects:
    - If it's a single JSON object, return [obj]
    - If it's a JSON array, return the array
    - If it's NDJSON (multiple lines, one object per line), return a merged list
    Validate keys 'path' and 'line'

    Supports extracting JSON dicts from text that includes reasoning.
    """
    text = strip_code_fences(content)

    def validate_and_fix(obj: Dict) -> Dict:
        if not isinstance(obj, dict):
            raise ValueError("Parsed item is not an object")
        if "path" not in obj or "line" not in obj:
            raise ValueError("Missing required keys 'path' and/or 'line'")
        # Normalize types: path is string, line is integer.
        obj["path"] = str(obj["path"])
        # Allow line as a numeric string and convert once.
        if isinstance(obj["line"], str) and obj["line"].isdigit():
            obj["line"] = int(obj["line"])
        if not isinstance(obj["line"], int):
            raise ValueError("Key 'line' must be an integer")
        return obj

    # Strategy 1: Prefer parsing the whole JSON.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and 'defect_lines' in parsed:
            defect_lines = parsed['defect_lines']
            if isinstance(defect_lines, dict):
                return [validate_and_fix(defect_lines)]
            elif isinstance(defect_lines, list):
                return [validate_and_fix(x) for x in defect_lines]
        elif isinstance(parsed, dict) and 'path' in parsed and 'line' in parsed:
            # Already in the target format.
            return [validate_and_fix(parsed)]
        elif isinstance(parsed, list):
            # If it's a list, try to parse each element.
            results = []
            for item in parsed:
                if isinstance(item, dict):
                    if 'defect_lines' in item:
                        defect_lines = item['defect_lines']
                        if isinstance(defect_lines, list):
                            results.extend([validate_and_fix(x) for x in defect_lines])
                        elif isinstance(defect_lines, dict):
                            results.append(validate_and_fix(defect_lines))
                    elif 'path' in item and 'line' in item:
                        results.append(validate_and_fix(item))
            if results:
                return results
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract a JSON object from text.
    extracted_json = extract_json_from_text(text)
    if extracted_json:
        try:
            parsed = json.loads(extracted_json)
            if isinstance(parsed, dict) and 'defect_lines' in parsed:
                defect_lines = parsed['defect_lines']
                if isinstance(defect_lines, dict):
                    return [validate_and_fix(defect_lines)]
                elif isinstance(defect_lines, list):
                    return [validate_and_fix(x) for x in defect_lines]
            elif isinstance(parsed, dict) and 'path' in parsed and 'line' in parsed:
                return [validate_and_fix(parsed)]
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try NDJSON by parsing line by line.
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                if 'defect_lines' in obj:
                    defect_lines = obj['defect_lines']
                    if isinstance(defect_lines, list):
                        results.extend([validate_and_fix(x) for x in defect_lines])
                    elif isinstance(defect_lines, dict):
                        results.append(validate_and_fix(defect_lines))
                elif 'path' in obj and 'line' in obj:
                    results.append(validate_and_fix(obj))
        except json.JSONDecodeError:
            continue

    if not results:
        raise ValueError("Content is not valid JSON/NDJSON and could not extract JSON from text")
    return results

def save_as_json_array(items: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def save_as_ndjson(items: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")