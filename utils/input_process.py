import tiktoken
import re
from typing import List, Tuple, Dict, Optional


CONTEXT_WINDOWS = {
    "qwen3-235b-a22b": 128000,
    "gemini-2.5-pro": 896000,
    "gemini-2.5-pro-preview": 1000000,
    "gpt-5": 128000,
}


def get_encoder_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("o200k_base")

def safe_encode_len(enc, text: str, chunk_chars: int = 200000) -> int:
    total = 0
    for i_ in range(0, len(text), chunk_chars):
        piece = text[i_:i_+chunk_chars]
        try:
            total += len(enc.encode(piece))
        except Exception:
            total += int(len(piece) / 3.2)
    return total

def count_message_tokens(messages, model_name: str):
    enc = get_encoder_for_model(model_name)
    total = 0
    for m in messages:
        content = m.get("content", "") or ""
        total += safe_encode_len(enc, content)
    return total

def reduce_text_to_fit_context(messages, model_name: str, max_tokens: int, buffer_tokens: int = 20) -> list:
    while count_message_tokens(messages, model_name) > (max_tokens - buffer_tokens):
        user_message = messages[-1]['content']
        patch_start = user_message.find('**Patch:**')
        patch_end = user_message.find('Category:')
        patch_text = user_message[patch_start:patch_end]
        new_patch_text = ' '.join(patch_text.split(' ')[:-(len(patch_text.split(' '))//20)])  
        new_user_message = user_message[:patch_start] + new_patch_text + '\nCategory:'
        messages[-1]['content'] = new_user_message

    return messages


def locate_code_files_region(prompt: str) -> Tuple[int, int]:
    s = prompt.find('<code_files>')
    if s == -1:
        raise ValueError("No <code_files> tag found.")
    e = prompt.find('</code_files>', s)
    if e == -1:
        raise ValueError("No </code_files> closing tag found.")
    return s + len('<code_files>'), e  # inner region [inner_start:inner_end]

def find_file_blocks_in_inner(inner_text: str) -> List[Dict]:

    START_RE = re.compile(r"\[start of ([^\]]+)\]")
    blocks = []
    for m in START_RE.finditer(inner_text):
        file_path = m.group(1)
        start_idx = m.start()
        end_tag = f"[end of {file_path}]"
        end_rel = inner_text.find(end_tag, m.end())
        if end_rel == -1:
            continue
        end_idx = end_rel + len(end_tag)
        blocks.append({"file_path": file_path, "start_idx": start_idx, "end_idx": end_idx})
    return blocks


def reduce_code_text_to_fit_context(messages, model_name: str, max_tokens: int, buffer_tokens: int = 10000) -> list:
    while count_message_tokens(messages, model_name) > (max_tokens - buffer_tokens):
        user_message = messages[-1]['content']
        try:
            code_start, code_end = locate_code_files_region(user_message)
        except ValueError:
            break
        code_text = user_message[code_start:code_end]
        new_code_text = ' '.join(code_text.split(' ')[:-(len(code_text.split(' '))//20)])
        new_user_message = user_message[:code_start] + new_code_text + user_message[code_end:]
        messages[-1]['content'] = new_user_message

    return messages