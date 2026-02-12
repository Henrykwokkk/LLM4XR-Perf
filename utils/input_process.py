import tiktoken
import re
from typing import List, Tuple, Dict, Optional
from loguru import logger

# Gemini 3 Pro, Deepseek-v3.2, Claude-Sonnet-4.5, Gpt-5.2
CONTEXT_WINDOWS = {
    "qwen3-235b-a22b": 120000,
    "gemini-2.5-pro": 896000,
    "gemini-2.5-pro-preview": 1000000,
    "gemini-3-pro-preview": 1000000,
    "deepseek-v3.2": 128000,
    "claude-sonnet-4-5-20250929": 200000,
    "gpt-5.2": 400000,
    "gpt-5": 400000,
    "kimi-k2-250905": 256000,
    "deepseek-v3.1": 128000,
    "claude-sonnet-4-20250514": 200000,
    "gpt-4o": 128000,
}

# Token count correction factors for models where tiktoken doesn't accurately match the API tokenizer
# Based on observed discrepancies: API token count / estimated token count
TOKEN_COUNT_MULTIPLIERS = {
    "claude-sonnet-4-20250514": 1.24,  # Observed: 200,379 / 161,997 ≈ 1.236
}


def get_encoder_for_model(model_name: str):
    try:
        # Automatically select a suitable tokenizer based on the model name (if mapped in tiktoken)
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        # Fallback: a safe approximation for common OpenAI chat models
        return tiktoken.get_encoding("o200k_base")

def safe_encode_len(enc, text: str, chunk_chars: int = 200000) -> int:
    # Split by characters to avoid stack overflow when processing very long strings at once
    total = 0
    for i_ in range(0, len(text), chunk_chars):
        piece = text[i_:i_+chunk_chars]
        try:
            total += len(enc.encode(piece))
        except Exception:
            # Fallback estimate: English averages about 3–4 characters per token; use 3.2 here
            total += int(len(piece) / 3.2)
    return total

def count_message_tokens(messages, model_name: str):
    enc = get_encoder_for_model(model_name)
    # overhead_per_msg = 4  # Simple approximation
    total = 0
    for m in messages:
        content = m.get("content", "") or ""
        total += safe_encode_len(enc, content)
        # total += len(enc.encode(m.get("content", "")))
    
    # Apply correction factor for models where tiktoken doesn't accurately match API tokenizer
    multiplier = TOKEN_COUNT_MULTIPLIERS.get(model_name, 1.0)
    return int(total * multiplier)

def reduce_text_to_fit_context(messages, model_name: str, max_tokens: int, buffer_tokens: int = 20) -> list:
    while count_message_tokens(messages, model_name) > (max_tokens - buffer_tokens):
        # Remove the **patch** part in the last message of the prompt
        # print(count_message_tokens(messages, model_name), '>', (max_tokens - buffer_tokens), 'start trimming patch content')
        user_message = messages[-1]['content']
        patch_start = user_message.find('**Patch:**')
        patch_end = user_message.find('Category:')
        patch_text = user_message[patch_start:patch_end]
        # Truncate the patch until all message tokens are less than max_tokens - buffer_tokens
        new_patch_text = ' '.join(patch_text.split(' ')[:-(len(patch_text.split(' '))//20)])  # Remove 5% of patch content each time
        new_user_message = user_message[:patch_start] + new_patch_text + '\nCategory:'
        messages[-1]['content'] = new_user_message

    return messages


def locate_code_files_region(prompt: str) -> Tuple[int, int]:
    """Locate the unique <code_files> ... </code_files> region and return (inner_start, inner_end), excluding outer tags."""
    s = prompt.find('<code_files>')
    if s == -1:
        raise ValueError("No <code_files> tag found.")
    
    inner_start = s + len('<code_files>')
    # Search from the end to find the last </code_files>, ensuring the correct closing tag is found
    e = prompt.rfind('</code_files>')
    
    # Verify that the found position is valid (should be after the start tag)
    if e == -1 or e <= inner_start:
        # If not found from the end, try searching forward from the start position
        e = prompt.find('</code_files>', inner_start)
        if e == -1:
            raise ValueError("No </code_files> closing tag found.")
    
    return inner_start, e  # inner region [inner_start:inner_end]

def find_file_blocks_in_inner(inner_text: str) -> List[Dict]:
    """
    Find file blocks inside <code_files>:
    [start of file_path]
    ...content...
    [end of file_path]
    Return a list of blocks in order of appearance, with indices relative to inner_text.
    """
    START_RE = re.compile(r"\[start of ([^\]]+)\]")
    blocks = []
    for m in START_RE.finditer(inner_text):
        file_path = m.group(1)
        start_idx = m.start()
        end_tag = f"[end of {file_path}]"
        end_rel = inner_text.find(end_tag, m.end())
        if end_rel == -1:
            # Unpaired; skip this block
            continue
        end_idx = end_rel + len(end_tag)
        blocks.append({"file_path": file_path, "start_idx": start_idx, "end_idx": end_idx})
    # Already in order of start index
    return blocks


def reduce_code_text_to_fit_context(messages, model_name: str, max_tokens: int, buffer_tokens: int = 10000, target_reduction_ratio: float = None) -> list:
    """
    Reduce code text within messages to fit the context window.
    
    Args:
        messages: List of messages.
        model_name: Model name.
        max_tokens: Maximum number of tokens (context window).
        buffer_tokens: Number of buffer tokens.
        target_reduction_ratio: Target reduction ratio (0-1). If provided,
            reduce directly to this ratio; otherwise compute automatically.
    
    Returns:
        The reduced list of messages.
    """
    # Initial checks and setup
    target_tokens = max_tokens - buffer_tokens
    initial_tokens = count_message_tokens(messages, model_name)
    if initial_tokens <= target_tokens:
        return messages
    
    # If a target reduction ratio is provided, use it directly
    if target_reduction_ratio is not None:
        target_tokens = int(initial_tokens * target_reduction_ratio)
        logger.info(f"Using provided reduction ratio {target_reduction_ratio:.0%}: "
                   f"{initial_tokens:,} -> {target_tokens:,} tokens")
    else:
        excess_ratio = (initial_tokens - target_tokens) / initial_tokens
        logger.info(f"Starting context reduction: {initial_tokens:,} -> {target_tokens:,} tokens (excess: {excess_ratio:.1%})")
    
    enc = get_encoder_for_model(model_name)
    max_iterations = 20  # Reduce number of iterations to improve efficiency
    
    # Helper function: compute target code length
    def calculate_target_length(code_text: str, code_tokens: int, current_tokens: int, 
                                target_tokens: int, iteration: int) -> int:
        """Compute target code length."""
        # If a target reduction ratio is provided and this is the first iteration,
        # use that ratio directly. target_reduction_ratio is passed from
        # inference.py and already accounts for the reduction strategy.
        # No extra safety factor is needed because inference.py has computed
        # the precise ratio.
        if target_reduction_ratio is not None and iteration == 0:
            return int(len(code_text) * target_reduction_ratio)
        
        # For later iterations or when target_reduction_ratio is not provided:
        # compute precisely based on the target token count.
        # Calculate how many tokens of the code portion should be kept so
        # that the whole message reaches the target token count.
        total_other_tokens = current_tokens - code_tokens
        needed_code_tokens = max(0, target_tokens - total_other_tokens)
        
        if needed_code_tokens > 0 and code_tokens > 0:
            # Directly use the computed token ratio without extra safety factor.
            # To reach the target precisely, use the exact calculation rather
            # than an additional safety margin.
            code_keep_ratio = needed_code_tokens / code_tokens
            return int(len(code_text) * code_keep_ratio)
        else:
            # Fallback: use character ratio (only when the calculation fails)
            char_ratio = (target_tokens / current_tokens) if current_tokens > 0 else 0.5
            return int(len(code_text) * char_ratio)
    
    # Helper function: truncate code text
    def truncate_code_text(code_text: str, target_length: int) -> str:
        """Truncate code text to the target length, preferably at a newline."""
        if target_length >= len(code_text):
            return code_text
        
        new_text = code_text[:target_length]
        # Try truncating at a newline if it is reasonably close
        last_newline = new_text.rfind('\n')
        if last_newline > target_length * 0.85:
            new_text = new_text[:last_newline]
        
        return new_text
    
    # Main loop: iterative reduction
    for iteration in range(max_iterations):
        current_tokens = count_message_tokens(messages, model_name)
        if current_tokens <= target_tokens:
            if iteration > 0:
                logger.info(f"Context reduction completed after {iteration} iterations. Final tokens: {current_tokens:,}")
            break
        
        # Re-fetch code_text (it may have been modified)
        user_message = messages[-1]['content']
        try:
            code_start, code_end = locate_code_files_region(user_message)
        except ValueError:
            logger.warning("No <code_files> tag found, cannot reduce further")
            break
        
        code_text = user_message[code_start:code_end]
        code_tokens = safe_encode_len(enc, code_text)
        
        # Log info on first iteration
        if iteration == 0:
            logger.info(f"Located code_files region: length={len(code_text):,} chars, tokens={code_tokens:,}")
        
        # Compute target length
        target_code_length = calculate_target_length(code_text, code_tokens, current_tokens, target_tokens, iteration)
        
        if target_code_length <= 0 or target_code_length >= len(code_text):
            logger.warning(f"Cannot reduce further: invalid target length {target_code_length} (code_text: {len(code_text)})")
            break
        
        # Perform truncation
        new_code_text = truncate_code_text(code_text, target_code_length)
        new_user_message = user_message[:code_start] + new_code_text + user_message[code_end:]
        messages[-1]['content'] = new_user_message
        
        # Log progress (every 5 iterations or during the first 3)
        if iteration % 5 == 0 or iteration < 3:
            new_tokens = count_message_tokens(messages, model_name)
            reduction = current_tokens - new_tokens
            logger.info(f"Iteration {iteration + 1}: {current_tokens:,} -> {new_tokens:,} tokens "
                       f"(-{reduction:,}, {reduction/current_tokens*100:.1f}% reduction)")
    
    final_tokens = count_message_tokens(messages, model_name)
    if final_tokens > target_tokens:
        logger.warning(f"Context reduction reached max iterations. Final tokens: {final_tokens:,}, target: {target_tokens:,}")
    
    return messages