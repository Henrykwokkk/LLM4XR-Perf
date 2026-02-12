'''Utilze LLM to gnereate defect localization information based on issue and context files'''


from utils.prompt_loader import load_prompt
from utils.bm25_retrieval import (
    DOCUMENT_ENCODING_FUNCTIONS,
    build_documents,
    clone_repo,
)
from openai import OpenAI
import json
from tqdm import tqdm
from typing import Literal
from loguru import logger
from unidiff import PatchSet
from pathlib import Path
import random
import argparse
import re


from utils.input_process import (
    CONTEXT_WINDOWS,
    count_message_tokens,
    reduce_code_text_to_fit_context,
)

from utils.output_json_process import (
    parse_llm_content
)


client = OpenAI(
    api_key='',  
    base_url='',
    timeout=600
)

def normalize_path(p: str):
    if not p or p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p

def is_cs(path: str, case_insensitive: bool = True) -> bool:
    if path is None:
        return False
    return path.lower().endswith(".cs") if case_insensitive else path.endswith(".cs")

def extract_cs_changed_lines(diff_text: str, include: str = "both"):
    """
    include: 'added' | 'removed' | 'both'
    Return a list of dicts, each containing path and line; when include='both',
    also include 'kind' to distinguish 'added'/'removed'.
    """
    ps = PatchSet(diff_text)
    out = []
    for pf in ps:
        target = normalize_path(pf.target_file)  # New file path (b/...)
        source = normalize_path(pf.source_file)  # Old file path (a/...)

        for hunk in pf:
            for line in hunk:
                if include == "added":
                    if line.is_added and is_cs(target):
                        out.append({"path": target, "line": line.target_line_no})
                elif include == "removed":
                    if line.is_removed and is_cs(source):
                        out.append({"path": source, "line": line.source_line_no})
                else:  # both
                    if line.is_added and is_cs(target):
                        out.append({"path": target, "line": line.target_line_no, "kind": "added"})
                    elif line.is_removed and is_cs(source):
                        out.append({"path": source, "line": line.source_line_no, "kind": "removed"})
    return out

def get_all_files(
    original_data: dict,
    retrieval_output_dir: Path | None,
    k: int,
    tokens: list[str] | None = None,
) -> dict[str, str]:
    """
    Get all available files from the repository up to k limit.

    This retrieves all files from the repository at the base commit using
    a temporary git worktree for thread-safety and proper cleanup.

    Args:
        instance: The code review task instance
        retrieval_output_dir: Output directory for git operations
        k: Maximum number of files to retrieve (if None, retrieves all files)
        tokens: GitHub API tokens (optional)

    Returns:
        Dictionary mapping file paths to file contents
    """

    if retrieval_output_dir is None:
        raise ValueError("retrieval_output_dir is required for 'all' file source")

    all_files = {}

    try:
        # Setup repo path
        repo_path = Path(
            f"{retrieval_output_dir}/repos/{original_data['context_info']['repository']['owner']}_{original_data['context_info']['repository']['name']}"
        )

        # Clone the repository if it doesn't exist
        if not repo_path.exists():
            Path(retrieval_output_dir).mkdir(parents=True, exist_ok=True)
            repo_dir = clone_repo(
                f"{original_data['context_info']['repository']['owner']}/{original_data['context_info']['repository']['name']}",
                retrieval_output_dir,
                random.choice(tokens) if tokens else None,
            )
        else:
            repo_dir = str(repo_path)

        all_files = build_documents(
            repo_dir,
            original_data['context_info']['commit']['parents'][0]['sha'],
            DOCUMENT_ENCODING_FUNCTIONS["contents_only"],
            include_readmes=True,
        )
        for path in list(all_files.keys()):
            if not is_cs(normalize_path(path)):
                del all_files[path]

        # Limit the number of files
        if len(all_files) > k:
            # Convert to list, slice to k items, then convert back to dict
            all_files_items = list(all_files.items())[:k]
            all_files = dict(all_files_items)

        logger.info(
            f"Retrieved {len(all_files)} cs files for instance {original_data['context_info']['commit']['parents'][0]['sha']} using 'all' file source"
        )

    except Exception as e:
        logger.error(
            f"Failed to retrieve all files for {original_data['context_info']['repository']['owner']}/{original_data['context_info']['repository']['name']}@{original_data['context_info']['commit']['parents'][0]['sha']}: {e}"
        )
        # Return empty dict on failure
        return {}

    return all_files



def create_defect_localization_data(
        original_data: dict, 
        file_source: Literal["oracle", "bm25"],
        k: int | None = None,
) -> dict:
    """
    Create defect localization data using LLM based on the issue and context files.

    Args:
        original_data: The original data containing issue and context information.
        file_source: The source of the context files, either "oracle" or "bm25".
        k: The number of top relevant files to retrieve if file_source is "bm25".

    Returns:
        A dictionary containing the original data along with the generated defect localization information.
    """
    problem_statement = ''
    for i in original_data['context_info']['associated_pull_requests'][0]['issues']['closing_issues']:
        problem_statement += i['title'] + '\n' + i['body'] + '\n'
    for i in original_data['context_info']['associated_pull_requests'][0]['issues']['linked_issues']:
        problem_statement += i['title'] + '\n' + i['body'] + '\n'

    diff_patch_text = original_data['context_info']['commit_patch_unified']
    defect_line_ground_truth = extract_cs_changed_lines(diff_patch_text, include="removed")

    base_commit_files = []  # List of files and contents before the change
    for file_info in original_data['context_info']['files']:
        if is_cs(normalize_path(file_info['path'])):
            base_commit_files.append({
                'path': normalize_path(file_info['path']),
                'content': file_info['pre_change']['content'] if file_info['pre_change'] else ''
            })

    if file_source == "oracle":
        context_files = base_commit_files
    else:  # bm25
        retrieval_files = get_all_files(
            original_data,
            retrieval_output_dir=Path("./data"),
            k=k,
            tokens=""  # Replace with your GitHub API tokens
        )
        context_files = [{"path": p, "content": c} for p, c in retrieval_files.items()]
    
    instance = {
        "problem_statement": problem_statement.strip(),
        "context_files": context_files,
        "defect_line_ground_truth": defect_line_ground_truth,}
    
    return instance



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for Retrieve mode")
    parser.add_argument('-f', '--file-source', default='bm25',
                       help='File source, oracle or bm25')
    parser.add_argument('-k', '--k', type=int, default=10,
                       help='Number of files returned in BM25 mode')
    parser.add_argument('-m', '--model', default='gemini-2.5-pro',
                       help='Model name')

    args = parser.parse_args()
    file_source = args.file_source
    k = args.k
    model = args.model
    context_window = CONTEXT_WINDOWS[model]
    output_path = f'./data/github_commit_request_output_with_issue_defect_localization_{model}_{file_source}_{k}.json'

    with open('./data/github_commit_request_output_with_issue_final_labeled.json','r', encoding='utf-8') as f, open(output_path,'w', encoding='utf-8') as fw:
    
        data = f.readlines()
        total_instances = len(data)
        pbar = tqdm(total=total_instances, desc=f"Processing ({model}, {file_source})", unit="instance")
        
        for i,line in enumerate(data):
            pbar.set_postfix({'current': i+1, 'total': total_instances})
            instance = json.loads(line)
            try:
                defect_localization_data = create_defect_localization_data(
                    original_data=instance,
                    file_source=file_source,  # "oracle" or "bm25"
                    k=k  # Only used if file_source is "bm25"
                )

            except Exception as e:
                logger.error(f"Error processing instance {i}: {e}")
                pbar.update(1)
                continue

            problem_statement = defect_localization_data['problem_statement']
            files = defect_localization_data['context_files']
            
            prompt_text= load_prompt(
                    "defect_localization_data",
                    problem_statement=problem_statement,
                    files=files,
                    add_line_numbers=True
                )
            
            sysyem_prompt, user_prompt = prompt_text
            # print(user_prompt)
            messages = [{'role': 'system', 'content': sysyem_prompt}]
            messages.append({'role': 'user', 'content': user_prompt})

            token_count = count_message_tokens(messages, model)
            
            # For gpt-5, use a larger buffer (consider output tokens and API limits)
            buffer_tokens = context_window // 10
            
            initial_target_tokens = context_window - buffer_tokens
            target_tokens = initial_target_tokens
            
            # Retry mechanism: up to 3 retries; the first token reduction also counts as one retry
            max_retries = 3
            retry_count = 0
            chat_completion = None
            last_error = None
            
            def reduce_tokens(messages, current_token_count, current_target, attempt_num, reason="initial check"):
                """Unified token reduction function"""
                excess_ratio = (current_token_count - current_target) / current_token_count if current_token_count > 0 else 0
                
                # Adjust reduction factor based on excess ratio and retry count (more conservative strategy)
                if excess_ratio > 0.3:  
                    reduction_factor = 0.75  
                elif excess_ratio > 0.15:  
                    reduction_factor = 0.80  
                else:  # Subsequent attempts, moderate reduction
                    reduction_factor = 0.90  
                
                # If the API returns an error and token_count < target_tokens, the actual limit is stricter.
                # In this case, compute the new target based on current_token_count instead of current_target;
                # otherwise target_reduction_ratio would be > 1, making the reduction ineffective.
                if excess_ratio < 0 and "API returned" in reason:
                    # API error but token_count < target, meaning the actual limit is stricter than target
                    # Compute new target based on actual token_count to ensure target_reduction_ratio < 1
                    new_target = int(current_token_count * reduction_factor)
                else:
                    # Normal case: compute based on current_target
                    new_target = int(current_target * reduction_factor)
                
                target_reduction_ratio = new_target / current_token_count if current_token_count > 0 else reduction_factor
                
                logger.info(f"[Attempt {attempt_num}/{max_retries}] {reason}: "
                           f"tokens {current_token_count:,} > target {current_target:,} "
                           f"(excess: {excess_ratio:.1%}), reducing to {new_target:,} tokens (factor: {reduction_factor:.0%})")
                
                # Perform reduction, passing target reduction ratio to improve efficiency
                messages = reduce_code_text_to_fit_context(
                    messages, model, new_target + buffer_tokens, 
                    buffer_tokens=buffer_tokens,
                    target_reduction_ratio=target_reduction_ratio
                )
                new_token_count = count_message_tokens(messages, model)
                reduction_achieved = current_token_count - new_token_count
                actual_reduction_ratio = reduction_achieved / current_token_count if current_token_count > 0 else 0
                
                logger.info(f"Reduction result: {current_token_count:,} -> {new_token_count:,} tokens "
                           f"(-{reduction_achieved:,}, {actual_reduction_ratio:.1%} reduction)")
                
                return messages, new_token_count, new_target
            
            while retry_count < max_retries:
                # Check whether token reduction is needed
                if token_count > target_tokens:
                    retry_count += 1
                    reason = "Token count exceeds limit" if retry_count == 1 else "Still exceeds after reduction"
                    messages, token_count, target_tokens = reduce_tokens(
                        messages, token_count, target_tokens, retry_count, reason
                    )
                
                # Try the API call
                try:  
                    if retry_count == 0:
                        logger.info(f"Requesting instance {i+1}/{total_instances} with model {model} (tokens: {token_count:,})")
                    else:
                        logger.info(f"Requesting instance {i+1}/{total_instances} with model {model} "
                                   f"(tokens: {token_count:,}, after {retry_count} reduction{'s' if retry_count > 1 else ''})")
                    
                    model_kwargs = {"extra_body":{"enable_thinking":False}}
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0.1,
                        **model_kwargs
                    )
                    # Success, exit retry loop
                    if retry_count > 0:
                        logger.info(f"Successfully processed instance {i+1} after {retry_count} reduction{'s' if retry_count > 1 else ''}")
                    break
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    error_code = None
                    
                    # Try to extract error code (support multiple error formats)
                    try:
                        # OpenAI SDK error format
                        if hasattr(e, 'response') and e.response is not None:
                            if hasattr(e.response, 'json'):
                                error_data = e.response.json()
                            elif hasattr(e.response, 'text'):
                                import json
                                error_data = json.loads(e.response.text)
                            else:
                                error_data = {}
                            
                            if 'error' in error_data:
                                error_code = error_data['error'].get('code')
                        
                        # Check whether error message contains error code (string format)
                        if 'context_length_exceeded' in error_str:
                            error_code = 'context_length_exceeded'
                    except:
                        pass
                    
                    # Check whether this is a context window overflow error
                    is_context_error = (
                        'context_length_exceeded' in error_str or 
                        'context window' in error_str.lower() or
                        'exceeds the context window' in error_str.lower() or
                        'input exceeds the context window' in error_str.lower() or
                        'Range of input length' in error_str.lower() or
                        error_code == 'context_length_exceeded'
                    )
                    
                    if is_context_error and retry_count < max_retries:
                        # Context window error: continue reducing and retry
                        retry_count += 1
                        messages, token_count, target_tokens = reduce_tokens(
                            messages, token_count, target_tokens, retry_count, "API returned context window error"
                        )
                        # Continue retry loop
                        continue
                    else:
                        # Not a context error, or maximum retries reached
                        if retry_count >= max_retries and is_context_error:
                            logger.error(f'Error at line {i}: Context window exceeded after {max_retries} retries. '
                                       f'Final tokens: {token_count:,}, target: {target_tokens:,}')
                        else:
                            logger.error(f'Error at line {i}: {error_str}')
                        # Exit retry loop and continue with the next instance
                        break
            
            # If all retries fail, skip this instance
            if chat_completion is None:
                logger.error(f'Failed to get response for instance {i} after {max_retries} retries: {last_error}')
                pbar.update(1)
                continue

            response_text = chat_completion.choices[0].message.content
            print(response_text)
            # If model is gpt-5, remove content between <think>...</think>
            if model == 'gpt-5.2':
                response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            try:
                parse_results = parse_llm_content(response_text)
            except Exception as e:
                parse_results = []
                logger.error('Error parsing response at line {}: {}'.format(i, str(e)))
            
            instance['inference_results'] = parse_results
            instance['ground_truth_results'] = defect_localization_data['defect_line_ground_truth']
            instance['context_files_paths'] = [cf['path'] for cf in files]

            fw.write(json.dumps(instance, ensure_ascii=False) + '\n')
            fw.flush()
            pbar.update(1)
        
        pbar.close()

