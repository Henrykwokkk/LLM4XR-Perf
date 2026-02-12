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


from utils.input_process import (
    CONTEXT_WINDOWS,
    count_message_tokens,
    reduce_code_text_to_fit_context,
)

from utils.output_json_process import (
    parse_llm_content
)


client = OpenAI(
    api_key=<API_KEY>,  #
    base_url=<BASE_URL>
)
model = 'gemini-2.5-pro'   # qwen3-235b-a22b gemini-2.5-pro
context_window = CONTEXT_WINDOWS[model]

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
    """
    ps = PatchSet(diff_text)
    out = []
    for pf in ps:
        target = normalize_path(pf.target_file)  # new file path（b/...）
        source = normalize_path(pf.source_file)  # old file path（a/...）

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

    base_commit_files = [] 
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
            tokens=[<GITHUB_API_TOKEN>] 
        )
        context_files = [{"path": p, "content": c} for p, c in retrieval_files.items()]
    
    instance = {
        "problem_statement": problem_statement.strip(),
        "context_files": context_files,
        "defect_line_ground_truth": defect_line_ground_truth,}
    
    return instance



if __name__ == "__main__":
    file_source = 'bm25'  # "oracle" or "bm25"
    k = 10  # Only used if file_source is "bm25"
    if file_source == 'bm25':
        output_path = f'./data/github_commit_request_output_with_issue_defect_localization_{model}_{file_source}_{k}.json'
    else:
        output_path = f'./data/github_commit_request_output_with_issue_defect_localization_{model}_{file_source}.json'



    with open('./data/github_commit_request_output_with_issue_final_labeled.json','r') as f, open(output_path,'w') as fw:
    
        data = f.readlines()
        for i,line in enumerate(tqdm(data, desc="Processing instance", unit="instance")):
            instance = json.loads(line)
            try:
                defect_localization_data = create_defect_localization_data(
                    original_data=instance,
                    file_source=file_source,  # "oracle" or "bm25"
                    k=k  # Only used if file_source is "bm25"
                )

            except Exception as e:
                logger.error(f"Error processing instance {i}: {e}")
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

            if count_message_tokens(messages, model) > (context_window - 10000):
                messages = reduce_code_text_to_fit_context(messages, model, context_window)

            
            try:
                if 'qwen' in model:
                    model_kwargs = {"extra_body":{"enable_thinking":False}}
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0.1,
                        **model_kwargs
                    )
                else:
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0.1,
                    )
            except Exception as e:
                print('Error at line {}: {}'.format(i, str(e)))
                break

            response_text = chat_completion.choices[0].message.content
            print(response_text)
            try:
                parse_results = parse_llm_content(response_text)
            except Exception as e:
                parse_results = []
                print('Error at line {}: {}'.format(i, str(e)))
            
            instance['inference_results'] = parse_results
            instance['ground_truth_results'] = defect_localization_data['defect_line_ground_truth']
            instance['context_files_paths'] = [cf['path'] for cf in files]

            fw.write(json.dumps(instance) + '\n')
            fw.flush()

