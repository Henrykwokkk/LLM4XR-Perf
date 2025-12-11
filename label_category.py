import yaml
from utils.prompt_loader import load_prompt
from openai import OpenAI
import json
import tqdm
import tiktoken
import httpx
from utils.input_process import (
    CONTEXT_WINDOWS,
    get_encoder_for_model,
    safe_encode_len,
    count_message_tokens,
    reduce_text_to_fit_context,
)

performance_defect_categories = ['Misusage of API and Language','Negative Visualization','High-cost Operation','Thread Conflict','Useless Operation','Heap Misallocation','Overcomplex Rendering and UI','Compatibility']

custom_timeout = httpx.Timeout(1200.0, connect=60.0)
client = OpenAI(
    api_key=<API_KEY>,  #柏拉图的api
    base_url=<BASE_URL>,
    timeout=custom_timeout,
)




model = "gemini-2.5-pro"   # qwen3-235b-a22b gemini-2.5-pro
print(model)


data_file = '/data/guohy/VRopenproject/crawl-script/202509_PerfDetector/data/github_commit_request_output_with_pr.json'
context_window = CONTEXT_WINDOWS[model]


with open(data_file,'r') as f, open(data_file.replace('.json','_{}_labeled.txt'.format(model)),'a') as fw:
    data = f.readlines()
    for i,line in enumerate(tqdm.tqdm(data, desc="Processing instance", unit="instance")):
        instance = json.loads(line)
        pull_request_message = instance['context_info']['associated_pull_requests'][0]['title'] + '\n' + instance['context_info']['associated_pull_requests'][0]['body']
        if instance['context_info']["associated_pull_requests"][0]["issues"]["closing_issues"] != []:
            problem_statement = instance['context_info']["associated_pull_requests"][0]["issues"]["closing_issues"][0]['title'] + '\n' + instance['context_info']["associated_pull_requests"][0]["issues"]["closing_issues"][0]['body']
        elif instance['context_info']["associated_pull_requests"][0]["issues"]["linked_issues"] != []:
            problem_statement = instance['context_info']["associated_pull_requests"][0]["issues"]["linked_issues"][0]['title'] + '\n' + instance['context_info']['associated_pull_requests'][0]["issues"]["linked_issues"][0]['body']
        else:
            problem_statement = ''
        commit_message = instance['context_info']['commit']['message']
        patch = instance['context_info']['commit_patch_unified']
        if problem_statement != '':
            prompt_text= load_prompt(
                "classify_defect_category",
                problem_statement=problem_statement,
                pull_request_message=pull_request_message,
                commit_message=commit_message,
                patch=patch
            )
        else:
            prompt_text= load_prompt(
                "classify_defect_category",
                pull_request_message=pull_request_message,
                commit_message=commit_message,
                patch=patch
            )
        sysyem_prompt, user_prompt = prompt_text
        # print(sysyem_prompt)
        # print('-------------------')
        # print(user_prompt)
        # print('===================')
        messages = [{'role': 'system', 'content': sysyem_prompt}]
        messages.append({'role': 'user', 'content': user_prompt})

        try:
            if count_message_tokens(messages, model) > (context_window - 20):
                messages = reduce_text_to_fit_context(messages, model, context_window)
            if model == 'qwen3-235b-a22b':
                model_kwargs = {"extra_body":{"enable_thinking":False}}
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0,
                    **model_kwargs
                )
            else:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0,
                )
        except Exception as e:
            print('Error at line {}: {}'.format(i, str(e)))
            fw.write('Error\n')
            fw.flush()
            continue

        response = chat_completion.choices[0].message.content

        tries = 0
        while response not in performance_defect_categories and tries < 3:
            tries += 1
            print('Response not in categories, retry {}/3'.format(tries))
            try:
                if model == 'qwen3-235b-a22b':
                    model_kwargs = {"extra_body":{"enable_thinking":False}}
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0,
                        **model_kwargs
                    )
                else:
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0,
                    )
            except Exception as e:
                print('Error at line {}: {}'.format(i, str(e)))
                break

            response = chat_completion.choices[0].message.content
        
        if response not in performance_defect_categories:
            fw.write('Error\n')
            fw.flush()
            print('Final response not in categories, write Error')
        else:
            fw.write(response + '\n')
            fw.flush()