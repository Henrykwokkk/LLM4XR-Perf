from tqdm import tqdm
import json


with open('./data/github_commit_request_output_with_issue_defect_localization_gemini-2.5-pro_bm25_10.json', 'r') as f:
    data = f.readlines()
    TP = 0
    FP = 0
    FN = 0
    for line in tqdm(data, desc="Processing instance", unit="instance"):
        instance = json.loads(line)
        ground_truth_in_context = []
        target_file_paths = []
        # 
        for inference_result in instance['inference_results']:
            for file_path in inference_result.keys():
                if file_path not in target_file_paths:
                    target_file_paths.append(file_path)

        for ground_truth_location in instance['ground_truth_results']:
            for file_path, line_numbers in ground_truth_location.items():
                if file_path in target_file_paths:
                    ground_truth_in_context.append(ground_truth_location)

        for inference_result in instance['inference_results']:
            if inference_result in instance['ground_truth_results'] and inference_result in ground_truth_in_context:
                TP += 1
            elif inference_result not in instance['ground_truth_results']:
                FP += 1
        for ground_truth_location in ground_truth_in_context:
            if ground_truth_location not in instance['inference_results']:
                FN += 1
    print('TP:',TP)
    print('FP:',FP)
    print('FN:',FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
