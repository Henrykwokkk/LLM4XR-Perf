from tqdm import tqdm
import json
import os
import argparse

parser = argparse.ArgumentParser(description="Settings for Retrieve mode")
parser.add_argument('-e', '--eval_dir', default='./data/gemini',
                       help='Evaluation dataset directory')
parser.add_argument('-n', '--line', default=0,
                       help='Number of context lines for evaluation (default: 0)')
args = parser.parse_args()
eval_dir = args.eval_dir
eval_line = args.line

if eval_line is not None:
    eval_line = int(eval_line)

for file in os.listdir(eval_dir):
    if file.endswith('.json'):
        with open(os.path.join(eval_dir, file), 'r', encoding='utf-8') as f:
            data = f.readlines()
            TP = 0
            FP = 0
            FN = 0
            hit_num = 0
            total_instances = 0

            skipped_lines = 0
            
            for line_num, line in enumerate(tqdm(data, desc="Processing instance", unit="instance"), 1):
                try:
                    instance = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f'Warning: Skipping malformed JSON at line {line_num}: {str(e)}')
                    skipped_lines += 1
                    continue
                
                # Treat each data line as the smallest unit; initialize hit status for the line.
                is_hit = False
                total_instances += 1
                
                ground_truth_in_context = []
                ground_truth_paths = []
                target_file_paths = []
                # Use only file paths present in generated results as a filter for ground truth,
                # and place them into ground_truth_in_context.
                # inference_results are LLM outputs; ground_truth_results are ground truth
                # (lines removed in the commit diff).
                for inference_result in instance['inference_results']:
                    file_path = inference_result['path']
                    if file_path not in target_file_paths:
                        target_file_paths.append(file_path)

                for ground_truth_location in instance['ground_truth_results']:
                    ground_truth_path = ground_truth_location['path']
                    if ground_truth_path in target_file_paths:
                        if ground_truth_path not in ground_truth_paths: 
                            ground_truth_paths.append(ground_truth_path)
                        ground_truth_in_context.append(ground_truth_location)
                # Only treat ground truth that appears in inference_results as ground_truth_in_context.

                for inference_result in instance['inference_results']:
                    file_path = inference_result['path']
                    line_no = inference_result['line']
                    # Only evaluate predictions whose file path is in target_file_paths.
                    if file_path in ground_truth_paths:
                        for ground_truth_location in ground_truth_in_context:
                            if line_no >= ground_truth_location['line']-eval_line and line_no <= ground_truth_location['line']+eval_line:
                                TP += 1
                                is_hit = True  # At least one TP on this line; mark as hit.
                                break
                        else:
                            FP += 1
                
                for ground_truth_location in ground_truth_in_context:
                    if ground_truth_location not in instance['inference_results']:
                        FN += 1
                    # If ground_truth_location is in inference_results, it was already counted as TP; do not count as TN.
                
                # If this line has at least one TP, then hit_num += 1.
                if is_hit:
                    hit_num += 1

            print('--------------------------------')
            print('File:', file)
            print('TP:',TP)
            print('FP:',FP)
            print('FN:',FN)
            if skipped_lines > 0:
                print(f'Skipped {skipped_lines} malformed line(s)')
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            # accuracy = number of lines with at least one TP / total lines (excluding skipped lines)
            accuracy = hit_num / total_instances if total_instances > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f'Precision: {precision:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'F1 Score: {f1_score:.4f}')
            print(f'Recall: {recall:.4f}')
    