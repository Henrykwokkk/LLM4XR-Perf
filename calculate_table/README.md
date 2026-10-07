# Paper table regeneration scripts

This directory contains the scripts used to calculate the data reported in
Tables 1–7. 

## Script map

| Paper table | Purpose | Script |
|---|---|---|
| Table 1 | Evaluate answer streams and calculate end-to-end metrics | `Table_1_evaluation_three_answers.py` |
| Table 2 | Calculate abstention rates on retrieval-successful subsets | `Table_2_generate_abstention_table.py` |
| Tables 3–5 | Calculate pooled category-level BM25, CodeBERT, and Oracle metrics for all six models | `Table_3_Table_4_Table_5_batch_eval_to_csv.py` |
| Table 6 | Calculate BM25 file-level Recall@k | `Table_6_evaluate_bm25_file_recall.py` |
| Table 6 | Calculate CodeBERT file-level Recall@k | `Table_6_evaluate_codebert_file_recall.py` |
| Table 7 | Calculate conditional and end-to-end metrics for varying k | `Table_7_recompute_conditional_metrics.py` |


## Usage

Install the dependencies, then run scripts from the project root:

```powershell
python -m pip install -r calculate_table/requirements.txt
python calculate_table/Table_7_recompute_conditional_metrics.py
```

The Table 6 scripts reuse `inference.py` and modules in `utils/`. BM25 may need
GitHub access when its local retrieval cache is incomplete; supply a token with
the `GITHUB_TOKEN` environment variable or the `--token` option. 

Use `--help` on scripts that expose command-line options.
