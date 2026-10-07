# Paper table calculation scripts

This directory contains the scripts used to calculate the data reported in
Tables 1–7. The scripts read experiment artifacts from the parent project and
write generated JSON, CSV, or LaTeX files under the project's `data/` or
evaluation-output directories.

## Script map

| Paper table | Purpose | Script |
|---|---|---|
| Table 1 | Evaluate answer streams and calculate end-to-end metrics | `Table_1_evaluation_three_answers.py` |
| Table 2 | Calculate abstention rates on retrieval-successful subsets | `Table_2_generate_abstention_table.py` |
| Tables 3–5 | Calculate pooled category-level BM25, CodeBERT, and Oracle metrics for all six models | `Table_3_Table_4_Table_5_batch_eval_to_csv.py` |
| Table 6 | Calculate BM25 file-level Recall@k | `Table_6_evaluate_bm25_file_recall.py` |
| Table 6 | Calculate CodeBERT file-level Recall@k | `Table_6_evaluate_codebert_file_recall.py` |
| Table 7 | Calculate conditional and end-to-end metrics for varying k | `Table_7_recompute_conditional_metrics.py` |

## Project layout

The directory name is `calculate_table`; the former `calulate_table` typo has
been corrected.

```text
PerfDector/
├── calculate_table/
├── data/
├── inference.py
└── utils/
```

## Usage

Install the dependencies, then run scripts from the project root:

```powershell
python -m pip install -r calculate_table/requirements.txt
python calculate_table/Table_7_recompute_conditional_metrics.py
```

The Table 6 scripts reuse `inference.py` and modules in `utils/`. BM25 may need
GitHub access when its local retrieval cache is incomplete; supply a token with
the `GITHUB_TOKEN` environment variable or the `--token` option. No access
token is stored in these scripts.

Use `--help` on scripts that expose command-line options. Generated data files
are ignored by the directory's `.gitignore`.

Line-level scoring uses the LaTeX evaluation scope (retrieved files intersected
with ground-truth files), unique `(path, line)` locations, and one-to-one
tolerance matching. Table 7 and Tables 3–5 calculate their reported metrics
from pooled integer counts. End-to-end accuracy retains the full benchmark (or
full category) population in its denominator, including retrieval failures and
missing outputs.
