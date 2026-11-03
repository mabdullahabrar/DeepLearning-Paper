# Assignment 02 — Reproduction of LogADReft (ReFT baselines for log anomaly detection)

Deep Learning (AIC 401), Bahria University Islamabad — Muhammad Abdullah Abrar (01-134232-223), Shehryar Zafar (01-134232-236)

## Paper being reproduced
Lim, Y. F., Zhu, J., Pang, G. (2025). *Adapting Large Language Models for Parameter-Efficient Log Anomaly Detection.* PAKDD 2025, Springer LNCS vol. 15876.
DOI: https://doi.org/10.1007/978-981-96-8298-0_26 · arXiv: https://arxiv.org/abs/2503.08045

## What this repository reproduces
RoBERTa-ReFT and GPT2-ReFT (rank 8) on the **HDFS** and **BGL** log datasets, using the authors' released code
(https://github.com/mala-lab/LogADReft, cloned at run time and **not copied here**).
Not reproduced: Llama-3 models (8B parameters), LoRA models, Spirit/Thunderbird, and the paper's extra experiments.

## Contents
| File | Purpose |
|---|---|
| `LogADReft_Reproduction.ipynb` | Complete pipeline: data download, preprocessing, dataset statistics, model summary, training, evaluation, comparison with the paper |
| `requirements.txt` | Software environment |
| `Results/` | Tables, figures, logs and configs produced by the notebook (trained ReFT weights are not included: they were lost when the Colab session was recycled; rerun the notebook to regenerate them) |
| `Paper-Report.pdf` | Technical report |

## How to run (Google Colab, free T4 GPU)
1. Open the notebook in Colab: *Runtime → Change runtime type → GPU*.
2. *Runtime → Run all*. Edit only the `CONFIG` cell if you want different sample sizes.
3. Download `results_package.zip` at the end and copy its content into `Results/`.

The notebook is restart-safe: every cell reloads its settings from `results/config.json` and finished steps (data splits, finished training runs) are skipped. If Colab restarts, just run the cells again from the top.

Local machine: Python 3.11 or newer, an NVIDIA GPU, `pip install -r requirements.txt` plus a CUDA build of PyTorch, then run the notebook with Jupyter.

## Dataset
Parsed HDFS and BGL logs from Le & Zhang (ICSE 2022), Zenodo record 8115559: https://zenodo.org/record/8115559
(raw logs originate from Xu et al., SOSP 2009 for HDFS and Oliner & Stearley, DSN 2007 for BGL). The notebook downloads the files automatically.
Datasets are **not** stored in this repository.

## Model configuration and training settings
Taken from the authors' scripts `scripts/main_results/{roberta,gpt2}_reft.sh`: LoReFT rank 8, all layers, shared weights, AdamW, learning rate 1e-4,
linear schedule with 10% warmup, dropout 0.1, weight decay 0, max length 510, seed 42, effective batch size 32,
epochs RoBERTa 3 (HDFS) / 6 (BGL), GPT-2 6 (both). Intervention position: first token (RoBERTa), last token (GPT-2).

## Deviations from the original setup (also listed in the report)
* **Subsampling:** `N_TRAIN` / `N_TEST` examples (default 3000 / 3000) through the authors' own `--max_n_train_example` / `--max_n_eval_example` options, because full splits are too slow on free GPUs.
  The authors' sampler keeps at most 10% anomalies in the test sample, so class ratios differ from the full test set.
* **Hardware:** float32 and batch 16 x 2 gradient accumulation on GPUs without bf16 support (e.g. T4).
* **Metrics:** we report both the macro-averaged scores computed by the released evaluation code and anomaly-class (binary) scores, because the paper does not state which averaging it uses.

## Issues found in the original repository
* README says `python data_process/data_loader.py`; the folder is `data_process_logs/`.
* The README's default loader settings (`is_test_train_ratio=False`, `is_chronological=True`) crash for sliding-window datasets (`UnboundLocalError: n_train_for_n_test`). We use `is_test_train_ratio=True`, which gives the same chronological 80/20 split. (Demonstrated by the notebook on a tiny synthetic file.)
* The loader writes `0.8train.pkl` / `0.8test.pkl` but the training scripts read `train.pkl` / `test.pkl`.
* `wandb`, `scikit-learn` and `pandas` are imported but missing from `requirements.txt`.
* `pyreft==0.0.8` only requires `pyvene>=0.1.4`, so a fresh install picks the newest `pyvene` (0.1.8), which breaks the authors' code with `TypeError: 'LoreftIntervention' object is not subscriptable`. Pinning `pyvene==0.1.6` (released the same day as pyreft 0.0.8) fixes it.

## Results
Scaled-down reproduction (3,000 training and 3,000 test sequences per run, single seed), so the numbers are **not** an exact reproduction of the paper.
F1 of the paper versus our runs (anomaly class = binary F1 on the anomalous class; macro = the averaging used by the authors' released ReFT evaluation code):

| Dataset | Model | Paper F1 | Reproduced (macro) | Reproduced (anomaly class) |
|---|---|---|---|---|
| HDFS | RoBERTa-ReFT | 0.9965 | 0.8514 | 0.7265 |
| HDFS | GPT2-ReFT | 0.9971 | 0.8666 | 0.7628 |
| BGL | RoBERTa-ReFT | 0.9558 | 0.8911 | 0.8008 |
| BGL | GPT2-ReFT | 0.8919 | 0.5578 | 0.1944 |

The full table (precision, recall, differences), training curves, run times and logs are in `Results/` (`comparison_table.md`, `run_summary.csv`, `curves_*.png`, `logs/`). Discussion of the gaps is in `Paper-Report.pdf`.

## Credits
Authors' code: https://github.com/mala-lab/LogADReft · ReFT/pyreft: Wu et al. (2024) · Hugging Face Transformers (Wolf et al., 2020) · datasets: Le & Zhang (2022).
