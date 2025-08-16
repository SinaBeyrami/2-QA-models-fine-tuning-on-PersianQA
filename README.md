# 2 — QA Models Fine-Tuning on PersianQA

_Built during my **ML Engineer Intern** role at **Roshan**._


Fine-tuning and evaluating Persian question-answering models on a SQuAD-style Persian dataset (PersianQA). Two families are covered:

* **XLM-RoBERTa (large)** — strong multilingual baseline
* **ParsBERT (PQuAD checkpoint)** — Persian-specialized model

Both include **evaluation** and **fine-tuning** notebooks, plus compact metric pipelines (EM/F1 via `evaluate.squad_v2`).

---

## Repo structure

```
2-QA-models-fine-tuning-on-PersianQA/
├─ XLM-roberta-model/
│  ├─ Pretrained_XLM_roberta_Evaluation.ipynb
│  └─ XLM_roberta_FineTune.ipynb
├─ Parsbert-PQuAD-model/
│  ├─ Pretrained_Parsbert_PQuAD_Evaluation.ipynb
│  └─ Parsbert_PQuAD_FineTune.ipynb
└─ Data (PersianQA)/
   ├─ pqa_train.json   # SQuAD-style train
   └─ pqa_test.json    # SQuAD-style test
```

### Data format (SQuAD-v2 style)

Each file contains top-level fields:

* `title`
* `paragraphs`: list of { `context`, `qas` }
* Each `qa` has: `id`, `question`, `is_impossible`, and `answers`
* For **unanswerable** questions: `is_impossible=true` and `answers=[]`

Example snippets are in the repo under **Data (PersianQA)**.

---

## Environment

* Python 3.10+
* GPU recommended (T4/P100/V100 or better)

Common packages (notebooks install as needed):

```
pip install -q transformers datasets evaluate accelerate tqdm fsspec==2023.9.2
```

---

## XLM-RoBERTa (Large)

### 1) Evaluate a pretrained model

Notebook: `XLM-roberta-model/Pretrained_XLM_roberta_Evaluation.ipynb`

* Model: `pedramyazdipoor/persian_xlm_roberta_large`
* Tokenization window: `MAX_LEN=512` (question+context), `N_BEST=20`, `MAX_ANS_LEN=30`
* Uses HF `postprocess_qa_predictions` (with a lightweight fallback)
* Metrics: `evaluate.load("squad_v2")`

**Reported on `pqa_test.json`:**

* **EM: 51.08**
* **F1: 65.08**

> Adjust paths if not on Kaggle (default expects `/kaggle/input/test-set/pqa_test.json`).

### 2) Fine-tune with 5-fold CV

Notebook: `XLM-roberta-model/XLM_roberta_FineTune.ipynb`

* Sliding window: `max_len=384`, `doc_stride=128`
* 5-fold CV via `KFold`
* **Partial unfreeze**: QA head + **last 2** transformer layers (`freeze_backbone(unfreeze_last_n=2)`)
* TrainingArguments (per fold):
  `epochs=10`, `lr=2e-5`, `bs=8`, `fp16=True`, cosine schedule, warmup=10%
* Fast EM/F1 metric function for **near-instant** fold evaluation

**Observed (early stop due to GPU time):**

* \~**EM 67**, **F1 82** after **5 epochs** (not the full 10–50 planned)

> To reproduce: set the correct train/test JSON paths, keep the same tokenization and doc stride, and run all cells.

---

## ParsBERT (PQuAD)

### 1) Evaluate a pretrained model

Notebook: `Parsbert-PQuAD-model/Pretrained_Parsbert_PQuAD_Evaluation.ipynb`

* Model: `pedramyazdipoor/parsbert_question_answering_PQuAD`
* Uses HF `pipeline("question-answering")`
* `max_seq_len=512`, `doc_stride=128`
* Metrics: `evaluate.squad_v2`

**Run** on `pqa_test.json`; prints EM and F1.

### 2) Fine-tune with 5-fold CV

Notebook: `Parsbert-PQuAD-model/Parsbert_PQuAD_FineTune.ipynb`

* Flattens SQuAD → token features with sliding window `(max_len=384, stride=128)`
* Fast metrics (EM/F1) using a single best span per example
* TrainingArguments (per fold):
  `epochs=5`, `lr=2e-5`, `bs=8`, cosine schedule, warmup=10%, `fp16=True`
* CSV loss logger callback per fold

**Observed (first fold, 5 epochs):**

* \~**EM 40**, **F1 56**
  (Kaggle disk limits cleared checkpoints before full test evaluation)

---

## Results snapshot

| Model                              | Split        | EM   | F1   | Notes                              |
| ---------------------------------- | ------------ | ---- | ---- | ---------------------------------- |
| XLM-RoBERTa large (pretrained)     | `pqa_test`   | 51.1 | 65.1 | `MAX_LEN=512`, strict postprocess  |
| XLM-RoBERTa large (fine-tune, 5ep) | CV (val)     | 67   | 82   | Early stop; last-2 layers unfrozen |
| ParsBERT PQuAD (fine-tune, 5ep)    | Fold-1 (val) | 40   | 56   | Disk limits; partial run           |

> Expect gains from: more epochs, larger batch sizes, longer contexts, and better checkpoint retention.

---

## Tips & knobs

* **Context window**: `max_len=384/512` and `doc_stride=128` work well; increase stride if answers near chunk edges.
* **No-answer handling**: both pipelines treat CLS (or empty span) as no-answer; metrics are SQuAD-v2 compliant.
* **Freezing policy**: unfreezing only last N layers often stabilizes fine-tuning with limited compute.
* **Mixed precision**: keep `fp16=True` on GPUs that support it.
* **Kaggle/Colab paths**: update JSON paths if running locally.

---

## Reproduce locally (minimal)

```bash
pip install -q transformers datasets evaluate tqdm
python - << 'PY'
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from evaluate import load
import json, torch
from pathlib import Path

MODEL = "pedramyazdipoor/persian_xlm_roberta_large"
TEST  = Path("Data (PersianQA)/pqa_test.json")

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
m   = AutoModelForQuestionAnswering.from_pretrained(MODEL).eval().to("cuda" if torch.cuda.is_available() else "cpu")
metric = load("squad_v2")

# See notebook for full postprocessing; this is a placeholder.
print("Load the notebook for complete evaluation with offsets & n-best.")
PY
```

For full evaluation and fine-tuning, open the corresponding notebooks and run all cells (GPU recommended).

---

## Acknowledgements

* Models: `pedramyazdipoor/persian_xlm_roberta_large`, `pedramyazdipoor/parsbert_question_answering_PQuAD`
* Libraries: Hugging Face (transformers/datasets/evaluate), scikit-learn, PyTorch
* Metrics: `evaluate.squad_v2`

Thanks to **Roshan** for the opportunity to work on Persian QA modeling during my **ML Engineer internship**.

---

## License & usage

For academic use. Respect licenses of datasets and upstream models.

---

## Contact

Sina Beyrami — Sina.Bey743@gmail.com

**Role:** ML Engineer Intern @ Roshan

Questions/feedback welcome.
