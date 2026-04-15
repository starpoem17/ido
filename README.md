# ido

> A study-driven attempt to build a Korean-first LLM from scratch, end to end.

`ido` is a public learning project focused on the full data pipeline behind a Korean language model: raw data collection, normalization, deduplication, tokenizer training, and training-ready dataset construction.

This repository is **not** a production LLM framework and does **not** claim state-of-the-art results. The point of the project is narrower and more practical: to build the stack myself, understand each failure mode, and leave behind a readable pipeline instead of a vague “from-scratch LLM” claim.

At the current stage, the repo reaches the **Lance/LanceDB-ready corpus build** stage. Model training is the next planned step.

---

## Current status

| Stage | Status | Notes |
|---|---|---|
| Raw data collection | Done / ongoing | AIHub, NamuWiki crawling, and Hugging Face datasets |
| Canonical preprocessing | Implemented | Raw sources are normalized into a unified schema |
| Exact dedup | Implemented | Duplicate rows with identical `content` are removed |
| Near dedup | Implemented | MinHash + LSH candidate generation with Jaccard verification |
| Tokenizer build | Implemented | Custom Korean BBPE tokenizer, 48k vocab |
| Token counting | Implemented | Final counts are recomputed with the project tokenizer |
| Lance dataset build | Implemented | Training-ready `dataset.lance` build is in place |
| Pretraining model | Planned | Small 0.2B model |
| Finetuning / context extension | Planned | RoPE-based context expansion roadmap |

---

## What this project is trying to do

The goal is to build a **Korean-specialized LLM pipeline from scratch** with explicit control over the data and preprocessing decisions.

Instead of starting from a pretrained base model and calling it a day, this repo focuses on the less glamorous part that usually decides whether training is even worth doing:

- collecting Korean raw data directly,
- converting heterogeneous sources into one canonical format,
- removing exact and near duplicates,
- training a tokenizer on the cleaned corpus,
- recomputing token counts with that tokenizer,
- and exporting the final corpus into a format that is easy to inspect and train on.

In other words, the project is more about **data quality and pipeline discipline** than about chasing a large parameter count.

---

## Data sources

Current raw sources include:

- **AIHub** datasets
- **NamuWiki** crawling collected directly through the repo’s crawler
- **Hugging Face** datasets
  - [`kuotient/gsm8k-ko`](https://huggingface.co/datasets/kuotient/gsm8k-ko)
  - [`HAERAE-HUB/KOREAN-WEBTEXT`](https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT)
  - [`HAERAE-HUB/HR-Instruct-Math-v0.1`](https://huggingface.co/datasets/HAERAE-HUB/HR-Instruct-Math-v0.1)
  - [`nohurry/Opus-4.6-Reasoning-3000x-filtered`](https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered)

Some data is Korean-native, while some reasoning-oriented data is intended to be translated into Korean through the repo pipeline.

**Important:** this repository does not override the original licenses or usage restrictions of upstream datasets. Check the original dataset cards and licenses before reuse.

---

## Corpus design

The dataset pipeline standardizes everything into a single canonical row schema:

```text
source: String
data_usage: String   # PT | SFT | REASONING
split: String        # train | val
content: String | null
messages: List[Struct{role, content}] | null
token_count: int32
```

Design choices:

- `PT` rows use `content`
- `SFT` and `REASONING` rows use `messages`
- at least one of `content` or `messages` must exist
- final token counts are computed with the project tokenizer
- the final validation split is rebuilt as **99:1**

This makes the raw sources easier to mix, deduplicate, shard, and later train on without carrying around dataset-specific logic forever.

---

## Preprocessing pipeline

The current pipeline is roughly:

```text
raw collection
  -> canonical parquet build
  -> exact dedup
  -> MinHash + LSH near dedup
  -> BBPE tokenizer training
  -> token recounting + row splitting
  -> Lance dataset build
  -> (planned) model pretraining / finetuning
```

### 1. Collection
- Direct collection scripts exist for sources such as **NamuWiki**.
- The project also ingests locally prepared AIHub and Hugging Face data.

### 2. Canonical parquet build
- Heterogeneous raw sources are converted into a unified parquet schema.
- The intermediate staging area is designed to make later dedup/tokenizer steps source-agnostic.

### 3. Exact dedup
- Rows with identical `content` are removed first.
- This stage writes manifests and deletion logs so the process is inspectable.

### 4. Near dedup
- Near-duplicate candidates are generated with **MinHash + LSH**.
- Verification is then done with explicit **Jaccard similarity** on normalized Korean character **7-gram shingles**.
- When duplicates collide, representative rows are chosen with the priority:

```text
REASONING > SFT > PT
```

### 5. Tokenizer
- A custom **Korean BBPE** tokenizer is trained on the deduplicated corpus.
- Tokenizer spec:
  - vocab size: **48,000**
  - algorithm: **Byte-Level BPE**
  - special tokens: `<|bos|>`, `<|eos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|eot_id|>`, `<|pad|>`
- By the current project accounting, the corpus is about **5.9B tokens** under the custom tokenizer.

### 6. Lance build
- Deduplicated parquet shards are converted into a final Lance dataset.
- SFT rows serialize message turns into a training string.
- Long PT and SFT rows are split before export.
- REASONING rows are expanded into:
  - a `PT`-derived row for plain text learning,
  - and a `REASONING` row for structured supervision.

Current output target:

```text
data/korean_processed/final_lancedb/dataset.lance
```

---

## Planned model

The planned training target is intentionally small:

- **Model size:** ~0.2B parameters
- **Attention:** gated attention
- **Positional encoding:** RoPE
- **Context plan:**
  - pretraining at **1024** context length
  - finetuning at **4096**
  - later context window extension as a follow-up step

This is not meant to be a frontier-scale model. The point is to keep the model small enough that pipeline decisions, data quality, and tokenizer behavior remain visible instead of disappearing behind scale.

---

## Repository structure

```text
ido/
├── configs/
├── data/
├── docs/
│   ├── personal/
│   ├── plans/
│   └── pseudo/
├── notebooks/
├── src/
│   ├── collect/
│   ├── preprocess/
│   ├── tokenizer/
│   └── translate/
├── pyproject.toml
└── README.md
```

### `src/`
- `collect/` — raw data collection utilities such as NamuWiki crawling
- `preprocess/` — canonical parquet build, exact dedup, near dedup, Lance export
- `tokenizer/` — tokenizer build, benchmark, and interactive inspection tools
- `translate/` — translation pipeline for turning selected English reasoning data into Korean

### `docs/`
This repo is intentionally documentation-heavy.

- `docs/plans/` contains the design decisions the pipeline is supposed to follow.
- `docs/pseudo/` contains pseudocode/spec-style notes.
- `docs/personal/` contains working notes used while building the project.

---

## Environment

The repo currently assumes:

- **Python** `>= 3.12`
- **uv** for environment management
- **PyTorch** from the configured CUDA 13.0 wheel index in `pyproject.toml`

Main Python dependencies include:

- `accelerate`
- `datasets`
- `duckdb`
- `lance`, `lancedb`
- `pyarrow`
- `tokenizers`
- `torch`
- `transformers`

---

## Getting started

```bash
uv sync
```

Most scripts in this repo are currently driven by a **user configuration block at the top of each file**, not by a polished CLI. Edit those values first, then run the relevant module.

### NamuWiki collection
```bash
uv run python -m src.collect.run_namu
```

### Build canonical parquet shards
```bash
uv run python -m src.preprocess.build_all_parquet
```

### Exact dedup
```bash
uv run python -m src.preprocess.exact_dedup
```

### Near dedup
```bash
uv run python -m src.preprocess.near_dedup
```

### Build tokenizer
```bash
uv run python -m src.tokenizer.build_tokenizer
```

### Translate selected reasoning data into Korean
```bash
uv run python -m src.translate.runner
```

### Build final Lance dataset
```bash
uv run python -m src.preprocess.build_lance
```

---

## Outputs

Typical output locations used by the current pipeline:

```text
data/korean_processed/_staging/
data/korean_processed/exact_dedup/
data/korean_processed/near_dedup/
data/tokenizers/korean_bbpe_vN/
data/korean_processed/lance_by_source/
data/korean_processed/final_lancedb/dataset.lance
```

The pipeline also writes manifests, progress files, deleted-row logs, and tokenizer metadata so intermediate decisions can be audited later.

---

## Roadmap

- [x] Collect Korean raw data from multiple sources
- [x] Normalize raw data into a canonical parquet schema
- [x] Implement exact dedup
- [x] Implement MinHash + LSH near dedup
- [x] Build a custom 48k Korean BBPE tokenizer
- [x] Build a Lance/LanceDB-ready final corpus
- [ ] Implement the 0.2B model
- [ ] Run first pretraining experiments
- [ ] Finetune at longer context length
- [ ] Explore context window extension beyond 4096
- [ ] Evaluate whether the data pipeline choices were actually worth the complexity

---

## What this repo is not

To keep expectations realistic:

- This is **not** a general-purpose LLM training framework.
- This is **not** a finished foundation model release.
- This is **not** a claim that “training from scratch” is automatically better than adapting an existing model.

It is a serious study project, but still a study project.