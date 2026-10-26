# NL-to-SQL Fine-Tuning with QLoRA (Qwen2.5-1.5B)

Fine-tunes a small open-weight language model (Qwen2.5-1.5B) to convert natural-language
questions into SQL queries, using QLoRA so the whole pipeline runs on a 4GB laptop GPU
(RTX 3050). Trained and evaluated on the [Spider](https://yale-lily.github.io/spider)
text-to-SQL benchmark.

## What this project shows

- Joining and cleaning two mismatched data sources into a single training format
- Memory-constrained ML engineering: 4-bit quantization, LoRA, gradient accumulation,
  paged optimizers — all the tricks needed to fine-tune on consumer hardware
- A baseline-vs-fine-tuned evaluation methodology, not just a single accuracy number
- Debugging real training issues (dtype mismatches, OOM risk) rather than a scripted demo

## Results

| Model | Exact-Match Accuracy | Execution Accuracy |
|---|---|---|
| Base Qwen2.5-1.5B (zero-shot prompting) | 0% |  |
| Qwen2.5-1.5B + QLoRA fine-tuned ||  |

> The base model's SQL logic was often already correct, but it didn't follow Spider's exact
> formatting conventions (casing, semicolons, column qualifiers), which is what fine-tuning
> was expected to fix. See `baseline_results.json` for the full baseline output and
> per-example predictions.

## Task

Given a database schema and an English question, generate the correct SQL query.

**Example prompt:**
```
### Database Schema:
department : Department_ID (number), Name (text), Creation (text), ...
head: head_ID (number), name (text), born_state (text), age (number)
management : department_ID (number), head_ID (number), temporary_acting (text)

### Question:
How many heads of the departments are older than 56?

### SQL:
```

**Target completion:**
```sql
SELECT count(*) FROM head WHERE age > 56
```

## Approach

- **Base model**: [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), loaded in
  4-bit (NF4) via `bitsandbytes`
- **Fine-tuning method**: QLoRA — LoRA adapters (r=16, alpha=32) on the attention
  projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), base weights frozen and quantized
- **Data**: [Spider](https://huggingface.co/datasets/xlangai/spider) (7,000 train / 1,034
  validation question-SQL pairs across 166 schemas), joined with schema definitions from
  [richardr1126/spider-schema](https://huggingface.co/datasets/richardr1126/spider-schema)
- **Training**: `trl`'s `SFTTrainer`, bf16 precision, `paged_adamw_8bit` optimizer,
  effective batch size 16 (batch 2 × grad accumulation 8), 3 epochs
- **Hardware**: single RTX 3050 Laptop GPU (4GB VRAM)

## Repo structure

```
prepare_data.py       # joins Spider questions/SQL with schema data, writes formatted jsonl
baseline_eval.py       # evaluates the base (non-fine-tuned) model on validation examples
train_smoke_test.py    # quick sanity check (50 examples) before committing to full training
train.py               # full QLoRA fine-tuning run
baseline_results.json  # baseline evaluation output (predictions + accuracy)
```

Note: formatted training/validation data (`train_formatted.jsonl`, `val_formatted.jsonl`) and
the trained LoRA adapter are not committed — regenerate them with the steps below.

## Reproducing this

```bash
# 1. Environment
conda create -n nl2sql python=3.11 -y
conda activate nl2sql
pip install -r requirements.txt

# 2. Authenticate with Hugging Face (needed to download the base model)
hf auth login

# 3. Build the training data
python prepare_data.py

# 4. (Optional but recommended) sanity-check the training loop on a tiny slice
python train_smoke_test.py

# 5. Establish the baseline (pre-fine-tuning) numbers
python baseline_eval.py

# 6. Fine-tune
python train.py
```

Training produces a LoRA adapter in `./qwen-nl2sql-lora/`, which can be loaded on top of the
base model for inference.

## Key engineering decisions

- **4-bit quantization + LoRA (QLoRA)** instead of full fine-tuning — a 1.5B model fine-tuned
  in full precision would not fit in 4GB VRAM; QLoRA reduces the trainable footprint to a
  small fraction of total parameters while keeping the frozen base model in 4-bit.
- **bf16 over fp16** — fp16 mixed-precision training requires a gradient scaler that doesn't
  support bfloat16 tensors, which caused an early crash. Ampere GPUs (including the RTX 3050)
  support bf16 natively, which also removes the need for a gradient scaler entirely.
- **`paged_adamw_8bit`** — keeps optimizer state memory low enough to fit alongside the model
  and activations on a 4GB card.
- **Exact-match *and* execution accuracy** as complementary metrics — exact-match alone
  under-reports a model that reasons correctly but formats differently than the reference.

## Possible extensions

- Quantize the fine-tuned model (e.g. GGUF via `llama.cpp`) and benchmark inference
  latency/memory versus the unquantized version
- Serve the model behind a small FastAPI endpoint
- Ablate LoRA rank (8 vs. 16 vs. 32) and target modules (attention-only vs. attention+MLP)
  with a results table
