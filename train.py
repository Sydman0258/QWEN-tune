"""
Fine-tunes Qwen2.5-1.5B with QLoRA on the Spider text-to-SQL dataset.
Designed to run on a 4GB VRAM laptop GPU (RTX 3050).
"""

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = "./qwen-nl2sql-lora"

# ---- 1. Load data ----
print("Loading formatted data...")
dataset = load_dataset(
    "json",
    data_files={"train": "train_formatted.jsonl", "validation": "val_formatted.jsonl"},
)

def to_text(ex):
    return {"text": ex["prompt"] + ex["completion"] + "<|endoftext|>"}

dataset = dataset.map(to_text)
print(dataset)

# ---- 2. Load base model in 4-bit ----
print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# ---- 3. Attach LoRA adapters ----
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---- 4. Training config ----
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    per_device_eval_batch_size=2,
    bf16=True,
    optim="paged_adamw_8bit",
    max_length=512,
    report_to="none",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

# ---- 5. Train ----
print("Starting training...")
trainer.train()

# ---- 6. Save the LoRA adapter ----
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nDone. LoRA adapter saved to {OUTPUT_DIR}")
