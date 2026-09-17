"""
Smoke test: trains on a small slice of data for a few steps to confirm
the pipeline runs without OOM before launching the full training job.
"""

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
OUTPUT_DIR = "./smoke_test_output"

print("Loading formatted data (small slice)...")
dataset = load_dataset(
    "json",
    data_files={"train": "train_formatted.jsonl", "validation": "val_formatted.jsonl"},
)

# Take just 50 train examples and 10 val examples for the smoke test
dataset["train"] = dataset["train"].select(range(50))
dataset["validation"] = dataset["validation"].select(range(10))

def to_text(ex):
    return {"text": ex["prompt"] + ex["completion"] + "<|endoftext|>"}

dataset = dataset.map(to_text)

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

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=1,          # log every step so we see it's actually moving
    save_strategy="no",       # don't bother saving checkpoints for the smoke test
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

print("Starting smoke test training (should take ~1-2 minutes)...")
trainer.train()

print("\n✅ SMOKE TEST PASSED — no OOM, training loop runs correctly.")
print("Peak GPU memory used (GB):", torch.cuda.max_memory_allocated() / 1e9)
