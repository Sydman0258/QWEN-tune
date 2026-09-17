"""
Runs the base (non-fine-tuned) model on a sample of validation examples
to establish a baseline before fine-tuning.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
NUM_SAMPLES = 20  # keep small for a quick first pass

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()

# Load validation examples
examples = []
with open("val_formatted.jsonl") as f:
    for line in f:
        examples.append(json.loads(line))

examples = examples[:NUM_SAMPLES]

results = []
exact_match_count = 0

for i, ex in enumerate(examples):
    inputs = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    predicted_sql = generated.strip().split("\n")[0]  # take first line only

    is_exact_match = predicted_sql.strip().lower() == ex["completion"].strip().lower()
    if is_exact_match:
        exact_match_count += 1

    results.append({
        "question": ex["prompt"].split("### Question:\n")[1].split("\n\n### SQL:")[0],
        "gold_sql": ex["completion"],
        "predicted_sql": predicted_sql,
        "exact_match": is_exact_match,
    })

    print(f"[{i+1}/{len(examples)}] Match: {is_exact_match}")
    print(f"  Gold:      {ex['completion']}")
    print(f"  Predicted: {predicted_sql}")
    print()

accuracy = exact_match_count / len(examples)
print(f"\n=== BASELINE RESULTS ===")
print(f"Exact match accuracy: {accuracy:.2%} ({exact_match_count}/{len(examples)})")

with open("baseline_results.json", "w") as f:
    json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
