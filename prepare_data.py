"""
Formats the Spider dataset (question, SQL, db_id) joined with schema info
(from richardr1126/spider-schema) into prompt/completion pairs ready for
instruction fine-tuning.
"""

import json
from datasets import load_dataset

print("Loading datasets...")
spider = load_dataset("xlangai/spider")
schema_ds = load_dataset("richardr1126/spider-schema")

# Build a lookup: db_id -> schema string
schema_lookup = {
    row["db_id"]: row["Schema (values (type))"]
    for row in schema_ds["train"]
}

print(f"Loaded {len(schema_lookup)} schemas")


def format_example(ex):
    schema_text = schema_lookup.get(ex["db_id"], "")
    prompt = (
        "### Database Schema:\n"
        f"{schema_text}\n\n"
        "### Question:\n"
        f"{ex['question']}\n\n"
        "### SQL:\n"
    )
    completion = ex["query"].strip()
    return {"prompt": prompt, "completion": completion, "db_id": ex["db_id"]}


for split_name, split_data in [("train", spider["train"]), ("val", spider["validation"])]:
    formatted = []
    skipped = 0
    for ex in split_data:
        if ex["db_id"] not in schema_lookup:
            skipped += 1
            continue
        formatted.append(format_example(ex))

    out_path = f"{split_name}_formatted.jsonl"
    with open(out_path, "w") as f:
        for row in formatted:
            f.write(json.dumps(row) + "\n")

    print(f"{split_name}: wrote {len(formatted)} examples to {out_path} (skipped {skipped} with no schema match)")

print("\nSample formatted example:")
with open("train_formatted.jsonl") as f:
    print(f.readline())
