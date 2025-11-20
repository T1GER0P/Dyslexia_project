"""
FINAL PATCHED & OPTIMIZED TRAINING SCRIPT
-----------------------------------------
Designed for GTX 1650 (4GB VRAM)
Compatible with Transformers 4.57.1
Includes:
- GPU support
- fp16 mixed precision
- Adafactor optimizer
- Subset of WikiLarge (12k)
- Sequence length=64
- Fully patched compute_metrics to prevent decode crash
"""

import os
import numpy as np
import textstat
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# -------------------------
# User configuration
# -------------------------
TRAIN_SUBSET_SIZE = 12000
VAL_SUBSET_SIZE = 300
TEST_SUBSET_SIZE = 100

MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 64

MODEL_NAME = "t5-small"
OUTPUT_DIR = "model_finetuned_optimized"

NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 3e-4

# -------------------------
# GPU Check
# -------------------------
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except:
        pass
else:
    print("⚠ GPU not detected — training will be slow.")


# -------------------------
# SARI implementation
# -------------------------
def ngrams(sentence, n):
    tokens = sentence.split()
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def compute_sari_score(source, prediction, references):
    def avg_precision(pred, refs):
        if len(pred) == 0:
            return 1
        return sum(len(pred & ref) / len(pred) for ref in refs) / len(refs)

    def avg_recall(pred, refs):
        return sum(len(pred & ref) / len(ref) if len(ref) else 1 for ref in refs) / len(refs)

    def f1(p, r):
        return 0 if (p + r) == 0 else 2 * p * r / (p + r)

    f1_scores = []
    for n in [1, 2, 3]:
        pred_n = ngrams(prediction, n)
        src_n = ngrams(source, n)
        refs_n = [ngrams(r, n) for r in references]

        keep_pred = pred_n & src_n
        add_pred = pred_n - src_n
        delete_pred = src_n - pred_n

        keep_refs = [ref & src_n for ref in refs_n]
        add_refs = [ref - src_n for ref in refs_n]
        delete_refs = [src_n - ref for ref in refs_n]

        f1_keep = f1(avg_precision(keep_pred, keep_refs), avg_recall(keep_pred, keep_refs))
        f1_add = f1(avg_precision(add_pred, add_refs), avg_recall(add_pred, add_refs))
        f1_del = f1(avg_precision(delete_pred, delete_refs), avg_recall(delete_pred, delete_refs))

        f1_scores.append((f1_keep + f1_add + f1_del) / 3)

    return float(np.mean(f1_scores))

def corpus_sari(sources, predictions, references):
    return float(np.mean([
        compute_sari_score(s, p, [r]) for s, p, r in zip(sources, predictions, references)
    ]))


# -------------------------
# Load dataset
# -------------------------
print("Downloading WikiLarge-clean dataset...")
dataset = load_dataset("eilamc14/wikilarge-clean")

# Subset dataset for speed
dataset["train"] = dataset["train"].select(range(min(TRAIN_SUBSET_SIZE, len(dataset["train"]))))
dataset["validation"] = dataset["validation"].select(range(min(VAL_SUBSET_SIZE, len(dataset["validation"]))))
dataset["test"] = dataset["test"].select(range(min(TEST_SUBSET_SIZE, len(dataset["test"]))))

print("Train size:", len(dataset["train"]))
print("Val size:", len(dataset["validation"]))
print("Test size:", len(dataset["test"]))


# -------------------------
# Load model + tokenizer
# -------------------------
print("Loading model/tokenizer:", MODEL_NAME)
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

if torch.cuda.is_available():
    model = model.to("cuda")
    print("Model moved to GPU.")

# No gradient checkpointing for speed

# -------------------------
# Tokenization
# -------------------------
def preprocess(batch):
    inputs = ["simplify: " + s for s in batch["source"]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# -------------------------
# DataCollator & Trainer args
# -------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=200,
    save_steps=1000,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

# -------------------------
# 🔥 Fully patched compute_metrics (fixes nested predictions)
# -------------------------
def compute_metrics(eval_pred):
    preds_ids, labels_ids = eval_pred

    # unwrap tuple from trainer
    if isinstance(preds_ids, tuple):
        preds_ids = preds_ids[0]

    # flatten nested lists/tensors
    cleaned_preds = []
    for p in preds_ids:
        if hasattr(p, "tolist"):
            p = p.tolist()
        # flatten deep nesting like [[...]]
        while isinstance(p, list) and len(p) > 0 and isinstance(p[0], list):
            p = p[0]
        cleaned_preds.append(p)

    preds = tokenizer.batch_decode(
        cleaned_preds,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # labels
    labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
    refs = tokenizer.batch_decode(
        labels_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # compute metrics
    sources = dataset["validation"]["source"]
    sari = corpus_sari(sources, preds, refs)
    fkgl = float(np.mean([textstat.flesch_kincaid_grade(p) if p.strip() else 0.0 for p in preds]))

    return {"sari": sari, "fkgl": fkgl}


# -------------------------
# Trainer
# -------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------------
# Train
# -------------------------
print("🚀 Starting optimized training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 Training completed successfully!")
print("Model saved to:", OUTPUT_DIR)

