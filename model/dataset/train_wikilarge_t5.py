"""
Final working script for fine-tuning T5 on WikiLarge-clean.
Compatible with Transformers 4.57.1
No easse required (custom SARI implementation included)
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

# ============================================================
# CUSTOM SARI IMPLEMENTATION (NO EASSE NEEDED)
# ============================================================

def ngrams(sentence, n):
    tokens = sentence.split()
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def compute_sari_score(source, prediction, references):
    """
    Minimal SARI calculation for simplification evaluation.
    """
    def avg_precision(pred, refs):
        if len(pred) == 0:
            return 1
        return sum(len(pred & ref) / len(pred) for ref in refs) / len(refs)

    def avg_recall(pred, refs):
        return sum(
            len(pred & ref) / len(ref) if len(ref) > 0 else 1
            for ref in refs
        ) / len(refs)

    def f1(p, r):
        return 0 if (p + r) == 0 else 2 * p * r / (p + r)

    f1_scores = []

    for n in [1, 2, 3]:
        pred_n = ngrams(prediction, n)
        src_n = ngrams(source, n)
        refs_n = [ngrams(r, n) for r in references]

        # Keep, Add, Delete
        keep_pred = pred_n & src_n
        add_pred = pred_n - src_n
        delete_pred = src_n - pred_n

        keep_refs = [ref & src_n for ref in refs_n]
        add_refs = [ref - src_n for ref in refs_n]
        delete_refs = [src_n - ref for ref in refs_n]

        f1_keep = f1(avg_precision(keep_pred, keep_refs), avg_recall(keep_pred, keep_refs))
        f1_add = f1(avg_precision(add_pred, add_refs), avg_recall(add_pred, add_refs))
        f1_delete = f1(avg_precision(delete_pred, delete_refs), avg_recall(delete_pred, delete_refs))

        sari_n = (f1_keep + f1_add + f1_delete) / 3
        f1_scores.append(sari_n)

    return sum(f1_scores) / 3

def corpus_sari(sources, predictions, references):
    return np.mean([
        compute_sari_score(src, pred, [ref])
        for src, pred, ref in zip(sources, predictions, references)
    ])


# ============================================================
# LOAD DATASET
# ============================================================

print("Downloading WikiLarge-clean dataset...")
dataset = load_dataset("eilamc14/wikilarge-clean")

train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

print("Sample loaded:", train_data[0])


# ============================================================
# LOAD MODEL + TOKENIZER
# ============================================================

tokenizer = T5TokenizerFast.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))


# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def preprocess(batch):
    inputs = ["simplify: " + s for s in batch["source"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target"], max_length=256, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)


# ============================================================
# TRAINING ARGUMENTS (UPDATED FOR TRANSFORMERS 4.57)
# ============================================================

training_args = Seq2SeqTrainingArguments(
    output_dir="model_finetuned",
    eval_strategy="steps",     # updated argument name
    eval_steps=500,
    logging_steps=500,
    save_steps=500,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    learning_rate=3e-4,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # use GPU fp16 if available
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):
    preds_ids, labels_ids = eval_pred

    preds = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)

    labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
    refs = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    sources = val_data["source"]

    sari = corpus_sari(sources, preds, refs)
    fkgl = np.mean([textstat.flesch_kincaid_grade(p) for p in preds])

    return {
        "sari": sari,
        "fkgl": fkgl
    }


# ============================================================
# TRAINER
# ============================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("Training started...")
trainer.train()
trainer.save_model("model_finetuned")
tokenizer.save_pretrained("model_finetuned")

print("Training complete! Model saved to model_finetuned/")
