import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import os

# 🔥 IMPORTANT: USE ABSOLUTE PATH TO YOUR LOCAL MODEL
MODEL_PATH = r"D:\Project(Hackathon)\ReadRight\model_finetuned_optimized"

# Ensure folder exists
if not os.path.isdir(MODEL_PATH):
    raise ValueError(f"Model folder not found at: {MODEL_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)

def simplify_text(text):
    input_text = "simplify: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample = "The weather was extremely unpredictable, making it difficult to plan outdoor activities."
    print("Simplified:", simplify_text(sample))
