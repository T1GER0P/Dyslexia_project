import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from tts import speak
import textstat
import pandas as pd
import datetime
import os

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(page_title="ReadRight", page_icon="📘", layout="centered")

st.title("📘 ReadRight — AI Text Simplifier for Dyslexic & Slow Learners")
st.write("""
Paste any difficult English text below,  
and ReadRight will simplify it into an easier, child-friendly version.
""")

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
MODEL_PATH = r"D:\Project(Hackathon)\ReadRight\model_finetuned_optimized"

if not os.path.isdir(MODEL_PATH):
    st.error(f"❌ Model folder not found at: {MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_model():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device), device

tokenizer, model, device = load_model()

# ---------------------------------------------------------
# SIMPLIFICATION FUNCTION
# ---------------------------------------------------------
def simplify(text):
    input_text = "simplify: " + text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            num_beams=4,
            max_length=64
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
text = st.text_area("Enter text to simplify:", height=200)

# ---------------------------------------------------------
# BUTTON — SIMPLIFY
# ---------------------------------------------------------
if st.button("Simplify Text"):
    if text.strip():

        simplified = simplify(text)

        st.subheader("🎯 Simplified Text:")
        st.success(simplified)

        # Reading difficulty
        difficulty = textstat.flesch_kincaid_grade(text)
        st.info(f"📘 Reading Difficulty Score (FKGL): **{difficulty:.2f}**")

        # Save progress
        os.makedirs("app/data", exist_ok=True)
        log = pd.DataFrame([{
            "timestamp": datetime.datetime.now(),
            "original": text,
            "simplified": simplified
        }])

        log.to_csv("app/data/progress.csv", mode="a", header=not os.path.exists("app/data/progress.csv"), index=False)

        # TTS button
        if st.button("🔊 Read Aloud"):
            st.info("📢 Reading aloud...")
            speak(simplified)

