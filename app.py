import streamlit as st
from backend.simplify import simplify_text
from backend.tts import generate_audio
from backend.readability import readability_score

st.title("ReadRight — AI Reading Assistant")

text = st.text_area("Enter Text to Simplify:")

if st.button("Simplify"):
    simplified = simplify_text(text)
    score = readability_score(simplified)

    st.subheader("Simplified Text")
    st.write(simplified)

    st.subheader("Readability Score (lower = easier)")
    st.write(score)

    st.session_state.simplified = simplified

if "simplified" in st.session_state:
    if st.button("Read Aloud"):
        audio_file = generate_audio(st.session_state.simplified)
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes)
