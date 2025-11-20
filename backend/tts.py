from gtts import gTTS
import uuid
import os

def generate_audio(text):
    filename = f"assets/audio/{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename
