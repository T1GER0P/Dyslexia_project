from gtts import gTTS
import os
import uuid
from playsound import playsound

def speak(text):
    # Create unique audio file
    filename = f"tts_{uuid.uuid4().hex}.mp3"

    # Generate speech
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

    # Play audio
    playsound(filename)

    # Delete after playing
    os.remove(filename)
