from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load your trained model later:
MODEL_PATH = "model/fine_tuned"  # use this after training

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def simplify_text(text):
    prompt = "simplify: " + text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=256, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
