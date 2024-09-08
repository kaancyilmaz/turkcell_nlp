from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-fr" # eng to fr
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Hello, what is your name?"

# metin encode, modele input olarak veririz
translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))

# ceviri yapilir, string'e donusturulur
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print(f"Translated text: {translated_text}")


# https://huggingface.co/Helsinki-NLP?sort_models=downloads#models