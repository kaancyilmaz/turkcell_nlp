from transformers import GPT2LMHeadModel, GPT2Tokenizer

# modeli ve tokenizeri yukle
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# ornek baslangic metni
text = "I go to swim for"

# tokenizasyon
inputs = tokenizer.encode(text, return_tensors="pt")

# metin uretimi
outputs = model.generate(inputs, max_length = 15)

# sonucu decode etmemiz lazim
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

