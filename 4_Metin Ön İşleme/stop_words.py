import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

# stop word liste yukle
stop_words_eng = set(stopwords.words("english"))

# ornek metin
text = "This is an example of removing stop words from a text document."
filtered_words = [word for word in text.split() if word.lower() not in stop_words_eng]
print("filtered_words: ",filtered_words)

# stop word liste yukle
stop_words_tr = set(stopwords.words("turkish"))

# ornek metin turkce
text = "merhaba dünya ve bu güzel insanlar"
filtered_words = [word for word in text.split() if word.lower() not in stop_words_tr]
print("filtered_words: ",filtered_words)

# %%

turkish_stopwords = set(["ve", "bir", "bu", "ile", "için"])

# ornek metin
text = "Bu bir örnek metin ve stop words'leri temizlemek için kullanılıyor."
filtered_words = [word for word in text.split() if word.lower() not in turkish_stopwords]
print("filtered_words: ",filtered_words)





























