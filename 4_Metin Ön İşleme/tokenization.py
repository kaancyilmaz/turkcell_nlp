import nltk

nltk.download("punkt")

text = "Hello, World! How are you? Hi ..."

# kelimeleri tokenlara ayir
word_tokens = nltk.word_tokenize(text)

# cumle tokenization
sentence_tokens = nltk.sent_tokenize(text)
