from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# ornek veri seti
corpus = [
    "I love you", 
    "I love apple", 
    "I love programming",
    "You love me",
    "She loves apple",
    "They love you",
    "I love you and you love me"
    ]

# tokenize
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

# n-gram -> n:2
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))
    
# bigrams frekans counter
bigrams_freq = Counter(bigrams)

# n-gram -> n:3
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

# trigrams frekans counter
trigrams_freq = Counter(trigrams)

# "I love" bigramından sonra "you" veya "apple" gelme olasiliklarini hesapla
bigram = ("i", "love")
prob_you = trigrams_freq[("i", "love", "you")]/bigrams_freq[bigram]
prob_apple = trigrams_freq[("i", "love", "apple")]/bigrams_freq[bigram]

print("you kelimesinin olma olasiligi: ", prob_you)
print("apple kelimesinin olma olasiligi: ", prob_apple)



















