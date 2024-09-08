import nltk
from nltk.wsd import lesk

# gerekli nltk paketlerini indir
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")

# ilk cumle
sentence1 = "I went to the bank to deposit money"
word1 = "bank"
sense1 = lesk(nltk.word_tokenize(sentence1), word1)

print("Sentence: ",sentence1)
print("Word: ",word1)
print("Sense: ", sense1.definition())

"""
Sentence:  I went to the bank to deposit money
Word:  bank
Sense:  a container (usually with a slot in the top) for keeping money at home
"""

#ikinci cumle
sentence2 = "The river bank was flooded after the heavy rain."
word2 = "bank"
sense2 = lesk(nltk.word_tokenize(sentence2), word2)
 
print("Sentence: ",sentence2)
print("Word: ",word2)
print("Sense: ", sense2.definition())

"""
Sentence:  The river bank was flooded after the heavy rain.
Word:  bank
Sense:  a container (usually with a slot in the top) for keeping money at home
"""
























