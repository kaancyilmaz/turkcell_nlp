import nltk
nltk.download("wordnet")

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# ornek kelimeler
words = ["running", "runner", "ran", "runs", "better", "go", "went"]

stems = [stemmer.stem(w) for w in words]

print("Stem result: ",stems)

# %% lemma

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# ornek kelimeler
words = ["running", "runner", "ran", "runs", "better", "go", "went"]

lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words]

print("Lemma result: ",lemmas)





















