# https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv

# veri setini iceriye aktar
import pandas as pd

data = pd.read_csv("metin_siniflandirma_spam_veri_seti.csv", encoding="latin-1")
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace=True)
data.columns = ["label", "text"]

# eda: missing value 
print(data.isna().sum())

"""
text preprocessing 
    ozel karakterleri kaldir
    lowercase
    token
    remove stopwords
    lemmatize
"""

import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

text = list(data["text"])
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(text)):
    r = re.sub("[^a-zA-Z]", " ", text[i])
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words("english")]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = " ".join(r)
    corpus.append(r)

data["text2"] = corpus

# train test split: %67 egitim veri seti, %33 test veri seti
X = data["text2"]
y= data["label"]

from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

# feature extraction: bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# classifier training: model training and evaluation
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train_cv, y_train)

x_test_cv = cv.transform(x_test)

# prediction 
predictions = dt.predict(x_test_cv)
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_test, predictions)

print("Accuracy:", 100*(sum(sum(c_matrix)) - c_matrix[1,0] - c_matrix[0,1])/sum(sum(c_matrix)))























