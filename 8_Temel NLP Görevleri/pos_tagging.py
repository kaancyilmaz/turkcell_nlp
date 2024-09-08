import spacy

nlp = spacy.load("en_core_web_sm")

sentence1 = "What is the weather like today"
doc1 = nlp(sentence1)
for token in doc1:
    print(token.text, token.pos_)
    
sentence2 = "I went to the store, but they were closed, so I had to go to another store"
doc2 = nlp(sentence2)
for token in doc2:
    print(token.text, token.pos_)