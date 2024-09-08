from nltk.classify import MaxentClassifier

# egitim veri seti
train_data = [
    ({"love":True, "amazing":True}, "positive"), 
    ({"hate":True, "terrible":True}, "negative"),
    ({"happy":True, "joy":True}, "positive"),
    ({"sad":True, "depressed":True}, "negative")]

# max ent classifier training
classifier = MaxentClassifier.train(train_data, max_iter = 10)

# test
test_sentence = "I do not like this movie"
features = {word: (word in test_sentence.lower().split()) for word in ["love", "amazing", "hate", "terrible","happy", "joy","sad", "depressed"]}

label = classifier.classify(features)
print(label)