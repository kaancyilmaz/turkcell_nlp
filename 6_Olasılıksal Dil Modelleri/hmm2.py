import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

# gerekli veri paketini indir
nltk.download("conll2000")

# conll veri seti yukle
train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")

# hmm training
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# test
test_sentence = "I like going to park".split()
tags = hmm_tagger.tag(test_sentence)