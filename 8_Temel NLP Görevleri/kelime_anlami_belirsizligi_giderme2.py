from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk

# ornek cumle belirle
sentences = [
    "I went to the bank to deposit money.", 
    "The river bank was flooded after the heavy rain."]

word = "bank"
for sentence in sentences:
    
    print("Sentence: ",sentence)
    sense_simple = simple_lesk(sentence, word)
    print("Sense simple: ", sense_simple.definition())
    
    sense_adapted = adapted_lesk(sentence, word)
    print("Sense adapted: ", sense_adapted.definition())
    
    sense_cosine = cosine_lesk(sentence, word)
    print("Sense cosine: ", sense_cosine.definition())
    
    
"""
Sentence:  I went to the bank to deposit money.
Sense simple:  a financial institution that accepts deposits and channels the money into lending activities
Sense adapted:  a financial institution that accepts deposits and channels the money into lending activities
Sense cosine:  a container (usually with a slot in the top) for keeping money at home

Sentence:  The river bank was flooded after the heavy rain.
Sense simple:  sloping land (especially the slope beside a body of water)
Sense adapted:  sloping land (especially the slope beside a body of water)
Sense cosine:  a supply or stock held in reserve for future use (especially in emergencies)
"""
















