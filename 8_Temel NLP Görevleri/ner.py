
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

content = "John works at Microsoft and lives in New York. He visited the National History Museum."

doc = nlp(content)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]
df = pd.DataFrame(entities, columns=["text", "type","lemma"])