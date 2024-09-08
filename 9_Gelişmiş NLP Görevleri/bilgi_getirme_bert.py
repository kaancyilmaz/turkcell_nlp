from transformers import BertTokenizer, BertModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# bert model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Belgeler ve sorgu
documents = [
    "Machine learning is a field of artificial intelligence.",
    "Natural language processing involves understanding human language.",
    "Artificial intelligence encompasses machine learning and natural language processing.",
    "Deep learning is a subset of machine learning.",
    "Data science combines statistics, data analysis, and machine learning."
]
query = "What is machine learning?"

def get_embedding(text):
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state
    
    embedding = last_hidden_state.mean(dim=1)
    
    return embedding.detach().numpy()

# belgeler ve sorgu icin embedding vektorlerini al
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])
query_embedding = get_embedding(query)

# cosine sim
similarities = cosine_similarity(query_embedding, doc_embeddings)

for i, score in enumerate(similarities[0]):
    print(f"Document {i+1}: {score}")

most_similar_index = similarities.argmax()

print("Most similar document")
print(documents[most_similar_index])

"""
Most similar document
Deep learning is a subset of machine learning.
"""




















