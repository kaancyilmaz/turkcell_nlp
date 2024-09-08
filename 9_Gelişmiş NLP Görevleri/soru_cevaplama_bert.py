from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

# squad veri seti uzerinde ince ayar (fine-tuning) yapilmis bert dil modeli

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def predict_answer(context, question):
    
    # token
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    
    # giris tensorlar
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # modeli calistir ve skorlari al
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict = False)
        
    # en yuksek olasilikli baslangic vebitis indeksleri bul
    start_index = torch.argmax(start_scores, dim=1).item()
    end_index = torch.argmax(end_scores, dim=1).item()
    
    # tokenlari al ve cevabi coz
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index + 1])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer
    
question = "What is the capital of France"
context = "France, officially the French Republic, is a country whose capital is Paris"   

# question = '''What is Machine Learning?'''
# context = ''' Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance 
#                 on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or 
#                 decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection 
#                 of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning 
#                 is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, 
#                 theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory 
#                 data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics. '''

answer = predict_answer(context, question) 

print(question)
print(answer)
    
"""
What is the capital of France
paris
"""    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    