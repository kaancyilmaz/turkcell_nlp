# import libraries
from surprise import Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split

# veri seti yukle: movielens
data = Dataset.load_builtin("ml-100k")

# train-test-split
trainset, testset = train_test_split(data, test_size=0.2) # %80 train, %20 test

# KNN Modeli tanimla ve train
sim_options = {
    "name":"cosine",
    "user_based":True } #kullancilar arasi benzerlik

model = KNNBasic(sim_options = sim_options)
model.fit(trainset)

# test verileri ile evaluation
prediction = model.test(testset)
accuracy.rmse(prediction)

# oneri yapma
def get_top_n(predictions, n = 10):
    
    top_n = {}
    
    for uid, iid, true_r, est, _ in predictions:
        if not top_n.get(uid):
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key= lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n
 
top_n = get_top_n(prediction, n=5)

user_id = "2"
print(f"Top 10 recommendation for user {user_id}")
for item_id, rating in top_n[user_id]:
    print(f"Item ID: {item_id}, score: {rating}")
























