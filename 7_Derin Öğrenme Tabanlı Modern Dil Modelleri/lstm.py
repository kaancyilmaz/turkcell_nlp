
# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# create dataset
texts = [
    "Merhaba, nasılsınız? Ben oldukça iyiyim.",
    "Spor yaparken enerjim artıyor ve kendimi daha sağlıklı hissediyorum.",
    "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "Akşam yemeğinde ne yapsam diye düşünüyorum, belki makarna.",
    "Kitap okumak beni gerçekten rahatlatıyor.",
    "Bu hafta sonu sinemaya gitmek için plan yaptık.",
    "Sabah kahvemi içmeden kendime gelemiyorum.",
    "Yarın iş toplantım var, hazırlık yapmam gerekiyor.",
    "Yeni bir hobi edinmeyi düşünüyorum, belki resim yapmaya başlarım.",
    "Market alışverişi yapmam lazım, evde bazı malzemeler bitmiş.",
    "Evi temizlemek bugün yapmam gereken en önemli işlerden biri.",
    "Bugün arkadaşlarımla dışarıda buluşmayı planlıyorum.",
    "İş yerinde çok yoğun bir gün geçirdim, biraz dinlenmek istiyorum.",
    "Yatmadan önce biraz meditasyon yapıyorum, uykuya dalmamı kolaylaştırıyor.",
    "Bu akşam için bir dizi izlemek güzel olabilir.",
    "Sabah erken kalkıp yürüyüşe çıkmayı düşünüyorum.",
    "Yarın sınavım var, bu gece çalışmam lazım.",
    "Bugün gerçekten çok yoruldum, erken yatmayı planlıyorum.",
    "Yeni bir film izlemeyi düşünüyorum, tavsiyen var mı?",
    "Bu hafta sonu ailemi ziyaret etmeyi planlıyorum.",
    "Sabah koşusu beni çok zinde hissettiriyor.",
    "Kahvaltıda ne yesem diye düşünüyorum, belki omlet yaparım.",
    "Kitapçıda saatlerce vakit geçirmeyi seviyorum.",
    "Yarın sabah toplantım var, sunumu hazırlamalıyım.",
    "Ders çalışırken müzik dinlemeyi seviyorum, konsantrasyonumu artırıyor.",
    "Yaz tatili için bir plan yapmayı düşünüyorum.",
    "Yeni bir dil öğrenmek gerçekten ilginç bir deneyim.",
    "Evde zaman geçirmek de bazen dışarı çıkmaktan daha keyifli olabiliyor.",
    "Yemekten sonra bir fincan çay içmek çok rahatlatıcı.",
    "Bugün hava çok soğuk, dışarı çıkarken sıkı giyinmeliyim.",
    "Telefonumun şarjı bitmek üzere, şarja takmam lazım.",
    "Yarın sabah erkenden işe gitmem gerekiyor.",
    "Yeni bir restoran denemek için dışarı çıkıyoruz.",
    "Akşam yemeğinde arkadaşlarla buluşuyoruz.",
    "Bu hafta sonu evde film maratonu yapmayı düşünüyorum.",
    "Kitap kulübüne katıldım, bu ayın kitabını okuyorum.",
    "Sabahları spor yapmak enerjimi artırıyor.",
    "Çalışma masamı düzenledim, artık daha verimli çalışabilirim.",
    "Bir fincan kahveyle güne başlamak harika.",
    "Yolda yürürken bir arkadaşımı gördüm, biraz sohbet ettik.",
    "Yarın hava yağmurlu olacakmış, şemsiyemi yanıma almayı unutmamalıyım.",
    "Bugün çok fazla işim var, zaman yönetimi yapmam gerekecek.",
    "Yeni aldığım kitabı okumaya başlayacağım.",
    "Yarın sabah erkenden bir toplantıya katılmam gerekiyor.",
    "Akşam yemeğinde balık yapmayı düşünüyorum.",
    "Güneşli havada yürüyüş yapmak çok keyifli.",
    "Bu hafta sonu bir gezi planlıyoruz.",
    "Tiyatroya gitmeyi gerçekten çok özledim.",
    "Yemekten sonra biraz dinlenip televizyon izleyeceğim.",
    "Spor salonuna gitmek beni çok motive ediyor.",
    "Yeni bir tarif denemek için malzemeleri aldım.",
    "Bu akşam evde pizza yapmayı düşünüyorum.",
    "İş yerinde çok yoğun bir gün geçirdim, biraz dinlenmek iyi olacak.",
    "Bugün yürüyüş yapmak için çok güzel bir hava var.",
    "Sabah kahvaltısı benim en sevdiğim öğün.",
    "Bu hafta sonu arkadaşlarla buluşup piknik yapmayı planlıyoruz.",
    "Yeni bir diziyi izlemeye başladım, çok heyecanlı ilerliyor.",
    "Yatmadan önce biraz kitap okumayı seviyorum.",
    "Eve dönünce bir kahve yapıp dinleneceğim.",
    "Bugün uzun bir yürüyüşe çıktım, çok iyi geldi.",
    "İşlerim biraz yoğunlaştı, birkaç gün daha çalışmam gerekecek.",
    "Sabahları erken kalkıp yoga yapmayı seviyorum.",
    "Bu akşam dışarıda bir şeyler yemeği planlıyoruz.",
    "Uzun zamandır arkadaşlarımla görüşemedim, bu hafta bir araya gelmeyi planlıyoruz.",
    "Yarın için yapmam gereken bazı işler var.",
    "Kendime yeni bir çay demledim, biraz keyif yapacağım.",
    "Akşamüstü bir kahve molası vermek güzel oluyor.",
    "Bu hafta sonu evde biraz temizlik yapmam gerekiyor.",
    "Yeni bir spor salonuna yazıldım, çok heyecanlıyım.",
    "Bugün işlerimi tamamladıktan sonra biraz dinlenmek istiyorum.",
    "Hafta sonu için bir tatil planı yapmayı düşünüyoruz.",
    "Yatmadan önce biraz meditasyon yapmak iyi gelebilir.",
    "Sabahları erken kalkmayı pek sevmiyorum ama alışmaya çalışıyorum.",
    "Yarın önemli bir toplantım var, çok iyi hazırlanmalıyım.",
    "Hafta sonu için yeni bir aktivite bulmam gerekiyor.",
    "Sabahları kahvaltı yapmadan evden çıkmak istemiyorum.",
    "Bu hafta içinde biraz daha düzenli çalışmam gerekiyor.",
    "Evde spor yapmanın faydalarını yeni fark ettim.",
    "Yarın sabah erkenden yürüyüşe çıkmayı düşünüyorum.",
    "Kitap okumak benim en sevdiğim hobilerden biri.",
    "Yeni bir tarif denemek için sabırsızlanıyorum.",
    "Sabahları erken kalkmak biraz zor oluyor ama günüm daha verimli geçiyor.",
    "Bu akşam dışarıda bir şeyler yapmayı düşünüyoruz.",
    "Sınavlara çalışırken zaman yönetimi yapmam çok önemli.",
    "Evde film izlemek dışarı çıkmaktan bazen daha keyifli oluyor.",
    "Bu hafta sonu biraz doğayla iç içe olmak iyi gelecek.",
    "Yarın sabah spora gitmeyi planlıyorum.",
    "Yeni bir kitap aldım, bu hafta bitirmeyi planlıyorum.",
    "Sabah erken kalktığımda gün daha uzun ve verimli geçiyor.",
    "Evde kahvaltı yapmayı dışarıda kahvaltıya tercih ediyorum.",
    "Yarın yapılacak işler listemi hazırladım.",
    "Bu hafta sonu biraz dinlenmeye ihtiyacım var.",
    "Yeni bir dil öğrenmeye başlamak çok heyecan verici.",
    "Bugün dışarı çıkıp biraz yürüyüş yapmayı düşünüyorum.",
    "Spor yaparken zamanın nasıl geçtiğini anlamıyorum.",
    "Yarın sabah erkenden toplantım var, ona hazırlanıyorum."
]

# tokenizer and dizilerin hazirlanmasi
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1 

# metinleri siralayalim ve padding islemi uygulayalim
input_sequences = []
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding = "pre")

X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# create lstm model 
model = Sequential()
model.add(Embedding(total_words, 50, input_length = X.shape[1]))
model.add(LSTM(100, return_sequences = False))
model.add(Dense(total_words, activation = "softmax"))

model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# train lstm model
model.fit(X, y, epochs = 100, verbose = 1)

# evaluation ve metin tamamlama calismasi
def predict_next_word(seed_text, next_words):
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen= max_sequence_length-1, padding = "pre")
        predicted_probs = model.predict(token_list, verbose = 0)
        predicted_word_index = np.argmax(predicted_probs, axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text = seed_text + " " + predicted_word
    
    return seed_text

seed_text = "Kahvaltıda deniz kenarı"
print(predict_next_word(seed_text, 5))





































