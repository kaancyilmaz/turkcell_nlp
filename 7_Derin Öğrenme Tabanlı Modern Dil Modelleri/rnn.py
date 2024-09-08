# import library
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# create dataset
data = {
    'text': [
        'Otel çok temiz ve rahattı, çok memnun kaldık.',
        'Personel güler yüzlüydü, tekrar geleceğiz.',
        'Oda oldukça küçük ve rahatsızdı, memnun kalmadım.',
        'Yemekler harikaydı, odalar çok temizdi.',
        'Resepsiyonda çok bekledik, hizmet kötüydü.',
        'Konum harika, personel çok ilgiliydi.',
        'Fiyatlar çok yüksek, hizmet vasat.',
        'Odalar ses geçiriyor, hiç memnun kalmadım.',
        'Otelin konumu çok iyiydi, her yere yakındı.',
        'Temizlik konusunda çok daha iyi olabilirdi.',
        'Yataklar çok rahatsızdı, uyuyamadık.',
        'Kahvaltı çok çeşitiydi ve lezzetliydi.',
        'Personel çok kaba davrandı, hiç memnun kalmadık.',
        'Odada klima bozuktu, çok sıcak oldu.',
        'Konum mükemmel, personel ilgili ve yardımsever.',
        'Fiyat performans dengesi çok iyiydi.',
        'Odalar çok geniş ve ferah, memnun kaldık.',
        'Otelin havuzu çok küçük ve kalabalıktı.',
        'Personel çok anlayışlıydı, her sorunumuzu çözdüler.',
        'Odaların temizliği yetersizdi.',
        'Manzara müthişti, odada zaman geçirmesi çok keyifliydi.',
        'Banyodaki su çok soğuktu, sorun yaşadık.',
        'Otel çok gürültülüydü, sakinlik arayanlara göre değil.',
        'Çalışanlar profesyonel ve güler yüzlüydü.',
        'Wifi bağlantısı çok kötüydü, internete bağlanamadık.',
        'Yemekler çok kötüydü, hiç lezzetli değildi.',
        'Odalar oldukça temizdi, hizmet kaliteliydi.',
        'Restoran bölümü çok kalabalıktı, uzun süre bekledik.',
        'Konum mükemmeldi, her yere yürüyerek ulaşabildik.',
        'Odada çay kahve ikramı vardı, bu çok hoşumuza gitti.',
        'Personel çok yavaştı, hizmet almak zor oldu.',
        'Otelden memnun kaldık, tekrar gelmeyi düşünüyoruz.',
        'Odada minibar çalışmıyordu, bu çok kötüydü.',
        'Oda servisi hızlı ve eksiksizdi.',
        'Otelin dekorasyonu çok eski, yenilenmesi gerekiyor.',
        'Konforlu bir konaklama geçirdik, çok memnun kaldık.',
        'Odaların ses yalıtımı çok kötüydü, her şeyi duyduk.',
        'Otel genel olarak güzel ancak hizmet eksik.',
        'Kahvaltı oldukça zengindi, her şey taze ve lezzetliydi.',
        'Yataklar çok sertti, rahat edemedik.',
        'Otel oldukça modern ve temizdi.',
        'Resepsiyondaki personel çok yardımcı oldu.',
        'Odada sıcak su problemi vardı, rahatsız olduk.',
        'Otelin konumu harika, her yere yakın.',
        'Yemekler oldukça lezzetliydi, çok beğendik.',
        'Personel çok saygılı ve güler yüzlüydü.',
        'Odalar temizdi ancak biraz küçük.',
        'Otelin fiyatları çok yüksekti, verilen hizmete göre pahalı.',
        'Kaldığımız oda çok konforluydu, güzel bir deneyimdi.',
        'Otel çok kalabalıktı, dinlenme fırsatı bulamadık.',
        'Yemekler çok tazeydi ve lezzetliydi.',
        'Odada terlik eksikti, istemek zorunda kaldık.',
        'Otelin restoranı çok pahalıydı, dışarıda yemek yedik.',
        'Konaklamamız boyunca personel çok ilgiliydi.',
        'Otelin odaları oldukça temizdi.',
        'Odada havlu eksikti, çok bekledik.',
        'Manzara harikaydı, odalar geniş ve ferahtı.',
        'Personel çok yavaştı, servis gecikiyordu.',
        'Oda çok küçüktü ve penceresizdi, memnun kalmadık.',
        'Otelin bahçesi çok güzeldi, keyifli bir ortam vardı.',
        'Oda konforluydu ama biraz eskiydi.',
        'Otel genel olarak güzeldi ama bazı eksiklikler vardı.',
        'Personel çok sıcakkanlıydı, her konuda yardımcı oldular.',
        'Otelin spa merkezi harikaydı, rahatladık.',
        'Fiyatlar çok yüksekti, beklentilerimizi karşılamadı.',
        'Otel personeli çok saygısızdı, tekrar gelmem.',
        'Odada sıcak su akmıyordu, çok rahatsız olduk.',
        'Otelin yatakları çok rahatsızdı, uyumakta zorlandık.',
        'Odada televizyon çalışmıyordu, teknik destek gecikti.',
        'Otelin temizliği iyiydi ancak personel ilgisizdi.',
        'Kahvaltı çok zayıftı, seçenekler azdı.',
        'Otelin konumu çok merkeziydi, her yere yakındı.',
        'Oda genişti ancak mobilyalar eskiydi.',
        'Personel ilgisizdi, sorunlarımıza çözüm bulmadılar.',
        'Odada böcek gördük, hijyen konusunda endişeliyiz.',
        'Otelin restoranı çok kaliteli, yemekler mükemmeldi.',
        'Personel çok saygılıydı, güzel bir deneyimdi.',
        'Yatak çok rahattı, deliksiz uyuduk.',
        'Oda servisi çok hızlıydı, bekletmediler.',
        'Otelin internet bağlantısı çok kötüydü.',
        'Personel çok yardımseverdi, otelden memnun kaldık.',
        'Oda manzarası harikaydı, tekrar geleceğiz.',
        'Otel oldukça eskiydi, yenilenmesi gerekiyor.',
        'Fiyatlar uygundu, hizmet yeterliydi.',
        'Yemekler oldukça lezzetli ve doyurucuydu.',
        'Oda çok gürültülüydü, rahatsız olduk.',
        'Personel çok deneyimli ve güler yüzlüydü.',
        'Otelin konumu kötüydü, her yere uzak.',
        'Fiyat performans açısından çok iyiydi.',
        'Personel çok ilgisizdi, hizmet yetersizdi.',
        'Odalar çok küçük ve konforsuzdu.',
        'Yemekler güzeldi ama seçenek azdı.',
        'Otelin havuzu kirliydi, kullanamadık.',
        'Kahvaltı oldukça iyiydi, taze ve lezzetliydi.',
        'Oda manzarası müthişti, otelden çok memnun kaldık.',
        'Otel genel olarak güzeldi ama odalar biraz küçük.',
        'Temizlik personeli işini iyi yapmıyordu, odalar kirliydi.',
        'Fiyatlar çok yüksekti, karşılığını alamadık.',
        'Otel personeli çok cana yakındı, güzel bir tatil oldu.',
        'Odada klima çok gürültülüydü, rahatsız olduk.'
    ],
    'label': [
        'pozitif', 'pozitif', 'negatif', 'pozitif', 'negatif', 'pozitif', 'negatif', 'negatif', 'pozitif', 'negatif',
        'negatif', 'pozitif', 'negatif', 'negatif', 'pozitif', 'pozitif', 'pozitif', 'negatif', 'pozitif', 'negatif',
        'pozitif', 'negatif', 'negatif', 'pozitif', 'negatif', 'negatif', 'pozitif', 'pozitif', 'pozitif', 'negatif',
        'pozitif', 'negatif', 'pozitif', 'negatif', 'negatif', 'pozitif', 'negatif', 'pozitif', 'pozitif', 'negatif',
        'pozitif', 'pozitif', 'negatif', 'pozitif', 'pozitif', 'pozitif', 'negatif', 'pozitif', 'negatif', 'pozitif',
        'pozitif', 'negatif', 'negatif', 'pozitif', 'negatif', 'negatif', 'pozitif', 'pozitif', 'pozitif', 'pozitif',
        'negatif', 'negatif', 'pozitif', 'pozitif', 'negatif', 'negatif', 'negatif', 'pozitif', 'pozitif', 'negatif',
        'negatif', 'pozitif', 'pozitif', 'negatif', 'pozitif', 'pozitif', 'pozitif', 'pozitif', 'negatif', 'pozitif',
        'negatif', 'pozitif', 'pozitif', 'pozitif', 'pozitif', 'pozitif', 'negatif', 'pozitif', 'pozitif', 'negatif',
        'pozitif', 'negatif', 'pozitif', 'negatif', 'pozitif', 'negatif', 'pozitif', 'negatif', 'pozitif', 'negatif'
    ]
}

df = pd.DataFrame(data)

# metin verisi tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index
print("Vocab size: ", len(word_index))

# padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=maxlen)
print("X shape: ",X.shape)

# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
print("Y shape: ", y.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

# word embedding: word2vec, embedding matrisi olusturma
sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]
        
# built RNN model
model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(SimpleRNN(100, return_sequences=False))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train RNN model
model.fit(X_train, y_train, epochs = 10, batch_size = 2, validation_data=(X_test, y_test))

# evaluate RNN model
print(" ")
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss: ", loss)
print("Test accuracy: ", accuracy)

# cumle sinifilandirma calismasi
def classify_sentence(sentence):
    
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen=maxlen)
    
    prediction = model.predict(padded_seq)
    predicted_class = (prediction > 0.5).astype(int)
    label = "pozitif" if predicted_class[0][0] == 1 else "negatif"
    return label

# sentence = "Odada klima çok gürültülüydü rahatsız olduk"
sentence = "Otel çok temiz ve rahattı, çok memnun kaldık."

result = classify_sentence(sentence)
print("Tahmin: ",result)






















