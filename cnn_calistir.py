 #Gerekli kütüphaneleri ekle

import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np


# MNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Görüntü piksellerini 0-1 aralığına normalize et
x_train = x_train / 255.0
x_test = x_test / 255.0

# Etiketleri kategorik formata (one-hot encoding) dönüştür
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Sıralı bir model oluştur
model = Sequential()

# İlk Evrişim (Conv2D) katmanı (32 filtre) ve Havuzlama (MaxPooling) katmanı
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# İkinci Evrişim (64 filtre) ve Havuzlama katmanları
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Ezberlemeyi (overfitting) önlemek için Dropout katmanı (%25)
model.add(Dropout(0.25))


# Veriyi düzleştir (Flatten)
model.add(Flatten())

# Tam bağlı (Dense) katman (128 nöron)
model.add(Dense(128, activation='relu'))

# İkinci Dropout katmanı (%50)
model.add(Dropout(0.5))


# Çıkış katmanı (10 sınıf için 'softmax' aktivasyonu)
model.add(Dense(10, activation='softmax'))

# Modeli derle (optimizatör, kayıp fonksiyonu ve metrikleri belirle)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Modeli eğit (10 epoch, 32'lik batch'ler halinde)
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Modelin test verisi üzerindeki performansını değerlendir
test_loss, test_acc = model.evaluate(x_test, y_test)

# Test doğruluğunu ekrana yazdır
print(f"Test Dogrulugu (Accuracy): {test_acc * 100:.2f}%")

# Eğitilmiş modeli dosyaya kaydet
model.save('mnist_cnn_model.keras')
print("Model başariyla kaydedildi.")




