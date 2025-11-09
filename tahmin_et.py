import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# --- 1. KAYITLI MODELİ YÜKLE ---
print("Kayıtlı model 'mnist_cnn_model.keras' yükleniyor...")
model = tf.keras.models.load_model('mnist_cnn_model.keras')
print("Model yüklendi.")

# --- 2. RESMİ YÜKLE VE HAZIRLA ---
# Paint'ten kaydettiğimiz resmin yolu
img_path = 'rakamim5.png' 

# Resmi yükle:
# color_mode='grayscale' -> Resmi siyah-beyaz (gri tonlamalı) yükle
# target_size=(28, 28)   -> Resmi 28x28 piksele küçült (MNIST formatı)
img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))

# Resmi bir sayı dizisine (array) çevir
img_array = img_to_array(img)

# --- 3. EN ÖNEMLİ ADIM: VERİYİ İŞLEME ---

# a) Renkleri Ters Çevirme (Invert)
# MNIST veri seti: Beyaz (255) rakam, Siyah (0) zemin
# Paint'te çizim: Siyah (0) rakam, Beyaz (255) zemin
# Bu yüzden modelin anlaması için renkleri ters çevirmeliyiz (255 - değer)
img_array = 255.0 - img_array

# b) Normalizasyon (0-1 aralığına çekme)
img_array = img_array / 255.0

# c) Modelin "gördüğü" resmi gösterme (Bu çok faydalı)
# Not: plt.imshow'un düzgün çalışması için 28x28'lik şekle geri döndürüyoruz
print("Modelin resmi nasıl 'gördüğünü' gösteren bir pencere açılacak...")
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title("Modelin Gördüğü İşlenmiş Resim (28x28)")
plt.show()

# d) Modele uygun hale getirme
# Model, resimleri tek tek değil, "küme" (batch) halinde bekler.
# (28, 28, 1) olan şekli (1, 28, 28, 1) yapmalıyız. (1 tane resim var demek)
img_array = img_array.reshape(1, 28, 28, 1)

# --- 4. TAHMİNİ YAP! ---
print("Tahmin yapılıyor...")
prediction = model.predict(img_array)

# 'prediction' bize 10 elemanlı bir olasılık listesi verir
# [0.01, 0.0, 0.1, 0.85, 0.0, 0.03, ...]
# np.argmax() bu listedeki en yüksek olasılığın "indeksini" (yani rakamı) bulur
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("\n--- TAHMİN SONUCU ---")
print(f"Modelin Tahmini: {predicted_digit}")
print(f"Bu tahminden emin olma yüzdesi: {confidence:.2f}%")