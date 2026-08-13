# churn-analizi-
# Telco Customer Churn Prediction with XGBoost

Bu proje, telekomünikasyon sektörü müşteri verilerini kullanarak müşterilerin şirketten ayrılıp ayrılmayacağını (churn) tahmin eden uçtan uca bir makine öğrenmesi modelidir.

## 🚀 Proje Adımları
1. **Veri Keşfi ve Ön İşleme:** Eksik verilerin doldurulması ve kategorik değişkenlerin kodlanması (Encoding).
2. **Model Eğitimi:** XGBoost Sınıflandırıcı ile temel modelin kurulması.
3. **Hiperparametre Optimizasyonu:** Model performansını artıran parametre ayarları.
4. **Sınıf Dengesizliği Yönetimi:** `scale_pos_weight` ile azınlık sınıfının başarı oranının artırılması.
5. **Çapraz Doğrulama:** 5 Katlı Stratified K-Fold ile modelin kararlılığının test edilmesi.
6. **Eşik Değeri (Threshold) Optimizasyonu:** Youden İndeksi ile en uygun karar eşiğinin belirlenmesi.

## 🛠️ Kullanılan Teknolojiler
* Python
* Pandas, NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
