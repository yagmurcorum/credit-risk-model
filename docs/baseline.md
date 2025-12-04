# Baseline Model Summary

Bu doküman `04_baseline.ipynb` notebook'unda yapılan baseline model denemelerinin özetini içerir.

## Validasyon Şeması

**Train/Validation Split (80/20)**
- Train set: 120,000 gözlem
- Validation set: 30,000 gözlem
- `stratify=y` ile sınıf oranları korunmuştur
- `random_state=42` ile tekrarlanabilirlik sağlanmıştır

**Neden bu şema seçildi?**
- Baseline aşamasında hızlı değerlendirme için yeterli
- Class imbalance (%7 default) nedeniyle stratify kullanıldı
- Feature engineering istatistikleri tüm veri üzerinde hesaplanmıştır; bu adımlarda hedef değişken kullanılmadığı için ciddi bir *data leakage* beklenmemektedir, 
ancak bir sonraki iterasyonda bu istatistiklerin yalnızca train set üzerinde fit edilmesi planlanmaktadır.
- İleride XGBoost için aynı split kullanılarak tutarlı karşılaştırma yapılacak

## Baseline Feature Seti

Baseline modeller **yalnızca sayısal değişkenler** ile eğitilmiştir:
- 22 sayısal feature
- Kategorik/binned değişkenler encoding yapılmadığı için dahil edilmemiştir
- Amaç: En basit pipeline ile referans noktası oluşturmak

## Model 1: Logistic Regression

**Konfigürasyon:**
- StandardScaler ile ölçeklendirme
- `class_weight="balanced"` ile sınıf dengesizliği ele alındı
- `max_iter=1000`

**Performans Metrikleri:**
- **ROC-AUC:** 0.8622
- **Recall (1):** 0.7456
- **Precision (1):** 0.2293
- **F1-score (1):** 0.3508
- **Accuracy:** 0.8155

**Confusion Matrix:**
- TP = 1,495
- FP = 5,024
- FN = 510
- TN = 22,971

**Değerlendirme:**
- Yüksek recall (%75) → Pozitif sınıfı yakalama başarısı iyi
- Düşük precision (%23) → Çok fazla yanlış alarm üretiyor
- ROC-AUC oldukça iyi (0.86) → Olasılık bazında ayrıştırma gücü var
- Lineer model, veri setinin non-lineer yapısını tam yakalayamıyor

## Model 2: Random Forest

**Konfigürasyon:**
- `n_estimators=200`
- `min_samples_split=5`
- `min_samples_leaf=2`
- `class_weight="balanced_subsample"`
- Hiperparametre optimizasyonu yapılmadı (baseline amaçlı)

**Performans Metrikleri:**
- **ROC-AUC:** 0.8501
- **Recall (1):** 0.3017
- **Precision (1):** 0.4836
- **F1-score (1):** 0.3716
- **Accuracy:** 0.9318

**Confusion Matrix:**
- TP = 605
- FP = 646
- FN = 1,400
- TN = 27,349

**Değerlendirme:**
- Düşük recall (%30) → Pozitif sınıfı yakalama başarısı zayıf
- Yüksek precision (%48) → Daha az yanlış alarm üretiyor
- F1-score LR'dan biraz daha iyi
- Numeric-only feature seti RF'nin potansiyelini sınırlıyor

## Baseline Sonuçlarının Karşılaştırması
| Model               | ROC-AUC | Recall | Precision | F1-score |
| :------------------ | :-----: | :----: | :-------: | :------: |
| Logistic Regression | 0.8622  | 0.7456 |  0.2293   |  0.3508  |
| Random Forest       | 0.8501  | 0.3017 |  0.4836   |  0.3716  |


**Gözlemler:**
1. **Veri tamamen lineer değildir:** LR precision'da çökerken RF non-lineer yapıları kullanarak farklı bir precision/recall dengesi sunuyor.
2. **Sayısal feature setinin tahmin gücü sınırlıdır:** Non-lineer RF modeli bile recall'da düşük kalıyor.
3. **Class imbalance etkisi büyüktür:** Accuracy **tek başına** anlamlı bir metrik değildir.
4. **ROC-AUC her iki modelde de ~0.85 seviyesinde:** Veri orta-güçlü ayrıştırılabilirliğe sahip.
5. **Gerçek performans için kategorik ve binned feature'lar kritiktir:** Encoding yapılmadığı için modeller potansiyellerine tam olarak ulaşamamış.

## Sonraki Adımlar

Baseline modeller, XGBoost + tam feature set + encoding + optimizasyon ile karşılaştırma için referans noktası oluşturmuştur. Final modelin bu baseline'lardan ne kadar iyileşme sağladığı `docs/evaluation.md` içinde detaylandırılmıştır.
