# Model Optimization Summary

Bu doküman `05_xgboost.ipynb` notebook'unda yapılan XGBoost model optimizasyonu sürecinin özetini içerir.

## Model Seçimi

**XGBoost seçilme nedenleri:**
- Baseline analizlerinde veri setinin non-lineer yapıda olduğu görüldü.
- Gradient boosting algoritmaları kredi riski problemlerinde yaygın kullanılıyor.
- Feature importance ve SHAP ile açıklanabilirlik sağlanabiliyor.
- Sınıf dengesizliği (class imbalance) problemine `scale_pos_weight` ile çözüm sunuyor.

## Ön İşleme Pipeline'ı

**ColumnTransformer yapısı:**
- Sayısal değişkenler: `passthrough` (ölçeklendirme yapılmadı, XGBoost ağaç tabanlı olduğu için gerekmiyor).
- Kategorik değişkenler: `OneHotEncoder(handle_unknown="ignore")`
- **22 sayısal + 4 kategorik feature → encoding sonrası toplam ≈38 feature.**
- 
**Class Imbalance Yönetimi:**
- `scale_pos_weight = neg_count / pos_count ≈ 13.96`
- Train set üzerinden hesaplandı (111,979 negatif / 8,021 pozitif).

## Baseline XGBoost Performansı

**Konfigürasyon (optimizasyon öncesi):**
- `n_estimators = 400`
- `max_depth = 4`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `scale_pos_weight = 13.96`

**Performans Metrikleri (validation set, threshold = 0.50):**
- **ROC-AUC:** 0.8685  
- **Recall (1):** 0.7751  
- **Precision (1):** 0.2219  
- **F1-score (1):** 0.3450  
- **Accuracy:** 0.8033  

**Değerlendirme:**
- Baseline modellerden (LR: 0.8622, RF: 0.8501) daha iyi ROC-AUC.
- Yüksek recall (yaklaşık %78) ama düşük precision (yaklaşık %22) → Çok fazla yanlış alarm.
- Bu nedenle hyperparameter optimizasyonu ve threshold tuning adımları planlandı.

## Hyperparameter Optimizasyonu

**Yöntem:** `RandomizedSearchCV`
- **CV şeması:** 3-fold `StratifiedKFold(shuffle=True, random_state=42)`
- **Iterasyon sayısı:** 20
- **Scoring metrik:** ROC-AUC (sınıf dengesiz yapısı için uygun)
- **Arama uzayı:**
```python  
{
    "model__n_estimators": [200, 300, 400, 500],
    "model__max_depth": [3, 4, 5, 6],
    "model__min_child_weight": [1, 2, 3, 4],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__learning_rate": [0.03, 0.05, 0.07, 0.1]
}
```

**En İyi Parametreler (CV sonucu):**
- `subsample`: 0.9  
- `n_estimators`: 300  
- `min_child_weight`: 3  
- `max_depth`: 4  
- `learning_rate`: 0.03  
- `colsample_bytree`: 1.0  
- **CV ROC-AUC:** 0.8649  

**Optimizasyon Sonrası Performans (validation set, threshold = 0.50):**
- **ROC-AUC:** 0.8699 (+0.0014 iyileşme)  
- **Recall (1):** 0.7825  
- **Precision (1):** 0.2156  
- **F1-score (1):** 0.3381  
- **Accuracy:** 0.7952  

**Değerlendirme:**
- CV ile yapılan optimizasyon modelin stabilitesini artırdı.
- AUC hafif iyileşti → Modelin genel ayrıştırma kapasitesi daha güçlü.
- Precision ve F1'de büyük değişim yok (threshold hâlâ 0.50).
- Gerçek iyileşme, threshold tuning adımında elde edildi.

## Threshold Optimizasyonu

**Yöntem:**
- Validation set üzerinde `predict_proba` çıktıları kullanıldı.
- 0.10–0.90 aralığında 81 farklı threshold değeri tarandı (0.01 adım).
- Her threshold için precision, recall, F1 hesaplandı.
- **F1 skorunu maksimize eden threshold seçildi.**

**Optimal Threshold:** 0.81

**Threshold = 0.81 ile Final Performans (validation set):**
- **ROC-AUC:** 0.8699 (threshold'dan etkilenmez)  
- **Recall (1):** 0.4788  
- **Precision (1):** 0.4225  
- **F1-score (1):** 0.4489 (tüm modellerde en yüksek)  
- **Accuracy:** 0.9214  

**Confusion Matrix (threshold = 0.81):**
- TP = 960  
- FP = 1,312  
- FN = 1,045  
- TN = 26,683  

**Business Metrikleri (validation set, threshold = 0.81):**
- **Approval rate:** %92 (müşterilerin büyük kısmı onaylanıyor)  
- **Bad rate in approved:** %3.7 (kabul edilebilir düşük seviye)  
- **Catch rate of bads (TPR):** %47.9  

## Optimizasyon Sonuçlarının Karşılaştırması

| Aşama                     | ROC-AUC | Recall | Precision | F1-score   |
|---------------------------|---------|--------|-----------|----------  |
| Baseline XGBoost          | 0.8685  | 0.7751 | 0.2219    | 0.3450     |
| Tuned XGBoost (CV, 0.50)  | 0.8699  | 0.7825 | 0.2156    | 0.3381     |
| Tuned + Optimal Threshold | 0.8699  | 0.4788 | 0.4225    | **0.4489** |

**Gözlemler:**
1. **AUC değerleri tüm aşamalarda tutarlı ve yüksek (~0.87)** → Modelin ayrıştırma gücü oldukça iyi.
2. **Threshold tuning, precision ve F1'i ciddi şekilde geliştirdi:**
   - Precision: 0.22 → 0.42 (neredeyse 2 katına çıktı),
   - F1: 0.35 → 0.45 (en yüksek seviye).
3. **Recall'ın düşmesi işin doğası gereği kabul edilebilir bir trade-off:**
   - Daha az yanlış alarm → Daha az gereksiz müdahale ve operasyonel maliyet.
   - İş dünyasında daha sürdürülebilir bir risk–maliyet dengesi sağlıyor.

## Final Model Seçimi

**Final model:** XGBoost (tuned) + optimal threshold = 0.81

**Seçilme nedenleri:**
- En dengeli precision–recall ilişkisini sağlaması,
- İş dünyasında daha düşük yanlış alarm maliyeti yaratması,
- ROC-AUC açısından en güçlü modellerden biri olması,
- SHAP analizi için kararlı ve tutarlı bir yapı sunması.

**Model dosyası:**
- Kaydedilen dosya: `models/xgboost_credit_risk_final.pkl`  
- İçerik: `{"model": xgb_best, "threshold": 0.81, "features": all_feature_names}`

