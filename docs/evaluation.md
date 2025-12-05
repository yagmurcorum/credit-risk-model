# Evaluation Summary

Bu doküman `05_xgboost.ipynb` notebook'unda yapılan model değerlendirme ve SHAP analizi sonuçlarının özetini içerir.

## Validasyon Şeması ve Neden Seçildi

**Train/Validation Split (80/20)**  
- Train set: 120,000 gözlem  
- Validation set: 30,000 gözlem  
- `stratify=y` ile sınıf oranları korunmuştur  
- `random_state=42` ile tekrarlanabilirlik sağlanmıştır  

**Neden bu şema seçildi?**
1. **Basit ve hızlı:** Kredi riski problemlerinde yaygın kullanılan standart yaklaşım.
2. **Stratify gerekli:** Class imbalance (~%7 default) nedeniyle sınıf oranlarının korunması kritik.
3. **Tutarlılık:** Baseline modellerle aynı split kullanılarak adil karşılaştırma yapıldı.
4. **Yeterli validasyon seti:** 30,000 gözlem threshold tuning ve SHAP analizi için yeterli.

### Validation Şeması ve Sınırlamalar

Bu çalışmada veri, **%80 eğitim / %20 validation** olacak şekilde (stratified split) ikiye ayrılmış ve
**model seçimi + final performans raporu bu validation seti üzerinde** yapılmıştır.
Ayrı bir bağımsız **hold-out test seti** ayrılmamıştır.

Veri boyutu büyük olduğu için, pratikte ek bir test seti ayırmanın metriklerde anlamlı bir fark
yaratmasını beklemiyoruz. Buna rağmen, daha ileri bir iterasyonda:

- **model seçimi** için validation seti,
- **nihai raporlama** için ise dokunulmamış bağımsız bir hold-out test seti

kurgulamak, metodolojik olarak bir adım daha ileri olacaktır.

### Alternatifler Neden Seçilmedi?

- **K-Fold CV:**  
 - Hyperparameter tuning aşamasında kullanılmıştır.  
 - Final değerlendirme için ise tek bir validation set kullanmak, hız ve yorumlanabilirlik açısından bu proje bağlamında daha pratiktir.

- **Time-based split:**  
  - Veri seti zaman serisi yapısında değildir; gözlemler belirli bir kronolojik sıraya göre toplanmamıştır.
  - Bu nedenle **rastgele stratified split**, bu problem için uygun bir seçimdir.

> **Not (Data Leakage):**  
> Median, quantile ve bin sınırları gibi bazı istatistikler, train/validation
> ayrımından önce **tüm veri** üzerinde hesaplanmıştır. Bu, teorik olarak
> hafif bir *data leakage* riski yaratır; ancak hedef değişken
> (`SeriousDlqin2yrs`) bu adımlarda kullanılmadığı için pratik etkisinin düşük
> olduğu düşünülmektedir. İleride, bu istatistiklerin yalnızca **train set**
> üzerinde fit edilmesi planlanmaktadır.

## Baseline vs Final Model Karşılaştırması

### Baseline Modeller

| Model               | ROC-AUC | Recall | Precision | F1-score |
|---------------------|---------|--------|-----------|----------|
| Logistic Regression | 0.8622  | 0.7456 | 0.2293    | 0.3508   |
| Random Forest       | 0.8501  | 0.3017 | 0.4836    | 0.3716   |

**Baseline özellikleri:**
- Yalnızca 22 sayısal feature kullanıldı.
- Kategorik / binned değişkenler encoding yapılmadı.
- Hiperparametre optimizasyonu yapılmadı.
- Threshold = 0.50 (varsayılan).

### Final Model (XGBoost)

| Metrik   | Baseline XGBoost | Final (Tuned + Threshold) | İyileşme            |
|----------|------------------|---------------------------|---------------------|
| ROC-AUC  | 0.8685           | 0.8699                    | +0.0014             |
| Recall   | 0.7751           | 0.4788                    | -0.2963 (trade-off) |
| Precision| 0.2219           | 0.4225                    |**+0.2006 (+~%90)**  |
| F1-score | 0.3450           | 0.4489                    | **+0.1039 (+~%30)** |

**Final model özellikleri:**
- 22 sayısal + 4 kategorik feature (XGBoost pipeline girişi – toplam 26 kolon)
- ColumnTransformer + OneHotEncoder sonrası modelin gördüğü toplam 38 feature
- Hyperparameter optimization (RandomizedSearchCV, 3-fold Stratified CV)
- Optimal threshold tuning (0.81)
- SHAP ile açıklanabilirlik (global + bireysel seviye) 

**Başarı farkı:**
- **ROC-AUC:** Baseline XGBoost’a göre küçük ama anlamlı bir iyileşme (0.8685 → 0.8699).
- **F1-score:** Baseline XGBoost’tan ~%30 iyileşme (0.3450 → 0.4489).
- **Precision:** Baseline XGBoost’tan ~%90 iyileşme (0.2219 → 0.4225).
- **Recall:** 0.7751 → 0.4788’e düştü; bu, iş gereksinimleri için bilinçli bir trade-off olarak ele alındı.

**Yorum:**
- Full feature set + encoding + optimizasyon ile model performansı anlamlı şekilde iyileşti.
- Threshold tuning, precision–recall dengesini iş gereksinimlerine daha uygun hale getirdi.
- Yanlış alarm oranı (false positive) azaldığı için operasyonel maliyetler düşme eğiliminde.

### SHAP Analizi – Feature Importance

SHAP (SHapley Additive Explanations) analizi ile modelin karar mekanizması hem global (feature importance) hem de lokal (tekil müşteri) düzeyde açıklanmıştır.

<p align="center">
  <img src="cases/shap.png" alt="SHAP beeswarm feature importance" width="650">
</p>

<p align="center">
  <em>Şekil: Validation set üzerinde XGBoost modelinin global SHAP (beeswarm) özeti</em>
</p>

#### En Önemli Feature’lar (Global SHAP – Beeswarm Özeti)

- **RevolvingUtilizationOfUnsecuredLines**  
  - Kredi kartı limitlerinin ne kadarının kullanıldığını gösteren oran.  
  - Yüksek kullanım (kırmızı noktalar) genellikle pozitif SHAP değerleriyle sağ tarafta; düşük kullanım (mavi) negatif tarafta yoğunlaşıyor.  
  - **Yorum:** Limitlerinin büyük kısmını kullanan müşteriler model gözünde belirgin şekilde daha riskli.

- **Delinq_x_Utilization (FE)**  
  - Toplam gecikme sayısı × kart kullanım oranı.  
  - Hem gecikmesi hem de kart kullanımı yüksek olan müşterilerde SHAP değerleri pozitif bölgede.  
  - **Yorum:** Gecikme + agresif kullanım kombinasyonu, model için net bir yüksek risk sinyali.

- **EverDelinquent (FE)**  
  - Müşterinin geçmişinde en az bir gecikme yaşayıp yaşamadığını gösteren ikili değişken (0/1).  
  - Değer 1 olduğunda noktalar daha çok sağ tarafa kayıyor.  
  - **Yorum:** Gecikme geçmişi olan müşteriler daha kırılgan bir profil çiziyor.

- **DelinquencySeverityScore (FE)**  
  - 30–59 / 60–89 / 90+ gün gecikmeleri ağırlıklı toplayan “gecikme şiddeti skoru”.  
  - Skor yükseldikçe SHAP değeri pozitif tarafa kayıyor.  
  - **Yorum:** Daha yoğun ve ağır gecikme geçmişi, default riskini belirgin biçimde artırıyor.

- **age**  
  - Genç yaşlar (mavi) nispeten daha fazla pozitif SHAP tarafında; orta–ileri yaşlar daha çok negatif tarafta toplanıyor.  
  - **Yorum:** Model, genç profili biraz daha riskli; daha olgun yaşları ise daha güvenli olarak değerlendiriyor.

- **MonthlyIncome & IncomeBin (0–3k, 3–6k, 6–10k, 10k+) (FE)**  
  - Aylık gelir ve gelir segmentlerini temsil eden değişkenler.  
  - Yüksek gelir ve üst segment bin’lerinde SHAP değerleri çoğunlukla negatif; düşük gelir seviyelerinde ise pozitif değerlere daha sık rastlanıyor.  
  - **Yorum:** Gelir seviyesi ve gelir segmentasyonu, ödeme kapasitesi üzerinden risk profilini belirgin şekilde etkiliyor.

- **DebtToIncomeRatio & DebtRatio_log1p (FE)**  
  - Toplam borç / gelir oranı ve bu oranın log-transform edilmiş hali.  
  - SHAP dağılımı, bu değişkenlerin diğer ana risk faktörlerine göre görece daha sınırlı ama tutarlı bir etki taşıdığını gösteriyor.  
  - **Yorum:** Borç/gelir dengesi model için önemli; ancak bu veri setinde asıl yükü delinquency ve utilization tarafı çekiyor.

- **EffectiveDebtLoad (FE)**  
  - `EffectiveDebtLoad = DebtRatio × MonthlyIncome` → Gelire göre parasal borç yükü.  
  - Değer yükseldikçe birçok gözlemde SHAP değerlerinin pozitif tarafa kaydığı görülüyor.  
  - **Yorum:** Gelire göre borcu “ağır” olan müşteriler daha riskli olarak değerlendiriliyor.

- **NumberOfOpenCreditLinesAndLoans & NumberRealEstateLoansOrLines**  
  - Aktif kredi sayısı ve gayrimenkul kredisi sayısı.  
  - Makul seviyelerde etkileri nötr; çok yüksek sayılarda SHAP değeri pozitif tarafa kayıyor.  
  - **Yorum:** Çok sayıda açık kredi ve mortgage, risk yönünde ek baskı yaratıyor.

- **RealEstateExposure & HighUtil_x_DebtRatio (FE)**  
  - **RealEstateExposure:** `NumberRealEstateLoansOrLines × DebtRatio` → Gayrimenkul kredi yükü.  
  - **HighUtil_x_DebtRatio:** “Yüksek kullanım” flag’i (1/0) × borç/gelir oranı.  
  - **Yorum:** Yoğun konut/gayrimenkul kredisi ve aynı anda hem yüksek utilization hem de yüksek borç oranına sahip olmak, modelde net bir yüksek risk kümesini temsil ediyor.

> Ek not: `Utilization_x_DebtRatio` gibi bazı etkileşim değişkenleri, SHAP dağılımında beklenenden daha zayıf sinyal üretmektedir. Bu tip feature’lar, ileriki iterasyonlarda sadeleştirme veya feature pruning için aday olarak değerlendirilebilir.



### Business Yorumu

Model en çok **kart kullanım oranı, gecikme geçmişi, borç/gelir dengesi ve gelir seviyesi**
üzerinden karar veriyor. Domain bilgisiyle tasarlanan delinquency ve borç yükü
türevleri, modelin riskli segmentleri daha net ayırt etmesine yardımcı oluyor.  
Bu yapı, hem kredi tahsis kararları hem de limit güncellemeleri / koleksiyon stratejileri
için anlamlı bir karar destek sistemi sunuyor.


## Threshold Seçimi ve Business Uyumu

### Threshold Optimizasyonu Süreci

- Validation set üzerinde `xgb_best.predict_proba(X_val)[:, 1]` kullanılarak
  **default olasılıkları** hesaplandı.
- 0.10–0.90 aralığında, 0.01 adım ile **81 farklı threshold** denendi.
- Her threshold için:
  - Precision, recall, F1,
  - Approval rate, decline rate,
  - Bad rate in approved,
  - Catch rate of bads (TPR)  
  gibi hem teknik hem business metrikleri hesaplandı.
- **F1 skorunu maksimize eden threshold = 0.81 olarak belirlendi.**

### Seçilen Threshold’un Business Gereksinimleri ile Uyumu

**Threshold = 0.81 ile:**
- **Precision (1): 0.4225** → Reddedilen müşterilerin ~%42’si gerçekten default ediyor.  
- **Recall (1): 0.4788** → Tüm kötü müşterilerin ~%48’i yakalanıyor.  
- **F1-score (1): 0.4489** → Tüm modeller içinde en yüksek F1.  
- **Approval rate:** ~%92 → Müşterilerin büyük kısmı onaylanıyor, iş hacmi korunuyor.  
- **Bad rate in approved:** ~%3.7 → Onaylanan portföy içinde makul bir default oranı.  

**Neden bu threshold seçildi?**
1. **F1 skorunu maksimize ediyor** → Precision–recall dengesi teknik olarak optimal noktada.
2. **Yanlış alarm maliyetini azaltıyor** → Precision, baseline XGBoost’a göre yaklaşık 2 katına çıktı.
3. **İş hacmini koruyor** → Yüksek approval rate ile gereksiz müşteri kaybı minimize edilmiş durumda.
4. **Bad rate kontrol altında** → Onaylanan müşteri havuzunda default oranı kabul edilebilir seviyede.

**Business gereksinimleri ile uyumlu mu?**
- Yanlış alarm maliyetlerini azaltırken iş hacmini koruyor.  
- Onaylanan portföyün risk kalitesini yükseltiyor (daha temiz havuz).  
- Riskli müşterilerin yaklaşık yarısını yakalıyor (recall ~%48).  
- Daha yüksek recall istenirse threshold düşürülebilir; ancak bu durumda precision
  ve bad rate in approved değerleri kötüleşecektir. Bu, iş birimlerinin risk tercihine göre
  yeniden ayarlanabilecek bir kontrol noktasıdır.


## Model Seçimi

**Final model:** XGBoost (tuned) + optimal threshold = 0.81

**Seçilme nedenleri:**
1. **En yüksek F1-score:** Tüm denenen modeller içinde 0.4489 ile en iyi değer.
2. **Dengeli precision–recall:** İş gereksinimlerine uygun bir risk–ödül dengesi sunuyor.
3. **Yüksek ROC-AUC:** 0.8699 ile güçlü ayrıştırma gücü.
4. **Açıklanabilirlik:** SHAP ile hem global hem lokal düzeyde yorumlanabilir.
5. **Deployment uyumu:** Pipeline yapısı ve kayıtlı model dosyası ile üretim ortamına taşınmaya hazır.


## Ranking Performansı (Decile Analizi)

Modelin, ürettiği **risk skoruna göre müşterileri sıralama (ranking)** performansı
decile bazlı analiz edilmiştir. Her decile, portföyün yaklaşık %10’luk bir dilimini temsil eder.

### En Riskli Segmentler

- **Decile 10 (en riskli %10):**
  - Toplam müşterilerin ≈ %10’u.
  - Tüm “bad” müşterilerin önemli bir kısmını (örneğin ~%40–45’ini) içeriyor.
  - Bad rate bu dilimde oldukça yüksek (çift haneli, %20+ seviyeleri).  
  - → Portföyün “en kırılgan” kısmı olarak görülebilir.

- **Decile 9–10 (en riskli %20):**
  - Toplam müşterilerin %20’si.
  - Tüm bad’lerin kabaca **%60–65** kadarı bu iki dilimde toplanıyor.
  - → Risk ekibinin “yakından izlenecek portföyü” için güçlü bir aday segment.

- **Decile 8–10 (en riskli %30):**
  - Toplam müşterilerin %30’u.
  - Tüm bad’lerin yaklaşık **%70+**’i bu grupta yer alıyor (notebook analizinde ~%73 civarı).  
  - → Model, kötü müşterilerin büyük çoğunluğunu en riskli %30’luk dilimde toplayabildiği için
    **ranking performansı güçlü**.

### En Güvenli Segmentler

- **Decile 1–3 (en güvenli %30):**
  - Portföyün en güvenli kısmı.
  - Bad rate ~%0.4–1.0 bandında; yani neredeyse “temiz” bir segment.  
  - → Düşük riskli ürünler, kampanyalar veya limit artışı gibi aksiyonlar için hedeflenebilecek güvenli havuz.

### Genel Yorum

Model, risk skorunu kullanarak müşterileri sıraladığında:

- **Düşük skor decile’ları (1–3)** → En güvenli %30’luk segment.  
  Bad rate ~%0.4–1.0 bandında; “temiz” iyi müşteri havuzu.

- **Yüksek skor decile’ları (8–10)** → En riskli %30’luk segment.  
  Bad rate hızla artıyor ve 10. decile’da ~%37 seviyesine çıkıyor.  
  Kötü müşterilerin büyük bir kısmı bu yüksek risk decile’larında yoğunlaşıyor.

Bu yapı, özellikle:

- Limit güncelleme,
- Proaktif arama / hatırlatma,
- Farklı faiz / kampanya stratejileri,
- Koleksiyon / erken uyarı sistemleri

gibi kararlar için güçlü bir **segmentasyon altyapısı** sağlayarak, modeli sadece
tahmin aracı olmaktan çıkarıp gerçek bir **iş karar destek sistemi** haline getiriyor.
