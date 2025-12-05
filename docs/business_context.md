# Business Context

## Problem Tanımı

**Sektör:** Bankacılık / Finansal Hizmetler  

**Problem:**  
Kredi başvurularının geri ödeme riskini (default risk) tahmin etmek.  
Müşterilerin önümüzdeki 2 yıl içinde ciddi finansal sıkıntı (serious delinquency) yaşama olasılığını öngörmek.

**Hedef Değişken:** 
`SeriousDlqin2yrs`  
- 0 → Önümüzdeki 2 yıl içinde ciddi finansal gecikme YOK  
- 1 → Önümüzdeki 2 yıl içinde en az bir kez ciddi finansal gecikme VAR  

**İş Değeri:**
- Doğru risk tahmini ile kredi kayıplarını azaltmak
- Yanlış red kararlarını minimize etmek (iyi müşteriyi kaybetmemek)
- Risk bazlı fiyatlandırma ve limit belirleme
- Regülasyon gereksinimlerini karşılamak (model açıklanabilirliği ve izlenebilirlik)

## Dataset

**Kaynak:** *Give Me Some Credit* (Kaggle Competition)

- **Satır sayısı:** ~150,000  
- **Feature sayısı (ham veri):** 11 açıklayıcı değişken + 1 hedef = **12 kolon**  
- **Final tablo:** Feature engineering sonrası, hedef dahil **27 kolon**  
  (`data/training_prepared.csv` içinde saklanır; 26 feature + 1 hedef)  
- *Class imbalance:* ~%7 default, ~%93 non-default  
- **Format:** Tabular (CSV)

**Dataset Özellikleri:**
- Gerçek bankacılık verisinden türetilmiş, anonimleştirilmiş bir kredi portföyü
- Yeterli hacim (150k satır) → istatistiksel olarak güvenilir analiz
- Feature sayısı hem modelleme hem de feature engineering için yeterli
- Kaggle discussion ve örnek notebook’lar ile domain bilgisi desteklenebilir

## İş Hedefleri

1. **Risk Ayrıştırma**  
   Default riski taşıyan müşterileri olabildiğince doğru segmentlere ayırmak.

2. **Maliyet Optimizasyonu**  
   Yanlış pozitifleri (gereksiz yere reddedilen “iyi” müşteriler) azaltmak; böylece hem kredi kaybını hem de operasyonel maliyetleri düşürmek.

3. **İş Hacmi**  
   Onay oranını makul seviyede tutmak; model yüzünden gereksiz müşteri kaybı yaratmamak.

4. **Açıklanabilirlik**  
   Model kararlarının hem iş birimine hem de regülatörlere açıklanabilir olması  
   (feature importance, müşteri bazlı karar açıklamaları, dokümantasyon).

## Operasyonel Gereksinimler

1. **Model Performansı**
   - ROC-AUC > **0.85** (güçlü ayrıştırma kapasitesi)
   - Precision > **0.40** (yanlış alarm oranı düşük)
   - Recall > **0.45** (riskli müşterilerin en az yarısının yakalanması)

2. **Deployment**
   - REST API üzerinden online tahmin alınabilmeli (FastAPI)
   - Demo / iş birimi kullanımı için Streamlit arayüzü
   - Batch prediction desteği (portföy taraması, kampanya seçimi vb.)

3. **Monitoring**
   - Model drift takibi (PSI, skor dağılımları)
   - Prediction latency (ortalama < 100 ms hedefi)
   - Default rate ve bad rate sapmalarına yönelik alarm mekanizması

4. **Açıklanabilirlik**
   - SHAP değerleri ile global feature importance
   - Müşteri bazında karar açıklaması (örnek force plot’lar)
   - Business-friendly raporlama (`docs/` klasörü, dashboard açıklamaları)

## Business Metrikleri (Threshold = 0.81)

Bu projede, validation set üzerinde **F1 skorunu maksimize eden threshold** `0.81` olarak seçilmiştir.

*Threshold 0.81 ile (validation set üzerinde):*
- **Approval rate:** ~%92  
  → Müşterilerin büyük kısmı onaylanıyor, iş hacmi korunuyor.
- **Bad rate in approved:** ~%3.7  
  → Onaylananlar içinde kabul edilebilir seviyede düşük default oranı.
- **Catch rate of bads (Recall):** ~%47.9  
  → Tüm riskli müşterilerin yaklaşık yarısı yakalanıyor.
- **Precision:** ~%42.25  
  → Reddedilen müşterilerin yaklaşık %42’si gerçekten riskli.

Bu metrikler, bankanın “yanlış alarm maliyeti” ile “kaçan riskli müşteri maliyeti” arasında dengeli bir trade-off sağlamayı hedeflemektedir.

## Modelin İş Dünyasına Faydası

1. **Risk Yönetimi**
   - Kredi kayıplarını azaltmaya yardımcı olur (özellikle en riskli segmentlerde)
   - Risk bazlı limit, kampanya ve fiyatlama stratejilerini destekler
   - Segment bazlı politika üretimine (decile analizi) altyapı sağlar

2. **Operasyonel Verimlilik**
   - Basit başvurular için otomatik onay/red akışı kurulabilir
   - Manuel kredi değerlendirme iş yükünü azaltır
   - API üzerinden milisaniye seviyesinde yanıt süreleri ile hızlı karar verir

3. **Müşteri Deneyimi**
   - Hızlı başvuru ve karar süreci
   - Gereksiz red kararlarının azaltılması ile daha adil yaklaşım
   - Şeffaf karar mekanizması (gerekirse SHAP bazlı açıklamalarla desteklenebilir)

4. **Uyumluluk ve Governance**
   - Regülasyon gereksinimlerini destekleyen açıklanabilir model yapısı
   - Model versiyonlama, monitoring ve audit trail ile takip edilebilirlik
   - `docs/` klasörü ve monitoring planı ile kurumsal model risk yönetimi süreçlerine uyum

