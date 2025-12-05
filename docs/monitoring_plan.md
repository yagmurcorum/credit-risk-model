# Monitoring Plan

Model canlıya alındıktan sonra izlenecek metrikler ve alarm kuralları.

> Not: Bu dokümandaki eşik değerler (PSI, latency, approval / bad rate aralıkları),
> `src/config.py` içinde tanımlanan business kuralları ile tutarlı olacak şekilde seçilmiştir.

## Model Deployment Süreci

### 1. Model Paketleme
- Model dosyası: models/xgboost_credit_risk_final.pkl
- İçerik: Pipeline (preprocessing + model), threshold (0.81), feature listesi
- Versiyonlama: Model versiyonu ve eğitim tarihi mutlaka kaydedilmeli

### 2. API Deployment
- *FastAPI* ile REST API olarak deploy edilebilir
- *Streamlit* ile demo / business dashboard arayüzü sağlanabilir
- *Docker* ile containerization önerilir
- *Cloud deployment:* AWS, GCP, Azure veya Heroku/Render vb.

### 3. Entegrasyon
- Kredi başvuru sistemine entegre edilmeli
- Real-time prediction endpoint’i sağlanmalı
- Batch prediction için:
  - Mevcut /predict endpoint’i liste halinde kayıt kabul ederek batch çalışabilir
  - Gerekirse buna özel ayrı bir batch endpoint tasarlanabilir

## İzleme Metrikleri

### 1. Model Performans Metrikleri

*Günlük Takip:*
- *Prediction Volume:* Günlük tahmin sayısı
- *Average Prediction Probability:* Ortalama risk skoru (günlük)
- *Approval Rate:* Onaylanan başvuru oranı
- *Decline Rate:* Reddedilen başvuru oranı

*Haftalık Takip:*
- *Actual Default Rate:* Gerçekleşen default oranı (gecikmeli olarak)
- *Precision (Actual):* Reddedilenler içinde gerçekten default’a düşenlerin oranı
- *Recall (Actual):* Tüm default’lar içinde yakalananların oranı
- *F1-score (Actual):* Precision ve recall dengesi

*Aylık Takip:*
- *ROC-AUC (Backtest):* Model skorlarının gerçekleşen sonuçlarla karşılaştırılması
- *Model Drift:* PSI (Population Stability Index) hesaplaması

### 2. Data Drift Metrikleri

*PSI (Population Stability Index):*
- *Hesaplama:* Aylık olarak, kritik feature’lar ve toplam skor dağılımı için
- *Uyarı Eşiği:* PSI > 0.10 → *UYARI* (dağılım değişmeye başlamış)
- *Alarm Eşiği:* PSI > 0.25 → *ALARM* (veri dağılımı ciddi şekilde değişmiş)

*Feature Distribution Changes:*
- MonthlyIncome dağılımı
- age dağılımı
- Delinquency feature’larının dağılımı
- Kategorik feature’ların (AgeBin, IncomeBin, vb.) frekansları

### 3. Prediction Latency

*Hedefler:*
- *Ortalama latency:* < 100 ms
- *P95 latency:* < 200 ms
- *P99 latency:* < 500 ms

*İzleme:*
- Her API çağrısı için latency loglanmalı
- P95 / P99 değerleri dashboard üzerinden izlenmeli
- Yavaş çağrılar ve timeout’lar için alarm üretilmeli

### 4. Business Metrikleri

*Onay/Red Oranları:*
- Hedef approval rate: %85–95 aralığı
- Decline rate: %5–15 aralığı
- *Alarm:* Approval rate < %80 veya > %98 (iş hacmi / risk iştahı ile uyumsuzluk olabilir)

*Bad Rate in Approved:*
- Hedef: %3–5 aralığı
- *Uyarı:* Bad rate > %5  (config’teki MAX_BAD_RATE_IN_APPROVED ile uyumlu)
- *Kritik Alarm:* Bad rate > %7  (modelin bozulma ihtimali yüksek)

*Catch Rate of Bads (Yakalanan Kötülerin Oranı):*
- Hedef: %40–50 bandı
- *Alarm:* Catch rate < %30 (çok fazla riskli müşteri kaçıyor olabilir)

## Alarm Kuralları

### Kritik Alarmlar (Hemen Müdahale Gerektiren)

1. *PSI > 0.25*  
   Veri dağılımı ciddi şekilde değişmiş.
   - *Aksiyon:* Feature bazlı drift analizi yapılmalı, gerekirse model yeniden eğitilmeli.

2. *Bad Rate in Approved > %7*  
   Onaylanan müşteriler arasında default oranı çok yükselmiş.
   - *Aksiyon:* Model performansı ve threshold ayarları gözden geçirilmeli; gerekirse threshold yükseltilmeli veya model yeniden eğitilmeli.

3. *Catch Rate < %30*  
   Riskli müşterilerin çoğu model tarafından “iyi” sınıfa atılıyor.
   - *Aksiyon:* Threshold düşürülerek recall artırılmalı veya model yeniden değerlendirilmelidir.

4. *Average Prediction Probability, tarihsel ortalamanın belirgin şekilde üzerine çıkarsa*  
   (Örneğin uzun dönem ortalaması ~0.07 iken birkaç ay üst üste 0.12+ seviyelerine çıkarsa)
   - *Aksiyon:* Portföy kompozisyonu ve makroekonomik koşullar iş birimiyle birlikte incelenmeli.

### Uyarılar (Yakından İzleme Gerektiren)

1. *PSI > 0.10*  
   Veri dağılımında anlamlı fakat kritik olmayan bir değişim.
   - *Aksiyon:* Haftalık trend izlenmeli, hangi feature’larda değişim olduğu incelenmeli.

2. *Approval Rate < %85 veya > %95*  
   Onay oranı hedef aralığın dışında.
   - *Aksiyon:* İş birimiyle birlikte threshold ve iş kuralları değerlendirilmelidir.

3. *Prediction Latency P95 > 200 ms*  
   API yanıt süreleri artmaya başlamış.
   - *Aksiyon:* Performans optimizasyonu (donanım, parallelization, caching vb.) değerlendirilmelidir.


## Model Retraining Stratejisi

### Otomatik / Periyodik Retraining Kriterleri

*Koşullar:*
- PSI > 0.25 (kritik veri drift)
- Aylık backtest ROC-AUC < 0.85 (performans anlamlı şekilde düşmüş)
- Belirli bir zaman periyodu (örneğin her 6–12 ay) dolduğunda

*Süreç:*
1. Son X aya ait yeni veri ile model yeniden eğitilir.
2. Eski model ile yeni model A/B test edilir (aynı hold-out üzerinde).
3. Performans ve business metrikleri daha iyi ise yeni model deploy edilir.
4. Model versiyonu, eğitim tarihi ve kullanılan feature set kaydedilir.

### Manuel Retraining Senaryoları

*Durumlar:*
- Yeni feature set / yeni veri kaynakları eklendiğinde
- Business rule veya regulasyon değişiklikleri olduğunda
- Risk iştahı önemli ölçüde değiştiğinde

Bu durumlarda retraining kararı Data Science ekibi + Risk Management ortak kararıyla alınmalıdır.


## Logging ve Audit

### Loglanması Gerekenler

1. *Her prediction için:*
   - Timestamp
   - Input feature set (mümkünse maskeleme / anonimizasyon sonrası)
   - Prediction probability (default olasılığı)
   - Predicted label (0/1)
   - Kullanılan model versiyonu
   - Threshold değeri

2. *Model performansı için:*
   - Günlük/haftalık metrikler (approval rate, bad rate, vs.)
   - Aylık ROC-AUC ve F1 skorları (backtest)
   - PSI değerleri
   - Actual vs predicted karşılaştırmaları

3. *Sistem metrikleri:*
   - API latency (ortalama, P95, P99)
   - HTTP hata oranları (5xx, 4xx)
   - Günlük istek sayısı ve trafik pattern’leri

### Audit Trail

- Model versiyonları ve her versiyonun eğitim tarihi
- Kullanılan training veri aralığı ve temel istatistikler
- Retraining tarihleri ve gerekçeleri
- Threshold değişiklikleri (ne zaman, kim tarafından, neden)
- Feature set değişiklikleri
- Performans metrikleri geçmişi (zaman serisi)

## Dashboard Önerileri

*Real-time Dashboard:*
- Günlük prediction volume
- Approval / decline rates
- Average prediction probability
- Sistem sağlığı (latency, error rate, uptime)

*Haftalık Dashboard:*
- Actual default rate (gecikmeli)
- Precision / Recall / F1 (actual)
- PSI değerleri (kritik feature’lar için)
- Feature distribution changes (boxplot / histogram)

*Aylık Dashboard:*
- Model performance trendleri (ROC-AUC, F1)
- Business impact metrikleri (bad rate, catch rate, approval rate)
- Model drift analizi (PSI + skor dağılımları)
- Retraining ihtiyacı / önerileri

## İletişim ve Escalation

*Alarm Durumunda:*
1. Data Science ekibi bilgilendirilmeli (teknik analiz).
2. İş birimi (Risk Management / Kredi Tahsis) bilgilendirilmeli.
3. Kritik alarmlarda (örneğin bad rate ciddi artış, PSI çok yüksek) üst yönetim ve ilgili komiteler bilgilendirilmeli.

*Raporlama:*
- Haftalık özet rapor (kısa executive summary)
- Aylık detaylı analiz (teknik + business)
- Üç aylık (quarterly) business review: modelin portföy üzerindeki etkisi, strateji güncellemeleri
