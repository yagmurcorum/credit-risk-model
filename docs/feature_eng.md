# Feature Engineering Summary

Bu doküman, `03_feature_engineering.ipynb` notebook’unda yapılan feature engineering
adımlarının özetini içerir.

Genel Yaklaşım

Feature engineering aşamasında:

- Temizlenmiş veri (`cs-training-clean.csv`, 02_data_cleaning.ipynb çıktısı) üzerinde çalışıldı.
- Tüm dönüşümler, train/validation ayrımı yapılmadan veri setinin tamamına uygulandı.
- Sonuçta oluşan final feature seti, `training_prepared.csv` olarak kaydedildi ve model eğitiminde bu tablo kullanıldı.

Not: Bu notebook’ta median, quantile ve bin sınırları gibi istatistikler (örneğin gelir bin’leri),
pratik sebeplerle **tüm temizlenmiş veri** üzerinden hesaplanmıştır. Bu, teorik olarak küçük
bir data leakage anlamına gelir. Ancak bu hesaplamalarda hedef değişken (`SeriousDlqin2yrs`)
hiç kullanılmadığı ve veri seti büyük olduğu için pratik etkisinin düşük olduğu düşünülmektedir.

Projenin bu versiyonunda performans, 80/20 train–validation bölünmesi ile raporlanmıştır ve
ayrı bir bağımsız test seti yoktur. İleriki iterasyonlarda:

- önce yalnızca **train set** üzerinde bu istatistiklerin fit edilmesi,
- ardından aynı parametrelerin validation (ve varsa bağımsız test) setine uygulanması

şeklinde daha sıkı bir yapı kurulması planlanmaktadır.

## 1. Delinquency (Gecikme) Feature’ları

**Orijinal kolonlar:**

- `NumberOfTime30-59DaysPastDueNotWorse`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfTimes90DaysLate`

**Türetilen feature’lar:**

### TotalDelinquency

- Üç gecikme sayısının toplamı.
- **Etkisi:** Model için güçlü sinyal taşıyor; ancak VIF analizi sonrası
  `DelinquencySeverityScore` ile çok yüksek korelasyon sebebiyle final setten çıkarıldı.

### EverDelinquent

- Herhangi bir tipte en az bir gecikmesi olan müşteri (0/1).
- **Default oranı farkı:** Gecikme yaşamayanlar %2.7 civarı, yaşayanlar %10.4+.
- **Etkisi:** Model için kritik ayrıştırıcı; final sette tutuldu.

### Ever90DaysLate

- 90+ gün gecikme yaşamış müşteri bayrağı (0/1).
- **Default oranı:** 90+ gecikme yaşayanlarda %45–50 aralığında.
- **Etkisi:** En güçlü risk göstergelerinden biri; final sette tutuldu.

### MultipleDelinquencyFlag

- Toplam gecikme sayısı ≥ 2 ise 1, aksi halde 0.
- **Default oranı:** Çoklu gecikme yaşayanlarda %24–48 bandında.
- **Etkisi:** Yüksek riskli gecikme geçmişini işaretleyen güçlü bir bayrak; final sette tutuldu.

### DelinquencySeverityScore

- Ağırlıklı gecikme skoru:  
  `1 × (30–59 gün)` + `2 × (60–89 gün)` + `3 × (90+ gün)` gecikmeler.
- **Etkisi:** Ham delinquency kolonlarının yerine geçen özet bir skor olarak kullanıldı
  ve final sette tutuldu.


## 2. Risk Flag’leri (Binary Indicators)

Bu başlık altında modelde 0/1 bayrak gibi çalışan temel risk göstergeleri yer alır.

**HighUtilizationFlag**  
- **Tanım:** `RevolvingUtilizationOfUnsecuredLines ≥ 1.0` → kredi kartı limitini aşan veya limite kadar dayanmış müşterileri işaretler.  
- **Dağılım:** Müşterilerin yaklaşık %2.2’si bu flag’i alıyor.  
- **Etkisi:** Çok yüksek kullanım segmentini yakalayarak en riskli gruplardan birini öne çıkarır; final feature setinde tutulmuştur.

**HighDebtFlag**  
- **Tanım:** `DebtToIncomeRatio` dağılımının üst ~%7–8’lik dilimindeki müşterileri işaretler  
  (train set’te bu eşik yaklaşık **0.40** civarına denk gelmiştir).  
- **Dağılım:** Müşterilerin yaklaşık %7.2’si yüksek borç/gelir oranına sahiptir.  
- **Etkisi:** Borç yükü aşırı olan segmenti ayırır; final sette tutulmuştur.

**MultipleDelinquencyFlag**  
- **Tanım:** Toplam gecikme sayısı `TotalDelinquency ≥ 2`.  
- **Etkisi:** Birden fazla gecikmesi olan müşterileri işaretler; delinquency tarafındaki yoğun problemi özetleyen güçlü bir bayraktır ve final sette tutulmuştur.

> Not: `EverDelinquent` ve `Ever90DaysLate` de 0/1 flag niteliğindedir; ancak gecikme hikâyesiyle birlikte yorumlandıkları için Delinquency bölümünde anlatılmıştır.

## 3. Log Dönüşümleri

**Dönüştürülen değişkenler:**

- `RevolvingUtilizationOfUnsecuredLines_log1p`
- `DebtRatio_log1p`
- `MonthlyIncome_log1p`

**Neden `log1p`?**

- Sağa çarpık dağılımları daha simetrik hale getirmek.
- Uç değerlerin model üzerindeki etkisini yumuşatmak.
- Özellikle ağaç tabanlı olmayan modeller için daha uygun ölçek sağlamak.

**Sonuç:**

- `MonthlyIncome_log1p`, orijinal `MonthlyIncome` ile çok yüksek korelasyon
  ve VIF nedeniyle final setten çıkarıldı.
- Diğer log dönüşümleri final sette tutuldu.


## 4. Binning (Segmentasyon)

### AgeBin

- Segmentler: **18–30**, **31–45**, **46–60**, **60+**.
- **Default oranları:** 18–30 grubunda ~%12, 60+ grubunda ~%3.
- **Etkisi:** Yaşın non-lineer etkisini yakalıyor; final sette tutuldu.

### IncomeBin

- Segmentler: **0–3k**, **3–6k**, **6–10k**, **10k+**.
- **Default oranları:** 0–3k grubunda ~%9.1, 10k+ grubunda ~%4.3.
- **Etkisi:** Gelir segmentasyonu güçlü sinyal üretiyor; final sette tutuldu.

### UtilizationBin

- Segmentler: **0–30%**, **30–70%**, **70–100%**, **100%+**.
- **Default oranları:** 0–30% grubunda ~%2.2, 100%+ grubunda ~%37.2.
- **Etkisi:** Kredi kullanım oranının seviyesini yakalayan önemli bir segmentasyon;
  final sette tutuldu.

### DelinqBin

- Segmentler: **0**, **1**, **2–3**, **4+** toplam gecikme.
- **Default oranları:** 0 gecikme için ~%2.7, 4+ gecikme için ~%51.6.
- **Etkisi:** Gecikme yoğunluğunu sınıflandırdığı için güçlü risk sinyali veriyor;
  final sette tutuldu.


## 5. Interaction Feature’ları

### Utilization_x_DebtRatio

- Tanım: `RevolvingUtilizationOfUnsecuredLines × DebtRatio`.
- **Etkisi:** İki önemli risk faktörünün birleşimini temsil ediyor; final sette tutuldu.

### Income_x_Age

- Tanım: `MonthlyIncome × age`.
- **Etkisi:** VIF analizi sonrası orijinal `MonthlyIncome` ile çok yüksek korelasyon
  nedeniyle final setten çıkarıldı.

### Delinq_x_Utilization

- Tanım: `TotalDelinquency × RevolvingUtilizationOfUnsecuredLines`.
- **Etkisi:** Geçmiş gecikme + mevcut limit kullanımı kombinasyonunu yakalıyor;
  güçlü sinyal ürettiği için final sette tutuldu.

### OpenLines_x_RealEstate

- Tanım: `NumberOfOpenCreditLinesAndLoans × NumberRealEstateLoansOrLines`.
- **Etkisi:** Çoklu kredi yükünü temsil ediyor; final sette tutuldu.

### HighUtil_x_DebtRatio

- Tanım: `HighUtilizationFlag × DebtRatio`.
- **Etkisi:** Limit aşımı yapan ve aynı zamanda yüksek borç oranına sahip
  müşterileri işaretliyor; final sette tutuldu.


## 6. Domain-Driven Feature’lar

### EffectiveDebtLoad

- Tanım: `DebtRatio × MonthlyIncome` → gelire göre **parasal borç yükü**.
- **Etkisi:** `DebtRatio` değişkeninin daha yoruma açık, parasal karşılığı olan
  versiyonu olduğu için `DebtRatio` yerine tutuldu; final sette kaldı.

### CreditLineDensity

- Tanım: `NumberOfOpenCreditLinesAndLoans / age`.
- **Etkisi:** VIF analizi sonrası yüksek VIF + zayıf hedef korelasyonu nedeniyle
  final setten çıkarıldı.

### RealEstateExposure

- Tanım: `NumberRealEstateLoansOrLines × DebtRatio`.
- **Etkisi:** Gayrimenkul kredileri üzerinden borçlanma riskini yansıtıyor;
  final sette tutuldu.

### FinancialStressIndex

- Tanım: `log1p(DebtRatio × RevolvingUtilizationOfUnsecuredLines)`.
- **Etkisi:** Borç oranı ve kredi kullanımını tek bir ölçekte birleştirerek
  genel finansal stresi temsil ediyor; final sette tutuldu.


## Feature Selection Süreci

### Korelasyon Analizi

- Hedef değişkenle en güçlü korelasyon, delinquency türevlerinde gözlendi
  (`Ever90DaysLate`, `DelinquencySeverityScore`, vb.).
- Yüksek feature–feature korelasyonu (> 0.9) tespit edilen başlıca çiftler:
  - `DebtRatio` ↔ `EffectiveDebtLoad`
  - `MonthlyIncome` ↔ `Income_x_Age`
  - Ham delinquency kolonları ↔ `TotalDelinquency` / `DelinquencySeverityScore`

### VIF Analizi

- **Sonsuz VIF:** Ham delinquency kolonları 
  (`NumberOfTime30-59DaysPastDueNotWorse`,
  `NumberOfTime60-89DaysPastDueNotWorse`,
  `NumberOfTimes90DaysLate`, `TotalDelinquency`,
  `DelinquencySeverityScore`) arasında tam ilişki olduğunu gösterdi.
- **Çok yüksek VIF (100+):** `DebtRatio`, `EffectiveDebtLoad`.
- **Yüksek VIF (20–50):** `Income_x_Age`, `MonthlyIncome_log1p`, `CreditLineDensity`
  gibi türev / dönüşmüş kolonlarda görüldü.

### Final Drop Listesi

Aşağıdaki feature’lar, korelasyon ve VIF analizi ile domain yorumu birleştirilerek **final setten çıkarılmıştır:**

1. Ham delinquency kolonları  
   (`NumberOfTime30-59DaysPastDueNotWorse`,  
   `NumberOfTime60-89DaysPastDueNotWorse`,  
   `NumberOfTimes90DaysLate`,  
   `TotalDelinquency`) — yerine `DelinquencySeverityScore` ve flag’ler tutuldu.
2. `DebtRatio` — yerine `EffectiveDebtLoad` tutuldu.
3. `Income_x_Age` — `MonthlyIncome` ile yüksek korelasyon.
4. `MonthlyIncome_log1p` — `MonthlyIncome` ile yüksek korelasyon.
5. `CreditLineDensity` — yüksek VIF + görece zayıf sinyal.

## Final Feature Seti

`training_prepared.csv` dosyasındaki **gerçek nihai yapı**:

- **Toplam kolon sayısı (CSV):** 35  
- **Hedef hariç feature sayısı (CSV):** 34  

Bu 34 feature’nin içinde:

- **Model tarafından aktif kullanılan feature sayısı:** 26  
  - **Sayısal feature sayısı:** 22  
  - **Kategorik feature sayısı (bin kolonları):** 4  
    (`AgeBin`, `IncomeBin`, `UtilizationBin`, `DelinqBin`)
    
- **Model tarafından kullanılmayan ama CSV’de bırakılan 8 feature:**  
  `NumberOfTime30-59DaysPastDueNotWorse`,  
  `NumberOfTime60-89DaysPastDueNotWorse`,  
  `NumberOfTimes90DaysLate`,  
  `TotalDelinquency`,  
  `DebtRatio`,  
  `MonthlyIncome_log1p`,  
  `Income_x_Age`,  
  `CreditLineDensity`.  

> Not: Bu 8 kolon, EDA / analiz ve ileride yapılabilecek daha agresif feature
> selection denemeleri için **dosyada tutulmuştur**, ancak final XGBoost pipeline’ının
> input listesine dahil edilmemiştir. Yani `training_prepared.csv` dosyasında
> 34 feature görünse de, model eğitiminde **yalnızca 26 feature** kullanılmaktadır.

## Model Performansına Etkisi

Modelleme notebook’unda yapılan deneyler üzerinden:

- **Basit baseline modeller**  
  (sadece sayısal kolonlar, bin’ler hariç) için ROC–AUC: **~0.85–0.86**
- **Feature engineering sonrası XGBoost (full feature set)**  
  ROC–AUC: **0.8685 → 0.8699** (hyperparameter optimizasyonu sonrası)
- Kategorik bin’ler, interaction terimleri ve domain-driven feature’lar sayesinde model
  non-lineer ilişkileri daha iyi yakalayabildi.
- Delinquency feature’ları (`Ever90DaysLate`, `DelinquencySeverityScore`,
  `MultipleDelinquencyFlag`, `DelinqBin` vb.) SHAP analiziyle de doğrulandığı üzere
  en güçlü risk sinyalini üreten grup oldu.
