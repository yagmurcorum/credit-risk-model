## Kredi Risk Skorlama Modeli (End-to-End ML Projesi)

Bankacılık sektöründe kredi başvurularının geri ödeme (*default) riskini tahmin etmek için hazırlanmış uçtan uca bir **makine öğrenmesi projesi*.

Proje; veri keşfinden (EDA), veri temizleme ve feature engineering adımlarına, baseline modellerden XGBoost optimizasyonuna, threshold seçiminden SHAP ile açıklanabilirliğe ve en sonunda *FastAPI + Streamlit* ile deployment’a kadar tüm süreci kapsar.

## Problem Tanımı

*Amaç:*  
Bir müşterinin, önümüzdeki *2 yıl içinde ciddi finansal gecikme (serious delinquency) yaşama olasılığını* tahmin etmek.

- *Hedef değişken:* SeriousDlqin2yrs  
  - 0 → 2 yıl içinde ciddi finansal gecikme yok  
  - 1 → 2 yıl içinde ciddi finansal gecikme var  

Bankacılık tarafında bu metrik, pratikte *default riski için güçlü bir temsilci değişken* olarak kullanılır; model de iş açısından “kredi geri ödememe riski”ni yönetmek için tasarlanmıştır.

*İş (business) açısından kritik noktalar:*

- *Yanlış pozitifler (False Positive):*  
  Aslında iyi olup reddedilen müşteriler → gelir kaybı, müşteri memnuniyetsizliği.
- *Yanlış negatifler (False Negative):*  
  Aslında kötü olup onaylanan müşteriler → kredi kayıpları, artan risk.

Bu nedenle model sadece teknik metriklere (ROC-AUC vb.) göre değil, *iş gereksinimlerine göre* de değerlendirilmiştir.

## Veri Seti

- *Kaynak:* [Give Me Some Credit – Kaggle Yarışması](https://www.kaggle.com/c/GiveMeSomeCredit)
- *Satır sayısı:* ~150.000
- *Ham feature sayısı:* 10 (hedef hariç)
- *Format:* Tabular CSV
- *Sınıf dağılımı:*
  - Yaklaşık %93 -> 0 (default yok)
  - Yaklaşık %7 -> 1 (default var) → *class imbalance* mevcut

*Örnek değişkenler:*

- RevolvingUtilizationOfUnsecuredLines – Limit kullanım oranı
- age – Müşteri yaşı
- MonthlyIncome – Aylık gelir
- NumberOfTime30-59DaysPastDueNotWorse – 30–59 gün gecikme sayısı
- NumberOfTimes90DaysLate – 90+ gün gecikme sayısı
- DebtRatio – Toplam borç / gelir oranı
- NumberOfOpenCreditLinesAndLoans – Açık kredi hattı sayısı
- NumberRealEstateLoansOrLines – Gayrimenkul kredisi sayısı
- NumberOfDependents – Bakmakla yükümlü kişi sayısı

*Veri sözlüğü:* data/Data Dictionary.xls dosyasında bulunmaktadır.

## Genel Yaklaşım ve Akış

Proje aşağıdaki adımlarla ilerler:

1. **EDA (01_eda.ipynb)**  
   Hedef dağılımı, eksik değerler, uç değerler, temel korelasyonlar.
2. **Veri Temizleme (02_data_cleaning.ipynb)**  
   Hatalı yaş değerleri, eksik gelir/dependent bilgileri, delinquency outlier’ları.
3. **Feature Engineering (03_feature_engineering.ipynb)**  
   Delinquency özetleri, risk flag’leri, binning, etkileşim ve domain feature’ları.
4. **Baseline Modeller (04_baseline.ipynb)**  
   Logistic Regression ve Random Forest ile ilk referans skorlar.
5. **XGBoost + Optimizasyon (05_xgboost.ipynb)**  
   Hyperparameter tuning, threshold optimizasyonu, SHAP analizi.
6. **Final Pipeline (06_final_pipeline.ipynb)**  
   Ham veriden final metriklere tek notebook’ta uçtan uca akış özeti.
7. **Deployment (src/ + app/)**  
   src/ içinde training / inference pipeline’ları,  
   app/ içinde FastAPI ve Streamlit arayüzü.

Detaylı yazılı açıklamalar docs/ klasöründedir (EDA, baseline, FE, model opt, evaluation, pipeline, monitoring).


## Doğrulama (Validation) Şeması

### 80% Train – 20% Validation (Stratified Split)

- *Train set:* 120.000 gözlem  
- *Validation set:* 30.000 gözlem  
- train_test_split(..., test_size=0.2, stratify=y, random_state=42)

*Neden bu şema?*

- *Basit ve hızlı:* Kredi riski problemlerinde yaygın kullanılan bir yaklaşım.
- *Stratified:* Sınıf dengesizliği (%7 default) nedeniyle sınıf oranlarının korunması kritik.
- *Tutarlılık:* Baseline ve final model karşılaştırmaları aynı split üzerinden yapıldı.
- *Yeterli validation boyutu:* 30k gözlem, threshold tuning ve SHAP analizi için fazlasıyla yeterli.

Not: Ayrı bir hold-out test seti ayrılmadı. Veri boyutu büyük olduğu için ek bir test setinin metrikleri anlamlı ölçüde değiştirmesi beklenmedi. Daha ileri bir iterasyonda, model seçimi için validation, son rapor için ise ek bir *dokunulmamış test seti* kurgulanması metodolojik olarak bir adım ileri olacaktır (bkz. docs/evaluation.md).


## Feature Engineering Özeti

Tüm feature engineering pipeline’ı src/data_preprocessing.py içindeki prepare_training fonksiyonunda toplanmıştır.

### 1. Temel Temizlik (clean_basic)

- Gereksiz ID kolonu (Unnamed: 0) kaldırılır.
- age == 0 hatası median ile düzeltilir.
- MonthlyIncome ve NumberOfDependents için median imputasyonu yapılır.
- Delinquency kolonlarındaki aşırı uç değerler *üstten cap* edilir (örneğin 98 → 10).

### 2. Çekirdek Sayısal Feature’lar (add_core_numeric_features)

- log1p dönüşümleri:
  - RevolvingUtilizationOfUnsecuredLines_log1p
  - DebtRatio_log1p
  - MonthlyIncome_log1p
- DebtToIncomeRatio = DebtRatio / MonthlyIncome
- HighUtilizationFlag (kredi kullanım oranı ≥ 1.0 ise 1).

### 3. Delinquency Feature’ları (add_delinquency_features)

- TotalDelinquency (tüm gecikme sayılarının toplamı)
- EverDelinquent (hiç gecikme yaşadı mı?)
- Ever90DaysLate (90+ gün gecikme bayrağı)
- MultipleDelinquencyFlag (toplam gecikme ≥ 2)
- DelinquencySeverityScore (30–59, 60–89, 90+ gecikmeleri ağırlıklı skor).

### 4. Risk Flag’leri (add_risk_flags)

- Örneğin yüksek borç yükünü işaretleyen HighDebtFlag (DebtToIncomeRatio’un üst quantile’ı vb.).

### 5. Binning / Segmentasyon (add_binning_features)

- AgeBin (18–30, 31–45, 46–60, 60+)
- IncomeBin (0–3k, 3–6k, 6–10k, 10k+)
- UtilizationBin (0–30%, 30–70%, 70–100%, 100%+)
- DelinqBin (0, 1, 2–3, 4+)

### 6. Etkileşim Feature’ları (add_interaction_features)

- Utilization_x_DebtRatio
- Income_x_Age
- Delinq_x_Utilization
- OpenLines_x_RealEstate
- HighUtil_x_DebtRatio

### 7. Domain-Driven Feature’lar (add_domain_features)

- EffectiveDebtLoad = DebtRatio * MonthlyIncome
- CreditLineDensity = NumberOfOpenCreditLinesAndLoans / age
- RealEstateExposure = NumberRealEstateLoansOrLines * DebtRatio
- FinancialStressIndex = log1p(DebtRatio * RevolvingUtilizationOfUnsecuredLines)

### 8. Feature Selection (apply_feature_selection)

- Yüksek korelasyonlu / yüksek VIF’li veya zayıf sinyalli bazı kolonlar düşürülür:
  - Ham delinquency kolonları (yerine DelinquencySeverityScore tutulur)
  - DebtRatio, Income_x_Age, MonthlyIncome_log1p, CreditLineDensity vb.

*Sonuç:* Yaklaşık *27 feature*’dan oluşan final feature seti.

> Not: FE sırasında kullanılan bazı istatistikler (median, quantile vb.) tüm veri üzerinde hesaplanmıştır. Bu, teorik olarak hafif bir data leakage kaynağıdır; ancak hedef kullanılmadığı ve veri seti büyük olduğu için pratik etki sınırlı kabul edilmiştir.  
> Daha ileri bir iterasyonda bu istatistiklerin de sklearn pipeline içinde yalnızca train set üzerinde fit edilmesi planlanabilir (bkz. docs/feature_eng.md).


## Baseline Modeller

04_baseline.ipynb ve docs/baseline.md içinde detaylı anlatılmaktadır.

- *Logistic Regression* (numeric-only, class_weight="balanced")
- *Random Forest* (non-lineer yapı testi için)

Özet:

- Her iki model de ROC-AUC ≈ 0.85 civarında performans verir.
- Logistic Regression daha yüksek recall, Random Forest biraz daha iyi F1 sunar.
- Veri non-lineer yapıda olduğu için, daha güçlü bir *gradient boosting* modeline (XGBoost) geçmek mantıklı bulunmuştur.


## Final Model: XGBoost + Threshold

Final model; 05_xgboost.ipynb, src/pipeline.py ve docs/model_opt.md / docs/evaluation.md içinde detaylıdır.

### Model

- *Algoritma:* XGBoost (tree-based gradient boosting)
- *Pipeline:* ColumnTransformer (numeric passthrough + OneHotEncoder) + XGBoost
- *Class imbalance:* scale_pos_weight ≈ 13.96 (negatif/pozitif oranına göre)

### Hyperparameter Optimizasyonu

- Yöntem: RandomizedSearchCV (3-fold stratified CV)
- Aranan parametreler:
  - n_estimators
  - max_depth
  - learning_rate
  - subsample
  - colsample_bytree
  - min_child_weight
- Seçilen en iyi kombinasyon, config.py altında XGBoost için tanımlanan parametre seti
  (ör. XGB_DEFAULT_PARAMS / MODEL_PARAMS) olarak saklanmıştır.

### Threshold Optimizasyonu

- Validation set üzerinde 0.10–0.90 aralığında farklı threshold’lar denenmiştir.
- Hedef:  
  - Teknik olarak *F1 skorunu maksimize etmek*,  
  - İş tarafında ise makul *approval rate* ve düşük *bad rate in approved* elde etmek.
- Seçilen threshold: *0.81*


## Nihai Performans (Validation Set)

docs/evaluation.md içinden özet:

### Baseline vs Final (Validation)

| Model                        | ROC-AUC    | Precision | Recall | F1-score    |
|----------------------------- |----------- |---------- |--------|-------------|
| Logistic Regression          | 0.8622     | 0.2293    | 0.7456 | 0.3508      |
| Random Forest                | 0.8501     | 0.4836    | 0.3017 | 0.3716      |
| *XGBoost (Final, th=0.81)*   | *0.8699*   | *0.4225   | 0.4788 | **0.4489*   |

*Öne çıkan noktalar:*

- F1 skoru baseline modellere göre *%30 civarı iyileşmiştir.*
- Precision yaklaşık *iki katına* çıkmıştır (≈ 0.22 → ≈ 0.42).
- Recall, daha yüksek precision ve daha düşük bad rate hedefi nedeniyle bir miktar düşmüş, bu bilinçli bir *trade-off* olarak seçilmiştir.
- Approval rate, bad rate ve catch rate metrikleri bankacılık açısından makul bir denge sunmaktadır (bkz. docs/evaluation.md).


## Açıklanabilirlik (SHAP)

SHAP analizi 05_xgboost.ipynb ve docs/evaluation.md içinde detaylıdır.

*En önemli feature’lardan bazıları:*

- RevolvingUtilizationOfUnsecuredLines
- Delinq_x_Utilization
- EverDelinquent
- DelinquencySeverityScore
- age
- MonthlyIncome
- DebtToIncomeRatio
- EffectiveDebtLoad
- NumberOfOpenCreditLinesAndLoans
- RealEstateExposure

*Business yorumu (özet):*

- Geçmiş gecikme (özellikle 90+ gün), yüksek limit kullanımı, yüksek borç/gelir oranı ve düşük gelir, default riskini ciddi şekilde artırır.
- Domain tabanlı feature’lar (EffectiveDebtLoad, FinancialStressIndex vb.) modelin riskli segmentleri daha keskin ayırt etmesine yardımcı olur.

## Kod ve Pipeline Yapısı

### src/ klasörü

- **config.py**  
  - Proje path’leri (DATA_DIR, MODELS_DIR vb.)  
  - *Business kuralları:* threshold, minimum precision/recall, hedef approval aralığı vb.  
  - Model parametreleri (XGB_DEFAULT_PARAMS / MODEL_PARAMS, SCALE_POS_WEIGHT).

- **data_preprocessing.py**  
  - *Ana temizlik + feature engineering pipeline’ı*  
  - prepare_training(df) → ham Kaggle formatındaki veriden final feature tablosunu üretir.

- **feature_engineering.py**  
  - Deneme amaçlı / alternatif FE fonksiyonları (asıl training pipeline’ın kaynağı değil; daha çok notebook denemelerinin script özeti).

- **predict.py**  
  - predict_from_df(df) →  
    - models/xgboost_credit_risk_final.pkl dosyasını yükler,  
    - Threshold uygulayarak 0/1 tahmin ve olasılık döndürür.  
  - Girdi: FE sonrası hazır feature setine sahip DataFrame.

- **inference.py**  
  - predict_from_raw(df) →  
    - Ham Kaggle formatındaki DataFrame’i alır,  
    - prepare_training ile temizlik + FE uygular,  
    - Ardından predict_from_df ile tahmin üretir.  
  - API ve Streamlit bu fonksiyonu kullanır.

- **pipeline.py**  
  - train_pipeline() → Eğitim pipeline’ı (ham veriden model eğitimine kadar).  
  - inference_pipeline() → Batch inference pipeline’ı.  
  - CLI kullanım:
    - python -m src.pipeline train  
    - python -m src.pipeline predict veya sadece python -m src.pipeline


## Deployment: FastAPI + Streamlit

### FastAPI – app/api.py

- GET /health → sağlık kontrolü
- POST /predict → JSON içinde records listesi alır; her kayıt Kaggle ham formatındadır.
- İçeride src.inference.predict_from_raw fonksiyonunu kullanır.
- Hatalı veya eksik feature durumunda anlamlı hata mesajları döner.

Başlatmak için:

bash
uvicorn app.api:app --reload
# http://127.0.0.1:8000/docs


### Streamlit – app/streamlit_app.py

- CSV upload ile ham Kaggle formatında dosya alır (örn. data/test_sample_raw.csv).
- Veri önizleme, satır/kolon sayısı, kolon listesi gösterir.
- predict_from_raw ile tahmin üretir:
  - Ortalama risk olasılığı
  - Yüksek / düşük riskli müşteri sayıları
  - Detaylı sonuç tablosu (Default_Probability, Predicted_Label, Risk_Segment)
- Sonuçları CSV olarak indirme imkânı verir.
- Gelişmiş bir dashboard sekmesiyle:
  - Risk dağılım histogramı
  - Risk segmentasyonu (düşük / orta / yüksek / çok yüksek)
  - Yaş–gelir scatter plot (risk segmentlerine göre)
  - Filtrelenebilir detay tablosu  
  sunar.

Başlatmak için:

bash
streamlit run app/streamlit_app.py


## Testler

tests/ klasörü:

- **test_sample.py**  
  - Model dosyasının varlığını ve yüklenebilirliğini kontrol eder.  
  - predict_from_df’nin temel bir örnek üzerinde beklendiği gibi çalıştığını test eder.

- **test_edge_inputs.py**  
  - Hedef kolon olmadan tahmin  
  - Fazladan kolon içeren veri seti  
  - Tek satırlık (single row) veri  
  gibi edge case senaryolarını test eder.

Testleri çalıştırmak için:

bash
python -m pytest -q


## 📁 Proje Yapısı

text
credit-risk-model/
├── app/
│   ├── api.py              # FastAPI – REST API (health + /predict)
│   └── streamlit_app.py    # Streamlit UI for CSV-based scoring
│
├── data/
│   ├── cs-training.csv
│   ├── cs-test.csv
│   ├── cs-training-clean.csv
│   ├── training_prepared.csv
│   ├── test_sample_raw.csv
│   └── Data Dictionary.xls
│
├── docs/
│   ├── business_context.md
│   ├── eda.md
│   ├── baseline.md
│   ├── feature_eng.md
│   ├── model_opt.md
│   ├── evaluation.md
│   ├── pipeline.md
│   └── monitoring_plan.md
│
├── models/
│   └── xgboost_credit_risk_final.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline.ipynb
│   ├── 05_xgboost.ipynb
│   └── 06_final_pipeline.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── pipeline.py
│   ├── predict.py
│   └── inference.py
│
├── tests/
│   ├── test_sample.py
│   └── test_edge_inputs.py
│
├── requirements.txt
└── README.md


## Çalıştırma Adımları (Özet)

### 1. Ortam Kurulumu

bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# veya
.venv\Scripts\activate

pip install -r requirements.txt


### 2. Model Eğitimi

bash
python -m src.pipeline train


### 3. Batch Inference

bash
python -m src.pipeline predict
# veya
python -m src.pipeline


### 4. API

bash
uvicorn app.api:app --reload
# http://127.0.0.1:8000/docs


### 5. Streamlit Dashboard

bash
streamlit run app/streamlit_app.py


Streamlit arayüzünü hızlıca test etmek için data/test_sample_raw.csv dosyasını yükleyip *Tahmin Et* butonuna basabilirsiniz; sonuç tablosunda her satır için Default_Probability, Predicted_Label ve Risk_Segment kolonları görünmelidir.


## Sınırlılıklar ve Gelecek Çalışmalar

- Feature engineering istatistiklerinin (median, quantile vb.) sadece train set üzerinde fit edildiği tam bir sklearn pipeline’a taşınması.
- Ayrı bir *hold-out test seti* ile ek performans doğrulaması ve model seçiminin yeniden gözden geçirilmesi.
- Model kalibrasyonu ve skor kart (scorecard) formatına dönüştürme.
- Gerçek bir MLOps altyapısında (MLflow, dashboard vb.) monitoring planının hayata geçirilmesi.
- Streamlit arayüzünün daha kurumsal bir kredi başvuru paneline dönüştürülmesi.


İyileştirme önerileri veya sorular için repo üzerinden issue açabilirsiniz.