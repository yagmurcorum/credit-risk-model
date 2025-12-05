# Pipeline Summary

Bu doküman final pipeline yapısının ve ön işleme stratejisinin özetini içerir.  
Model, `05_xgboost.ipynb` notebook'unda eğitilmiş ve `src/pipeline.py` / `src.inference.py` / `src.predict.py` 
ile hem eğitim hem de inference tarafında kullanılmaktadır.

## Final Pipeline Yapısı

### 1. Veri Yükleme
- **Giriş:** `data/cs-training.csv` (ham veri)
- **Çıkış:** `data/training_prepared.csv` (FE sonrası hazır veri)

### 2. Data Cleaning (`src/data_preprocessing.py::clean_basic`)
- Gereksiz ID kolonu kaldırma (`Unnamed: 0`)
- `age == 0` hatası düzeltme (median ile doldurma)
- `MonthlyIncome` eksik değerleri median ile doldurma
- `NumberOfDependents` eksik değerleri median ile doldurma
- Delinquency kolonlarında uç değerleri cap etme (98 → 10)

### 3. Feature Engineering (`src/data_preprocessing.py::prepare_training`)

**Adım 3.1: Core Numeric Features**
- `RevolvingUtilizationOfUnsecuredLines_log1p`
- `DebtRatio_log1p`
- `DebtToIncomeRatio`
- `HighUtilizationFlag`

**Adım 3.2: Delinquency Features**
- `TotalDelinquency`
- `EverDelinquent`
- `Ever90DaysLate`
- `MultipleDelinquencyFlag`
- `DelinquencySeverityScore`

**Adım 3.3: Risk Flags**
- `HighDebtFlag`

**Adım 3.4: Binning**
- `AgeBin` (18-30, 31-45, 46-60, 60+)
- `IncomeBin` (0-3k, 3-6k, 6-10k, 10k+)
- `UtilizationBin` (0-30%, 30-70%, 70-100%, 100%+)
- `DelinqBin` (0, 1, 2-3, 4+)

**Adım 3.5: Interaction Features**
- `Utilization_x_DebtRatio`
- `Delinq_x_Utilization`
- `OpenLines_x_RealEstate`
- `HighUtil_x_DebtRatio`

**Adım 3.6: Domain Features**
- `EffectiveDebtLoad`
- `RealEstateExposure`
- `FinancialStressIndex`
- `CreditLineDensity`

**Adım 3.7: Feature Selection**
- Ham delinquency kolonları çıkarıldı
- `DebtRatio`, `Income_x_Age`, `MonthlyIncome_log1p`, `CreditLineDensity` çıkarıldı

### 4. Model Eğitimi (`05_xgboost.ipynb`)

**Ön İşleme Pipeline:**
- **Sayısal değişkenler:** Passthrough (ölçeklendirme yok)
- **Kategorik değişkenler:** OneHotEncoder (`handle_unknown="ignore"`)
- **ColumnTransformer** ile birleştirildi

**XGBoost Model:**
- Hyperparameter optimization (RandomizedSearchCV, 3-fold CV)
- Optimal parametreler: `n_estimators=300`, `max_depth=4`, `learning_rate=0.03`, vb.
- `scale_pos_weight=13.96` ile class imbalance yönetimi
- Optimal threshold: 0.81

**Model Kaydetme:**
- Dosya: `models/xgboost_credit_risk_final.pkl`
- Format: `{"model": pipeline, "threshold": 0.81, "features": feature_names}`

### 5. Inference Pipeline

Inference tarafında iki seviyeli yapı kullanılır:

- `src.predict.predict_from_df(df_prepared)`  
  - Girdi: `training_prepared.csv` benzeri, FE sonrası hazır tablo  
  - Kullanım: Testler (`tests/`), training sonrası internal skorlamalar

- `src.inference.predict_from_raw(df_raw)`  
  - Girdi: Ham Kaggle formatındaki veri (10–11 kolon)  
  - Adımlar:
    1. `prepare_training(df_raw)` ile temizlik + FE
    2. `predict_from_df(df_prepared)` çağrısı
  - Kullanım: FastAPI (`app/api.py`), Streamlit UI (`app/streamlit_app.py`)

**Özet Inference Akışı (`predict_from_raw`):**
1. Ham veri yükleme (CSV veya DataFrame)
2. Feature engineering uygulama (`prepare_training`)
3. Modeli yükleme (`joblib.load(FINAL_MODEL)`)
4. Pipeline ile tahmin (`predict_proba`)
5. Threshold (0.81) uygulama → 0/1 karar üretimi
6. Sonuç döndürme (`y_pred`, `y_proba`)

## Final Feature Seti

**Final tablo:**  
- Satır sayısı: 150,000  
- Kolon sayısı: 27 (26 feature + 1 hedef)

**Toplam feature sayısı:** 26 (hedef hariç)

**Sayısal feature'lar (22 adet):**
- Orijinal:
  - `RevolvingUtilizationOfUnsecuredLines`
  - `age`
  - `MonthlyIncome`
  - `NumberOfOpenCreditLinesAndLoans`
  - `NumberRealEstateLoansOrLines`
  - `NumberOfDependents`
- Delinquency türevleri:
  - `EverDelinquent`
  - `Ever90DaysLate`
  - `MultipleDelinquencyFlag`
  - `DelinquencySeverityScore`
- Risk flag'leri:
  - `HighUtilizationFlag`
  - `HighDebtFlag`
- Log dönüşümleri:
  - `RevolvingUtilizationOfUnsecuredLines_log1p`
  - `DebtRatio_log1p`
- Ratio:
  - `DebtToIncomeRatio`
- Interaction:
  - `Utilization_x_DebtRatio`
  - `Delinq_x_Utilization`
  - `OpenLines_x_RealEstate`
  - `HighUtil_x_DebtRatio`
- Domain:
  - `EffectiveDebtLoad`
  - `RealEstateExposure`
  - `FinancialStressIndex`

**Kategorik feature'lar (4 adet):**
- `AgeBin`
- `IncomeBin`
- `UtilizationBin`
- `DelinqBin`

**Encoding sonrası:**  
OneHotEncoder ile kategorik feature'lar genişletilir; toplam feature sayısı ~30 civarına çıkar.

## Ön İşleme Stratejisi

### Eksik Değer Yönetimi
- **Median imputation:**  
  - `MonthlyIncome`  
  - `NumberOfDependents`  
  - `age` için 0 değerleri NaN yapılıp median ile dolduruldu
- Eksik değer oranları düşük olduğu için ekstra “missing flag” feature’ı eklenmedi.

### Outlier Yönetimi
- **Capping:** Delinquency kolonlarında 98 gibi uç değerler 10 seviyesinde sınırlandı.
- **Log dönüşümü:** Sağa çarpık değişkenler için `log1p` uygulandı (`RevolvingUtilizationOfUnsecuredLines`, `DebtRatio`).

### Encoding Stratejisi
- **OneHotEncoder:**  
  - Kategorik feature’lar için (`AgeBin`, `IncomeBin`, `UtilizationBin`, `DelinqBin`)  
  - `handle_unknown="ignore"` → Eğitimde görülmeyen yeni kategoriler inference sırasında hataya yol açmaz.
- **Sayısal feature’lar:**  
  - Tree-based XGBoost modeli kullanıldığı için ek ölçeklendirme yapılmadı.

### Feature Selection Stratejisi
- **Korelasyon analizi:**  
  - Yüksek korelasyonlu feature çiftleri tespit edildi.
- **VIF analizi:**  
  - Multicollinearity kontrolü yapıldı.
- **Domain bilgisi:**  
  - Daha anlamlı ve stabil feature’lar tercih edildi (örn. `EffectiveDebtLoad` vs. ham `DebtRatio`).
- Çıkarılanlar:
  - Ham delinquency kolonları (`NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`, `TotalDelinquency`)
  - `DebtRatio`, `Income_x_Age`, `MonthlyIncome_log1p`, `CreditLineDensity`

## Neden Bu Model ve Feature Seti Seçildi?

### Model Seçimi
- **XGBoost:**
  - Non-lineer ilişkileri yakalayabilen güçlü bir tree-based model
  - Class imbalance için yerleşik destek (`scale_pos_weight`)
  - SHAP ile açıklanabilirlik imkânı
- **Threshold 0.81:**
  - Validation set üzerinde F1 skorunu maksimize ediyor
  - İş gereksinimleri açısından kabul edilebilir precision–recall dengesi sağlıyor

### Feature Seti Seçimi
- **Delinquency feature’ları:**  
  - Gecikme geçmişi, modelin gördüğü en güçlü risk sinyallerinden biri (SHAP ile doğrulandı).
- **Risk flag’leri:**  
  - Binary flag’ler, modelin segmentler arasında keskin ayrım yapmasına yardımcı oluyor.
- **Binning:**  
  - Non-lineer etkileri yakalamak ve iş tarafının okuyabileceği segmentler üretmek için kullanıldı.
- **Interaction feature’ları:**  
  - Kart kullanımı × borç oranı gibi kombinasyonlar ile karmaşık risk kalıpları yakalandı.
- **Domain feature’ları:**  
  - `EffectiveDebtLoad`, `RealEstateExposure`, `FinancialStressIndex` gibi değişkenlerle finansal uzmanlık bilgisi modele eklendi.

### Çıkarılan Feature'lar
- **Ham delinquency kolonları:**  
  - Gecikme şiddeti ve yoğunluğu `DelinquencySeverityScore` ve türev feature’lar ile daha kompakt şekilde temsil ediliyor.
- **Yüksek korelasyonlu feature’lar:**  
  - Multicollinearity riskini azaltmak için sadeleştirildi.
- **Zayıf sinyal veren feature’lar:**  
  - VIF + korelasyon analizi + domain yorumu sonucunda elendi.

## Pipeline Kullanımı

### Eğitim Pipeline (Notebook)

    # 05_xgboost.ipynb içinde
    import pandas as pd
    from src.data_preprocessing import prepare_training

    df_raw = pd.read_csv("data/cs-training.csv")
    df_prepared = prepare_training(df_raw)
    # ... model eğitimi, CV, threshold tuning, SHAP, kayıt ...

### Eğitim & Inference Pipeline (Script)

    # src/pipeline.py

    from src.pipeline import train_pipeline, inference_pipeline
    from src.config import RAW_TRAIN, TRAINING_PREPARED, DATA_DIR

    # Eğitim
    train_pipeline(input_path=RAW_TRAIN)

    # Eğitim sonrası skorlamalar (FE sonrası tablo üzerinden)
    out_path = DATA_DIR / "training_predictions.csv"
    inference_pipeline(input_path=TRAINING_PREPARED, output_path=out_path)

### Inference Pipeline (Model Dosyası Üzerinden)

    # src/predict.py
    import pandas as pd
    from src.predict import predict_from_df

    df_prepared = pd.read_csv("data/training_prepared.csv")
    y_pred, y_proba = predict_from_df(df_prepared)

### API Pipeline

    # app/api.py (özet)

    from fastapi import FastAPI
    from src.inference import predict_from_raw

    app = FastAPI()

    @app.post("/predict")
    def predict(batch: CustomerBatch):
        df = pd.DataFrame(batch.records)
        y_pred, y_proba = predict_from_raw(df)
        return {
            "n_records": len(df),
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist(),
        }

### Streamlit Dashboard

    # app/streamlit_app.py (özet)

    from src.inference import predict_from_raw

    # Kullanıcı CSV yükler → df
    y_pred, y_proba = predict_from_raw(df)
    # Sonuçlar dashboard üzerinde görselleştirilir

## Deployment Notları

- **Model formatı:** `joblib` ile kaydedilen Python objesi  
- **Pipeline yapısı:** `ColumnTransformer` + XGBoost → tek bir pipeline objesi olarak saklanır  
- **Threshold:** Model dosyasının içinde (`"threshold": 0.81`) tutulur  
- **Feature sırası:** Pipeline içindeki `ColumnTransformer` tarafından otomatik yönetilir  
- **Yeni kategoriler:** OneHotEncoder `handle_unknown="ignore"` ile, eğitimde görülmeyen kategoriler inference sırasında hata üretmez  
- **Kullanım alanları:**
  - Batch skorlamalar (`src/pipeline.py`)
  - REST API (`app/api.py`)
  - Streamlit dashboard (`app/streamlit_app.py`)
