# src/predict.py

"""
Bu modül, eğitilmiş final XGBoost kredi risk modelini kullanarak
DataFrame üzerinden tahmin almak için yardımcı fonksiyon sağlar.

Final model dosyası:
    models/xgboost_credit_risk_final.pkl

Model paketi yapısı:
    {
        "model":    Pipeline veya estimator (XGBClassifier),
        "threshold": float (ör. 0.81),
        "features":  list[str] (isteğe bağlı – açıklama amaçlı)
    }
"""

from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import FINAL_MODEL


def _load_model_package() -> dict:
    """
    Kaydedilmiş final model paketini yükler.

    Hata alırsan:
    - src.config içindeki FINAL_MODEL yolunu,
    - İlgili .pkl dosyasının gerçekten var olup olmadığını

    kontrol et.
    """
    model_package = joblib.load(FINAL_MODEL)
    return model_package


def predict_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Girdi olarak verilen DataFrame üzerinden tahmin üretir.

    Beklenen:
    - df, 02_cleaning + 03_feature_engineering sonrası oluşan
      training_prepared.csv ile aynı feature kolonlarına sahip olmalı
      (SeriousDlqin2yrs kolonunun olması zorunlu değil).

    Adımlar:
    - Varsa hedef kolonu (SeriousDlqin2yrs) düşülür.
    - Final model dosyası içindeki model ile predict_proba çağrılır.
    - Kayıtlı threshold'a göre 0/1 tahmin üretilir.

    Dönen:
        y_pred  : (n,)  -> 0/1 tahminler
        y_proba : (n,)  -> default olasılığı
    """
    df_model = df.copy()

    # Hedef kolonu yanlışlıkla geldiyse düşelim (data leakage önleme)
    target_col = "SeriousDlqin2yrs"
    if target_col in df_model.columns:
        df_model = df_model.drop(columns=[target_col])

    model_package = _load_model_package()
    model = model_package["model"]
    threshold = model_package["threshold"]

    # Not:
    # model, 05_xgboost.ipynb içinde ColumnTransformer + XGBoost
    # pipeline'ı olarak kaydedildiği için burada ekstra scaler / OHE
    # yapılmaz; tüm preprocessing pipeline içinde halledilir.
    y_proba = model.predict_proba(df_model)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return y_pred, y_proba
