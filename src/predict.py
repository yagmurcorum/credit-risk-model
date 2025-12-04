# src/predict.py

"""
Bu modül, eğitilmiş final XGBoost modelini kullanarak
DataFrame üzerinden tahmin almak için yardımcı fonksiyon sağlar.

Final model:
    models/xgboost_credit_risk_final.pkl

Yapısı:
    {
        "model":    Pipeline veya estimator (xgb_best),
        "threshold": float (ör. 0.81),
        "features":  list[str] (isteğe bağlı – açıklama amaçlı)
    }
"""

from typing import Tuple
import joblib
import numpy as np
import pandas as pd
from src.config import FINAL_MODEL


def _load_artifact() -> dict:
    """
    Kaydedilmiş final modeli yükler.
    Hata alırsan önce FINAL_MODEL yolunu ve dosyanın gerçekten var
    olup olmadığını kontrol et.
    """
    artifact = joblib.load(FINAL_MODEL)
    return artifact


def predict_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Girdi olarak verilen DataFrame üzerinden tahmin üretir.

    Beklenen:
    - df, 03_feature_engineering + 02_cleaning sonrası oluşan
      training_prepared.csv ile aynı feature kolonlarına sahip olmalı
      (SeriousDlqin2yrs kolonunun olması zorunlu değil).

    Adımlar:
    - Varsa hedef kolonu (SeriousDlqin2yrs) düşülür.
    - Final model dosyası içindeki model ile predict_proba çağrılır.
    - threshold'a göre 0/1 tahmin üretilir.

    Dönen:
        y_pred  : (n,)  -> 0/1 tahminler
        y_proba : (n,)  -> default olasılığı
    """
    df_model = df.copy()

    # Hedef kolonu yanlışlıkla geldiyse düşelim (data leakage önleme)
    target_col = "SeriousDlqin2yrs"
    if target_col in df_model.columns:
        df_model = df_model.drop(columns=[target_col])

    artifact = _load_artifact()
    model = artifact["model"]
    threshold = artifact["threshold"]

    # NOT:
    # model zaten 05_xgboost.ipynb içinde ColumnTransformer + XGBoost
    # pipeline'ı olarak kaydedildiği için burada ekstra scaler / OHE
    # vs. yapmamıza gerek yok; pipeline kendi içinde hallediyor.
    y_proba = model.predict_proba(df_model)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return y_pred, y_proba
