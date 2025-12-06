# src/inference.py

"""
Eğitilmiş kredi risk modelinden tahmin alma ve gerekli preprocessing işlemleri.

Bu modül, ham (Kaggle formatındaki) veriyi alır,
data_preprocessing.prepare_training üzerinden temizleyip
feature engineering adımlarını uygular ve sonucu
modelin beklediği feature setine dönüştürür. Sonrasında
final model üzerinden tahmin üretir.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from src.data_preprocessing import prepare_training
from src.predict import predict_from_df


def predict_from_raw(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ham Kaggle formatındaki veriyi alır, preprocessing yapar ve tahmin döner.
    
    Bu fonksiyon, API veya başka bir sistemden gelen ham veriyi
    model için hazır hale getirir ve tahmin üretir.
    
    Args:
        df: Ham veri DataFrame (Kaggle formatındaki kolonlarla)
            - RevolvingUtilizationOfUnsecuredLines
            - age
            - MonthlyIncome
            - NumberOfTime30-59DaysPastDueNotWorse
            - DebtRatio
            - NumberOfOpenCreditLinesAndLoans
            - NumberOfTimes90DaysLate
            - NumberRealEstateLoansOrLines
            - NumberOfTime60-89DaysPastDueNotWorse
            - NumberOfDependents
            - (SeriousDlqin2yrs opsiyonel - varsa düşülür)
    
    Returns:
        y_pred  : (n,) -> 0/1 tahminler
        y_proba : (n,) -> default olasılığı
    """
    # 1) Ham veriyi preprocessing pipeline'ından geçir
    df_prepared = prepare_training(df)

    # 2) Final model dosyası üzerinden tahmin al
    y_pred, y_proba = predict_from_df(df_prepared)
    
    return y_pred, y_proba


if __name__ == "__main__":
    # Hızlı yerel test için mini örnek
    from src.config import RAW_TRAIN

    print("Ham veri yükleniyor...")
    df_raw = pd.read_csv(RAW_TRAIN)

    print(f"Ham veri shape: {df_raw.shape}")
    print("İlk 5 satır:")
    print(df_raw.head())

    print("\nTahmin yapılıyor (ilk 10 satır için)...")
    y_pred, y_proba = predict_from_raw(df_raw.head(10))

    print("\nTahmin sonuçları:")
    print(f"Predicted labels: {y_pred}")
    print(f"Probabilities   : {y_proba}")
