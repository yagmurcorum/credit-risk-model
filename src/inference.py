# src/inference.py

"""
Eğitilmiş modelden tahmin alma ve gerekli preprocessing işlemleri.

Bu modül, bootcamp gereksinimlerine uygun olarak inference.py dosyasıdır.
Ham veri formatından (Kaggle formatı) tahmin almak için gerekli tüm
preprocessing adımlarını içerir.
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
        y_pred: (n,) -> 0/1 tahminler
        y_proba: (n,) -> default olasılığı
    """
    # Ham veriyi preprocessing pipeline'ından geçir
    df_prepared = prepare_training(df)
    
    # Tahmin al
    y_pred, y_proba = predict_from_df(df_prepared)
    
    return y_pred, y_proba


if __name__ == "__main__":
    # Test için
    from src.config import RAW_TRAIN
    
    print("Ham veri yükleniyor...")
    df_raw = pd.read_csv(RAW_TRAIN)
    
    print(f"Ham veri shape: {df_raw.shape}")
    print("İlk 5 satır:")
    print(df_raw.head())
    
    print("\nTahmin yapılıyor...")
    y_pred, y_proba = predict_from_raw(df_raw.head(10))  # İlk 10 satır için test
    
    print(f"\nTahmin sonuçları:")
    print(f"Predicted labels: {y_pred}")
    print(f"Probabilities: {y_proba}")

