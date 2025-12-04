# tests/test_sample.py

import pandas as pd
from src.config import FINAL_MODEL, TRAINING_PREPARED
from src.predict import predict_from_df


def test_final_model_file_exists():
    """
    Final model dosyası gerçekten var mı?
    """
    assert FINAL_MODEL.exists(), f"Model dosyası bulunamadı: {FINAL_MODEL}"


def test_predict_from_df_on_small_sample():
    """
    training_prepared.csv'den küçük bir sample alıp
    predict_from_df fonksiyonunun çalıştığını test eder.
    """
    df = pd.read_csv(TRAINING_PREPARED).head(50)

    y_pred, y_proba = predict_from_df(df)

    # Boyut kontrolleri
    assert len(y_pred) == len(df)
    assert len(y_proba) == len(df)

    # Olasılıklar 0-1 aralığında mı?
    assert (y_proba >= 0).all() and (y_proba <= 1).all()
