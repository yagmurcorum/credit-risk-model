# tests/test_edge_inputs.py

import pandas as pd

from src.config import TRAINING_PREPARED
from src.predict import predict_from_df


def _load_base_df(n_rows: int = 10) -> pd.DataFrame:
    """
    Edge-case testlerinde kullanılacak küçük bir örnek seti yükler.
    """
    df = pd.read_csv(TRAINING_PREPARED).head(n_rows)
    assert len(df) > 0, "training_prepared.csv boş geldi!"
    return df


def test_model_works_without_target_column():
    """
    3.1 - Production senaryosu:
    SeriousDlqin2yrs kolonu olmadan da tahmin alabiliyor muyuz?
    """
    df_full = _load_base_df()

    if "SeriousDlqin2yrs" in df_full.columns:
        df_no_target = df_full.drop(columns=["SeriousDlqin2yrs"])
    else:
        df_no_target = df_full.copy()

    y_pred, y_proba = predict_from_df(df_no_target)

    # Boyutlar doğru mu?
    assert len(y_pred) == len(df_no_target)
    assert len(y_proba) == len(df_no_target)

    # Olasılıklar 0–1 aralığında mı?
    assert (y_proba >= 0).all() and (y_proba <= 1).all()


def test_model_works_with_extra_columns():
    """
    3.2 - Ekstra kolon (CustomerID) eklendiğinde model bozuluyor mu?
    """
    df_full = _load_base_df()

    if "SeriousDlqin2yrs" in df_full.columns:
        df_no_target = df_full.drop(columns=["SeriousDlqin2yrs"])
    else:
        df_no_target = df_full.copy()

    # Fazladan bir ID kolonu ekleyelim
    df_extra = df_no_target.copy()
    df_extra["CustomerID"] = range(len(df_extra))

    y_pred, y_proba = predict_from_df(df_extra)

    assert len(y_pred) == len(df_extra)
    assert len(y_proba) == len(df_extra)


def test_single_row_prediction():
    """
    3.3 - Tek müşteri (tek satır) için tahmin alabiliyor muyuz?
    """
    df_full = _load_base_df()

    if "SeriousDlqin2yrs" in df_full.columns:
        df_no_target = df_full.drop(columns=["SeriousDlqin2yrs"])
    else:
        df_no_target = df_full.copy()

    single = df_no_target.head(1)

    y_pred, y_proba = predict_from_df(single)

    assert len(single) == 1
    assert len(y_pred) == 1
    assert len(y_proba) == 1
    assert 0.0 <= float(y_proba[0]) <= 1.0
