# src/pipeline.py

"""
Tüm ML akışının gerçekleştiği final akış scripti.

Bu script bootcamp gereksinimlerine uygun olarak:
- Training pipeline: Ham veriden model eğitimine kadar tüm süreç
- Inference pipeline: Eğitilmiş model ile tahmin alma

Kullanım:
    # Training
    python -m src.pipeline train
    
    # Inference
    python -m src.pipeline predict
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    RAW_TRAIN,
    TRAINING_PREPARED,
    DATA_DIR,
    MODELS_DIR,
    FINAL_MODEL,
    SEED,
    MODEL_PARAMS,
    SCALE_POS_WEIGHT,
    DEFAULT_THRESHOLD,
)
from src.data_preprocessing import prepare_training
from src.predict import predict_from_df


def train_pipeline(
    input_path: Path = RAW_TRAIN,
    output_model_path: Path = FINAL_MODEL,
) -> dict:
    """
    Tüm ML akışını gerçekleştiren training pipeline'ı.
    
    Adımlar:
    1. Ham veriyi yükle
    2. Data cleaning + Feature engineering uygula
    3. Train/validation split
    4. Preprocessing pipeline oluştur (ColumnTransformer)
    5. XGBoost modeli eğit (hyperparameter optimization ile)
    6. Threshold optimization
    7. Model kaydet
    
    Args:
        input_path: Ham eğitim verisi yolu
        output_model_path: Model kayıt yolu
    
    Returns:
        Model artifact dictionary
    """
    print("=" * 60)
    print("TRAINING PIPELINE BAŞLADI")
    print("=" * 60)
    
    # 1. Veri yükleme ve preprocessing
    print("\n1. Veri yükleniyor ve preprocessing uygulanıyor...")
    df_raw = pd.read_csv(input_path)
    print(f"   Ham veri shape: {df_raw.shape}")
    
    df_prepared = prepare_training(df_raw)
    print(f"   Hazırlanmış veri shape: {df_prepared.shape}")
    
    # 2. Train/validation split
    print("\n2. Train/validation split yapılıyor...")
    TARGET_COL = "SeriousDlqin2yrs"
    X = df_prepared.drop(columns=[TARGET_COL])
    y = df_prepared[TARGET_COL]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"   Train set: {X_train.shape[0]} gözlem")
    print(f"   Validation set: {X_val.shape[0]} gözlem")
    
    # 3. Preprocessing pipeline
    print("\n3. Preprocessing pipeline oluşturuluyor...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print(f"   Sayısal feature'lar: {len(numeric_cols)}")
    print(f"   Kategorik feature'lar: {len(categorical_cols)}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    
    # 4. XGBoost model
    print("\n4. XGBoost modeli oluşturuluyor...")
    xgb_clf = XGBClassifier(
        **MODEL_PARAMS,
        scale_pos_weight=SCALE_POS_WEIGHT,
    )
    
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", xgb_clf),
        ]
    )
    
    # 5. Hyperparameter optimization (küçük ölçekli)
    print("\n5. Hyperparameter optimization yapılıyor...")
    param_distributions = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.07],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    
    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_distributions,
        n_iter=10,  # Hızlı test için
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=SEED,
    )
    
    xgb_search.fit(X_train, y_train)
    print(f"   En iyi CV ROC-AUC: {xgb_search.best_score_:.4f}")
    
    xgb_best = xgb_search.best_estimator_
    
    # 6. Validation set değerlendirmesi
    print("\n6. Validation set üzerinde değerlendirme...")
    y_val_proba = xgb_best.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= DEFAULT_THRESHOLD).astype(int)
    
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    
    print(f"   ROC-AUC: {val_auc:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall: {val_recall:.4f}")
    print(f"   F1-score: {val_f1:.4f}")
    
    # 7. Feature names (encoding sonrası)
    ohe = xgb_best.named_steps["preprocess"].named_transformers_["cat"]
    ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = list(numeric_cols) + list(ohe_feature_names)
    
    # 8. Model kaydetme
    print("\n7. Model kaydediliyor...")
    MODELS_DIR.mkdir(exist_ok=True)
    
    final_artifact = {
        "model": xgb_best,
        "threshold": DEFAULT_THRESHOLD,
        "features": all_feature_names,
    }
    
    joblib.dump(final_artifact, output_model_path)
    print(f"   Model kaydedildi: {output_model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE TAMAMLANDI")
    print("=" * 60)
    
    return final_artifact


def inference_pipeline(
    input_path: Path = TRAINING_PREPARED,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Inference pipeline: Eğitilmiş model ile tahmin alma.
    
    Args:
        input_path: Tahmin yapılacak veri yolu
        output_path: Sonuçların kaydedileceği yol (opsiyonel)
    
    Returns:
        Tahmin sonuçları DataFrame
    """
    print("=" * 60)
    print("INFERENCE PIPELINE BAŞLADI")
    print("=" * 60)
    
    print(f"\nVeri yükleniyor: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Veri shape: {df.shape}")
    
    print("\nTahmin yapılıyor...")
    y_pred, y_proba = predict_from_df(df)
    
    result_df = df.copy()
    result_df["Predicted_Label"] = y_pred
    result_df["Default_Probability"] = y_proba
    
    if output_path is not None:
        result_df.to_csv(output_path, index=False)
        print(f"\nSonuçlar kaydedildi: {output_path}")
    
    print("\n" + "=" * 60)
    print("INFERENCE PIPELINE TAMAMLANDI")
    print("=" * 60)
    
    return result_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode
        train_pipeline()
    else:
        # Inference mode (default)
        default_output = DATA_DIR / "training_predictions.csv"
        inference_pipeline(TRAINING_PREPARED, default_output)

