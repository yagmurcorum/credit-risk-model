# app/api.py

"""
FastAPI tabanlı REST API:

- GET  /health  -> Sağlık kontrolü
- POST /predict -> JSON input ile kredi riski tahmini

Çalıştırmak için:
    uvicorn app.api:app --reload

Not:
- Tahmin için src.predict.predict_from_df fonksiyonu kullanılır.
- Input formatı (örnek):
    {
        "records": [
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.12,
                "age": 45,
                "MonthlyIncome": 5000,
                ...
            },
            ...
        ]
    }
"""

from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import predict_from_raw


app = FastAPI(title="Credit Risk Scoring API")


class CustomerBatch(BaseModel):
    records: List[Dict[str, Any]]


@app.get("/health")
def health_check():
    """
    Sağlık kontrolü endpoint'i.
    """
    return {"status": "ok", "message": "Credit Risk API is running"}


@app.post("/predict")
def predict(batch: CustomerBatch):
    """
    Tahmin endpoint'i.

    Beklenen input:
    {
        "records": [
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.12,
                "age": 45,
                "MonthlyIncome": 5000,
                ...
            }
        ]
    }
    """
    if not batch.records:
        raise HTTPException(
            status_code=400,
            detail="`records` listesi boş. En az bir müşteri kaydı göndermelisiniz.",
        )

    df = pd.DataFrame(batch.records)

    try:
        # Ham Kaggle formatındaki veriyi preprocessing ile işle ve tahmin al
        y_pred, y_proba = predict_from_raw(df)
    except KeyError as e:
        # Örn: df[features] satırında eksik kolon olursa buraya düşer
        raise HTTPException(
            status_code=400,
            detail=f"Beklenen feature kolonları eksik veya hatalı: {e}",
        )

    return {
        "n_records": len(df),
        "predictions": y_pred.tolist(),
        "probabilities": y_proba.tolist(),
    }

