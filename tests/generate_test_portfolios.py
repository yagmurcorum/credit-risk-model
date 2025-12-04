# tests/generate_test_portfolios.py

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# --- Proje kökünü sys.path'e ekle (src'yi görebilmesi için) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.inference import predict_from_raw

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    # 1. Ham eğitim verisini oku
    raw_path = DATA_DIR / "cs-training.csv"
    print(f"[INFO] Ham veri okunuyor: {raw_path}")
    df_raw = pd.read_csv(raw_path)

    # Target varsa ayır 
    feature_cols = [c for c in df_raw.columns if c != "SeriousDlqin2yrs"]
    X_raw = df_raw[feature_cols].copy()

    print(f"[INFO] Ham shape: {X_raw.shape}")

    # 2. Modelden risk skorlarını al
    print("[INFO] Modelden risk skorları alınıyor...")
    _, y_proba = predict_from_raw(X_raw)
    df_raw["Default_Probability"] = y_proba

    # 3. Farklı portföy senaryoları oluştur
    def safe_sample(df, n):
        """Satır sayısı n'den küçükse hepsini al, değilse örnekle."""
        if len(df) <= n:
            return df.copy()
        return df.sample(n=n, random_state=RANDOM_STATE)

    # Low risk portföy (çoğunlukla < 0.3)
    low_risk_df = df_raw[df_raw["Default_Probability"] < 0.3]
    low_risk_portfolio = safe_sample(low_risk_df, 500)

    # Mixed portföy (genel dağılımdan random)
    mixed_portfolio = safe_sample(df_raw, 500)

    # Stressed portföy (çoğunlukla > 0.7)
    high_risk_df = df_raw[df_raw["Default_Probability"] > 0.7]
    stressed_portfolio = safe_sample(high_risk_df, 500)

    # 4. Target ve skor kolonunu çıkar, sadece özellikleri bırak
    def prepare_for_export(df):
        cols = [c for c in df.columns if c not in ["SeriousDlqin2yrs", "Default_Probability"]]
        return df[cols].copy()

    low_risk_export = prepare_for_export(low_risk_portfolio)
    mixed_export = prepare_for_export(mixed_portfolio)
    stressed_export = prepare_for_export(stressed_portfolio)

    # 5. Kaydet
    low_path = DATA_DIR / "test_portfolio_low_risk.csv"
    mixed_path = DATA_DIR / "test_portfolio_mixed.csv"
    stressed_path = DATA_DIR / "test_portfolio_stressed.csv"

    low_risk_export.to_csv(low_path, index=False)
    mixed_export.to_csv(mixed_path, index=False)
    stressed_export.to_csv(stressed_path, index=False)

    print("✓ Oluşturulan dosyalar:")
    print(f"  - {low_path.name} ({len(low_risk_export)} satır)")
    print(f"  - {mixed_path.name} ({len(mixed_export)} satır)")
    print(f"  - {stressed_path.name} ({len(stressed_export)} satır)")


if __name__ == "__main__":
    main()

