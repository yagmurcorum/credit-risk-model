from pathlib import Path

# === Proje kök klasörü ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# === Ana klasörler ===
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"
SRC_DIR = PROJECT_ROOT / "src"

# === Veri dosyaları ===
RAW_TRAIN = DATA_DIR / "cs-training.csv"
RAW_TEST = DATA_DIR / "cs-test.csv"
CLEAN_TRAIN = DATA_DIR / "cs-training-clean.csv"
TRAINING_PREPARED = DATA_DIR / "training_prepared.csv"

# === Final trained model ===
# Notebook'ta da bu isimle kaydediliyor:
# models/xgboost_credit_risk_final.pkl
FINAL_MODEL = MODELS_DIR / "xgboost_credit_risk_final.pkl"

# === Rastgelelik sabiti ===
SEED = 42

# === Business Kuralları (dokümantasyon amaçlı) ===

# Model threshold (validation set üzerinde optimal ~0.81 bulundu)
DEFAULT_THRESHOLD = 0.81

# Minimum performans beklentileri (recall / precision)
MIN_RECALL_REQUIREMENT = 0.45
MIN_PRECISION_REQUIREMENT = 0.40

# Onay oranı hedef aralığı (kredi başvurularının hangi oranı onaylanabilir?)
TARGET_APPROVAL_RATE_MIN = 0.85
TARGET_APPROVAL_RATE_MAX = 0.95

# Onaylanan müşteriler içinde kabul edilebilir maksimum bad rate
MAX_BAD_RATE_IN_APPROVED = 0.05

# === Model Ayarları (bilgi amaçlı; eğitim notebook'unda optimize edildi) ===

XGB_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 1.0,
    "min_child_weight": 3,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": SEED,
}

# Training pipeline'da kullanılan temel parametre seti
# (pipeline MODEL_PARAMS'i import ediyor)
MODEL_PARAMS = XGB_DEFAULT_PARAMS

# Sınıf dengesizliği için kullanılan yaklaşık scale_pos_weight
# (train set oranına göre ~111979 / 8021 ≈ 13.96)
SCALE_POS_WEIGHT = 13.96

# === Monitoring Eşikleri (canlı sistem tasarımı için rehber) ===

# PSI (Population Stability Index) alarm eşikleri
PSI_WARNING_THRESHOLD = 0.10
PSI_ALARM_THRESHOLD = 0.25

# Prediction latency hedefleri (milisaniye)
TARGET_AVG_LATENCY_MS = 100
TARGET_P95_LATENCY_MS = 200
TARGET_P99_LATENCY_MS = 500

