import numpy as np
import pandas as pd

from typing import List
from src.config import DATA_DIR  


TARGET_COL = "SeriousDlqin2yrs"

# Delinquency ile ilgili ham kolonlar
DELINQ_COLS: List[str] = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Feature selection sonrası drop edilecek kolonlar
FINAL_DROP_COLS: List[str] = [
    # Ham delinquency kolonları
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "TotalDelinquency",

    # Aşırı korelasyonlu / türev kolonlar
    "DebtRatio",
    "Income_x_Age",
    "MonthlyIncome_log1p",

    # Çok yüksek VIF + zayıf anlam
    "CreditLineDensity",
]

# 1) TEMEL TEMİZLİK (Data Cleaning notebook ile uyumlu)
def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    02_data_cleaning.ipynb ile aynı mantığı kod tarafına taşır.
    - Gereksiz ID kolonunu drop eder
    - age == 0 hatasını düzeltir (median ile)
    - MonthlyIncome ve NumberOfDependents eksiklerini median ile doldurur
    - Delinquency kolonlarındaki 98 gibi uç değerleri 10 seviyesinde sınırlar
    """
    df = df.copy()

    # 1) Teknik ID kolonu varsa kaldır
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 2) age == 0 → NaN → median ile doldur
    if "age" in df.columns:
        df.loc[df["age"] == 0, "age"] = np.nan
        age_median = df["age"].median()
        df["age"] = df["age"].fillna(age_median)

    # 3) MonthlyIncome ve NumberOfDependents için median imputasyonu
    if "MonthlyIncome" in df.columns:
        income_median = df["MonthlyIncome"].median()
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(income_median)

    if "NumberOfDependents" in df.columns:
        dep_median = df["NumberOfDependents"].median()
        df["NumberOfDependents"] = df["NumberOfDependents"].fillna(dep_median)

    # 4) Delinquency kolonlarında uç değerleri cap et (98 → 10 vb.)
    for col in DELINQ_COLS:
        if col in df.columns:
            df[col] = df[col].clip(upper=10)

    return df


# 2) CORE NUMERIC FEATURES (log1p + ratio + basic flag)

def add_core_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blok 1: log1p dönüşümleri, basic ratio ve HighUtilizationFlag üretimi.
    """
    df = df.copy()

    # log1p dönüşümleri
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["RevolvingUtilizationOfUnsecuredLines_log1p"] = np.log1p(
            df["RevolvingUtilizationOfUnsecuredLines"]
        )

    if "DebtRatio" in df.columns:
        df["DebtRatio_log1p"] = np.log1p(df["DebtRatio"])

    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome_log1p"] = np.log1p(df["MonthlyIncome"])

    # Debt-to-income oranı (DebtRatio / MonthlyIncome)
    if "DebtRatio" in df.columns and "MonthlyIncome" in df.columns:
        income_safe = df["MonthlyIncome"].replace(0, np.nan)
        df["DebtToIncomeRatio"] = (df["DebtRatio"] / income_safe).fillna(0)

    # HighUtilizationFlag (limitin üstüne çıkmış veya çok yakın)
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["HighUtilizationFlag"] = (
            df["RevolvingUtilizationOfUnsecuredLines"] >= 1.0
        ).astype(int)

    return df


# 3) DELINQUENCY FEATURES (flags + severity)

def add_delinquency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blok 2 + Blok 5: delinquency toplamı, flag'ler ve ağırlıklı severity skoru.
    """
    df = df.copy()

    # Toplam delinquency sayısı
    if all(col in df.columns for col in DELINQ_COLS):
        df["TotalDelinquency"] = df[DELINQ_COLS].sum(axis=1)
    else:
        df["TotalDelinquency"] = 0

    # EverDelinquent: herhangi bir gecikme yaşadı mı?
    df["EverDelinquent"] = (df["TotalDelinquency"] > 0).astype(int)

    # Ever90DaysLate: 90+ gün gecikme var mı?
    if "NumberOfTimes90DaysLate" in df.columns:
        df["Ever90DaysLate"] = (df["NumberOfTimes90DaysLate"] > 0).astype(int)
    else:
        df["Ever90DaysLate"] = 0

    # MultipleDelinquencyFlag: toplam gecikme sayısı >= 2 ise 1
    df["MultipleDelinquencyFlag"] = (df["TotalDelinquency"] >= 2).astype(int)

    # DelinquencySeverityScore: ağırlıklı gecikme skoru
    # 30–59: ×1, 60–89: ×2, 90+: ×3
    df["DelinquencySeverityScore"] = (
        df.get("NumberOfTime30-59DaysPastDueNotWorse", 0) * 1
        + df.get("NumberOfTime60-89DaysPastDueNotWorse", 0) * 2
        + df.get("NumberOfTimes90DaysLate", 0) * 3
    )

    return df


# 4) RISK FLAGS (gelir / borç / delinquency davranış flag'leri)

def add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    HighDebtFlag gibi basit risk flag'lerini üretir.
    Buradaki eşikler, EDA/FE sırasında gördüğümüz dağılıma göre seçildi.
    """
    df = df.copy()

    # HighDebtFlag: DebtToIncomeRatio üst segment
    if "DebtToIncomeRatio" in df.columns:
        # Örn: en riskli ~%7-8'lik segment (üst quantile)
        thr = df["DebtToIncomeRatio"].quantile(0.93)
        df["HighDebtFlag"] = (df["DebtToIncomeRatio"] >= thr).astype(int)
    else:
        df["HighDebtFlag"] = 0

    # İstersen AgeRiskFlag vb. burada ekleyebilirsin

    return df

# 5) BINNING / SEGMENTASYON

def add_binning_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    AgeBin, IncomeBin, UtilizationBin, DelinqBin gibi segmentasyon feature'larını üretir.
    """
    df = df.copy()

    # AgeBin
    if "age" in df.columns:
        df["AgeBin"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, np.inf],
            labels=["18-30", "31-45", "46-60", "60+"],
            right=True,
        )

    # IncomeBin
    if "MonthlyIncome" in df.columns:
        df["IncomeBin"] = pd.cut(
            df["MonthlyIncome"],
            bins=[0, 3000, 6000, 10000, np.inf],
            labels=["0-3k", "3-6k", "6-10k", "10k+"],
            right=True,
        )

    # UtilizationBin
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["UtilizationBin"] = pd.cut(
            df["RevolvingUtilizationOfUnsecuredLines"],
            bins=[0, 0.3, 0.7, 1.0, np.inf],
            labels=["0-30%", "30-70%", "70-100%", "100%+"],
            right=True,
        )

    # DelinqBin (TotalDelinquency üzerinden)
    if "TotalDelinquency" in df.columns:
        df["DelinqBin"] = pd.cut(
            df["TotalDelinquency"],
            bins=[0, 1, 2, 4, np.inf],
            labels=["0", "1", "2-3", "4+"],
            right=False,
        )

    return df



# 6) INTERACTION FEATURES

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blok 4: temel etkileşim feature'larını üretir.
    """
    df = df.copy()

    # Borç yükü × kredi kullanım oranı
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns and "DebtRatio" in df.columns:
        df["Utilization_x_DebtRatio"] = (
            df["RevolvingUtilizationOfUnsecuredLines"] * df["DebtRatio"]
        )

    # Yaş × gelir
    if "MonthlyIncome" in df.columns and "age" in df.columns:
        df["Income_x_Age"] = df["MonthlyIncome"] * df["age"]

    # Toplam gecikme × kullanım
    if "TotalDelinquency" in df.columns and "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["Delinq_x_Utilization"] = (
            df["TotalDelinquency"] * df["RevolvingUtilizationOfUnsecuredLines"]
        )

    # Açık kredi hatları × gayrimenkul kredileri
    if "NumberOfOpenCreditLinesAndLoans" in df.columns and "NumberRealEstateLoansOrLines" in df.columns:
        df["OpenLines_x_RealEstate"] = (
            df["NumberOfOpenCreditLinesAndLoans"] * df["NumberRealEstateLoansOrLines"]
        )

    # HighUtilizationFlag × DebtRatio
    if "HighUtilizationFlag" in df.columns and "DebtRatio" in df.columns:
        df["HighUtil_x_DebtRatio"] = df["HighUtilizationFlag"] * df["DebtRatio"]

    return df


# 7) DOMAIN-DRIVEN FEATURES
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blok 5: EffectiveDebtLoad, RealEstateExposure, FinancialStressIndex vb.
    domain odaklı feature'ları üretir.
    """
    df = df.copy()

    # 1) Gerçek borç yükü
    if "DebtRatio" in df.columns and "MonthlyIncome" in df.columns:
        df["EffectiveDebtLoad"] = df["DebtRatio"] * df["MonthlyIncome"]

    # 2) Yaşa göre kredi hattı yoğunluğu
    if "NumberOfOpenCreditLinesAndLoans" in df.columns and "age" in df.columns:
        df["CreditLineDensity"] = (
            df["NumberOfOpenCreditLinesAndLoans"] / df["age"].replace(0, np.nan)
        ).fillna(0)

    # 3) Gayrimenkul borçlanma riski
    if "NumberRealEstateLoansOrLines" in df.columns and "DebtRatio" in df.columns:
        df["RealEstateExposure"] = (
            df["NumberRealEstateLoansOrLines"] * df["DebtRatio"]
        )

    # 4) Finansal stres indeksi
    if "DebtRatio" in df.columns and "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["FinancialStressIndex"] = np.log1p(
            df["DebtRatio"] * df["RevolvingUtilizationOfUnsecuredLines"]
        )

    return df



# 8) FEATURE SELECTION (FINAL DROP UYGULAMA)

def apply_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blok 6 + 7: FINAL_DROP_COLS listesini kullanarak gereksiz kolonları çıkarır.
    """
    df = df.copy()
    cols_to_keep = [c for c in df.columns if c not in FINAL_DROP_COLS]
    return df[cols_to_keep]



# 9) ANA FONKSİYON: TRAINING İÇİN VERİ HAZIRLAMA

def prepare_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham (veya kısmen işlenmiş) bir eğitim datasını alır,
    tüm temizlik + FE + feature selection adımlarını uygular.
    """
    df = df.copy()

    df = clean_basic(df)
    df = add_core_numeric_features(df)
    df = add_delinquency_features(df)
    df = add_risk_flags(df)
    df = add_binning_features(df)
    df = add_interaction_features(df)
    df = add_domain_features(df)
    df = apply_feature_selection(df)

    return df

if __name__ == "__main__":
    # Hızlı manuel test
    import pandas as pd
    from src.config import RAW_TRAIN, DATA_DIR

    print("Ham veri okunuyor:", RAW_TRAIN)
    df_raw = pd.read_csv(RAW_TRAIN)
    df_prep = prepare_training(df_raw)

    print("Ham shape:", df_raw.shape)
    print("Hazır shape:", df_prep.shape)

    out_path = DATA_DIR / "training_prepared_from_script.csv"
    df_prep.to_csv(out_path, index=False)
    print("Hazırlanmış veri kaydedildi:", out_path)
