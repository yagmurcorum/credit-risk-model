# src/feature_engineering.py

"""
Bu modül, kredi risk verisi için kullanılan temel feature engineering
adımlarını fonksiyonlar halinde toplar.

Notlar:
- Asıl deneyler ve FE denemeleri 03_feature_engineering.ipynb içinde yapılmıştır.
- Buradaki fonksiyonlar, o notebook'taki mantığı script formunda özetlemek için
  tasarlanmıştır.
- İsimlendirmeler, training_prepared.csv ve dokümantasyon ile tutarlı olacak
  şekilde düzenlenmiştir.
"""

from typing import Iterable
import numpy as np
import pandas as pd


# Gecikme ile ilgili orijinal kolonlar
DELINQ_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def _check_required_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    """İç fonksiyon: Eksik kolon varsa daha okunaklı hata mesajı üretir."""
    required = set(required)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{context}] için gerekli kolonlar eksik: {sorted(missing)}")


def add_delinquency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gecikme ile ilgili türetilen değişkenler:

    - TotalDelinquency:
        30-59, 60-89 ve 90+ gün gecikmelerin toplam adedi
    - EverDelinquent:
        Müşteri geçmişte en az 1 kez gecikmiş mi? (0/1)
    - Ever90DaysLate:
        90+ gün gecikmesi var mı? (0/1)
    - MultipleDelinquencyFlag:
        Toplam gecikme sayısı >= 2 ise 1 (yüksek risk flag'i)
    - DelinquencySeverityScore:
        Ağırlıklı gecikme skoru:
            1 * 30-59  +  2 * 60-89  +  3 * 90+ gün gecikmeler
    """
    _check_required_columns(df, DELINQ_COLS, "add_delinquency_features")
    df = df.copy()

    # Toplam gecikme adedi
    df["TotalDelinquency"] = df[DELINQ_COLS].sum(axis=1)

    # En az bir gecikme var mı?
    df["EverDelinquent"] = (df["TotalDelinquency"] > 0).astype(int)

    # 90+ gün gecikme flag'i
    df["Ever90DaysLate"] = (df["NumberOfTimes90DaysLate"] > 0).astype(int)

    # Çoklu gecikme flag'i
    df["MultipleDelinquencyFlag"] = (df["TotalDelinquency"] >= 2).astype(int)

    # Ağırlıklı gecikme skoru
    df["DelinquencySeverityScore"] = (
        1 * df["NumberOfTime30-59DaysPastDueNotWorse"]
        + 2 * df["NumberOfTime60-89DaysPastDueNotWorse"]
        + 3 * df["NumberOfTimes90DaysLate"]
    )

    return df


def add_utilization_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kredi kartı limit kullanımı ile gecikme / borç etkileşimleri:

    - Delinq_x_Utilization:
        TotalDelinquency x kart kullanım oranı
    - Utilization_x_DebtRatio:
        Kart kullanım oranı x borç/gelir oranı
    - HighUtilizationFlag:
        Kart kullanım oranı > 1.0 (limit aşımı) ise 1, değilse 0
    - HighUtil_x_DebtRatio:
        HighUtilizationFlag x borç/gelir oranı
    """
    required = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]
    _check_required_columns(df, required, "add_utilization_interactions")

    df = df.copy()

    util = df["RevolvingUtilizationOfUnsecuredLines"].fillna(0)
    debt_ratio = df["DebtRatio"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # TotalDelinquency yoksa burada üret
    if "TotalDelinquency" not in df.columns:
        _check_required_columns(df, DELINQ_COLS, "add_utilization_interactions/TotalDelinquency")
        df["TotalDelinquency"] = df[DELINQ_COLS].sum(axis=1)

    # Interaction feature'lar
    df["Delinq_x_Utilization"] = df["TotalDelinquency"] * util
    df["Utilization_x_DebtRatio"] = util * debt_ratio

    # Yüksek kullanım flag'i (limit aşımı)
    df["HighUtilizationFlag"] = (util > 1.0).astype(int)
    df["HighUtil_x_DebtRatio"] = df["HighUtilizationFlag"] * debt_ratio

    return df


def add_debt_exposure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain tabanlı borç / maruziyet / stres feature'ları:

    - EffectiveDebtLoad:
        DebtRatio x MonthlyIncome  --> gelire göre parasal borç yükü
    - RealEstateExposure:
        NumberRealEstateLoansOrLines x DebtRatio
    - FinancialStressIndex:
        log1p(DebtRatio x RevolvingUtilizationOfUnsecuredLines)
    - CreditLineDensity:
        NumberOfOpenCreditLinesAndLoans / age
    - HighDebtFlag:
        DebtToIncomeRatio > 0.4 ise 1, değilse 0
    """
    required = ["DebtRatio", "MonthlyIncome"]
    _check_required_columns(df, required, "add_debt_exposure_features")

    df = df.copy()

    debt_ratio = df["DebtRatio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    income = df["MonthlyIncome"].fillna(0)

    # Parasal borç yükü
    df["EffectiveDebtLoad"] = debt_ratio * income

    # Gayrimenkul maruziyeti
    if "NumberRealEstateLoansOrLines" in df.columns:
        df["RealEstateExposure"] = (
            df["NumberRealEstateLoansOrLines"].fillna(0) * debt_ratio
        )

    # Finansal stres indeksi
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        util = df["RevolvingUtilizationOfUnsecuredLines"].fillna(0)
        df["FinancialStressIndex"] = np.log1p(debt_ratio * util)

    # Kredi hattı yoğunluğu (yaşa göre)
    if "NumberOfOpenCreditLinesAndLoans" in df.columns and "age" in df.columns:
        age = df["age"].replace(0, np.nan)
        density = (
            df["NumberOfOpenCreditLinesAndLoans"].fillna(0) / age
        )
        density = density.replace([np.inf, -np.inf], np.nan).fillna(0)
        df["CreditLineDensity"] = density

    # Yüksek borç flag'i (borç/gelir oranına göre)
    if "DebtToIncomeRatio" in df.columns:
        dti = df["DebtToIncomeRatio"].replace([np.inf, -np.inf], np.nan).fillna(0)
        df["HighDebtFlag"] = (dti > 0.4).astype(int)

    return df


def add_income_bins(
    df: pd.DataFrame, bins: Iterable[float] | None = None
) -> pd.DataFrame:
    """
    Aylık gelir için segmentler (bin'ler) oluşturur.

    Varsayılan aralıklar:
        0-3k, 3-6k, 6-10k, 10k+  (bin label'ları string olarak atanır)

    Üretilen kolon:
        - IncomeBin
    """
    _check_required_columns(df, ["MonthlyIncome"], "add_income_bins")
    df = df.copy()

    if bins is None:
        bins = [0, 3000, 6000, 10000, np.inf]

    labels = ["0-3k", "3-6k", "6-10k", "10k+"]

    income = df["MonthlyIncome"].fillna(0)
    df["IncomeBin"] = pd.cut(
        income,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    return df


def apply_all_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm FE adımlarını sırasıyla uygulayan high-level fonksiyon.

    Not:
    - Prediction tarafında şu anda training_prepared.csv zaten
      FE sonrası olduğu için bu fonksiyon zorunlu değil; daha çok,
      notebook'ta yaptığımız dönüşümlerin script formundaki özetidir.
    """
    df_fe = df.copy()
    df_fe = add_delinquency_features(df_fe)
    df_fe = add_utilization_interactions(df_fe)
    df_fe = add_debt_exposure_features(df_fe)
    df_fe = add_income_bins(df_fe)

    return df_fe
