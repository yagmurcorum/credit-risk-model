# app/streamlit_app.py

import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Proje kökünü sys.path'e ekle ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference import predict_from_raw
from src.config import DEFAULT_THRESHOLD

# Risk segment etiketleri (tüm uygulamada ortak)
RISK_CATEGORIES = ["Düşük", "Orta", "Yüksek", "Çok Yüksek"]

# Özellik isimleri için kullanıcı dostu etiketler
FEATURE_LABELS = {
    "age": "Yaş",
    "MonthlyIncome": "Aylık Gelir",
    "DebtRatio": "Borç Oranı (DebtRatio)",
    "RevolvingUtilizationOfUnsecuredLines": "Kredi Kartı Kullanım Oranı"
}

# ==================== SAYFA KONFİGÜRASYONU ====================
st.set_page_config(
    page_title="Kurumsal Kredi Risk Platformu",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== KURUMSAL TEMA - CSS STİLİ ====================
st.markdown("""
    <style>
    /* Ana Tema Renkleri */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --bg-light: #f8fafc;
        --bg-card: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
    }
    
    /* Genel Sayfa Stili */
    .main {
        background-color: var(--bg-light);
    }
    
    /* Header */
    .corporate-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        color: white;
    }
    
    .corporate-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .corporate-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        color: #e0e7ff;
        font-weight: 300;
    }
    
    /* Metrik Kartları */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid var(--secondary-color);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-delta {
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    .metric-delta.positive { color: var(--success-color); }
    .metric-delta.negative { color: var(--danger-color); }
    
    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low { background-color: #d1fae5; color: #065f46; }
    .risk-medium { background-color: #fef3c7; color: #92400e; }
    .risk-high { background-color: #fee2e2; color: #991b1b; }
    .risk-critical { background-color: #fecaca; color: #7f1d1d; }
    
    /* İnfo Kartları */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .info-card h3 {
        color: var(--primary-color);
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Sidebar Stili */
    .css-1d391kg {
        background-color: white;
    }
    
    /* Butonlar */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* DataFrame */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Alert Boxes */
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background-color: #dbeafe;
        border-color: #3b82f6;
        color: #1e40af;
    }
    
    .alert-success {
        background-color: #d1fae5;
        border-color: #10b981;
        color: #065f46;
    }
    
    .alert-warning {
        background-color: #fef3c7;
        border-color: #f59e0b;
        color: #92400e;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== YARDIMCI FONKSİYONLAR ====================

def get_analysis_data():
    """
    Session state'ten analiz sonuçlarını döndür.
    Sonuç yoksa standart uyarıyı gösterir.
    """
    if 'result_df' not in st.session_state:
        st.markdown("""
            <div class="alert-box alert-warning">
                <strong>⚠️ Henüz analiz yapılmadı</strong><br>
                Lütfen önce <b>"📋 Başvuru İşleme"</b> sekmesinde CSV yükleyip işlem başlatın.
            </div>
        """, unsafe_allow_html=True)
        return None
    return (
        st.session_state['result_df'],
        st.session_state['y_proba'],
        st.session_state['y_pred'],
        st.session_state['df'],
    )

# ==================== HEADER ====================
st.markdown("""
    <div class="corporate-header">
        <h1>🏦 Kredi Risk Değerlendirme Platformu</h1>
        <p>Gerçek zamanlı, makine öğrenmesi tabanlı kredi risk değerlendirme ve karar destek sistemi</p>
    </div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Platform Ayarları")
    st.markdown("---")
    
    # Logo/Branding alanı
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                        width: 60px; height: 60px; border-radius: 12px; 
                        display: inline-flex; align-items: center; justify-content: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
                <span style='font-size: 2rem;'>🏦</span>
            </div>
            <p style='margin-top: 0.5rem; font-weight: 600; color: #1e3a8a;'>Kredi Platformu v2.0</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk Parametreleri
    st.markdown("#### 🎯 Risk Parametreleri")
    
    threshold = st.slider(
        "Karar Eşiği",
        min_value=0.1,
        max_value=0.9,
        value=float(DEFAULT_THRESHOLD),
        step=0.01,
        help="Bu eşikten yüksek olasılığa sahip başvurular yüksek riskli olarak işaretlenir"
    )
    
    st.markdown("---")
    
    # Risk Sınıfları
    st.markdown("#### 📊 Risk Sınıfları")
    risk_levels = {
        "Düşük Risk": ("🟢", f"< {threshold:.2f}"),
        "Yüksek Risk": ("🔴", f"≥ {threshold:.2f}")
    }
    for label, (icon, cond) in risk_levels.items():
        st.markdown(f"{icon} **{label}:** {cond}")
    
    st.markdown("---")
    
    # Model Bilgileri
    st.markdown("#### 🤖 Model Bilgileri")
    model_info = {
        "Algoritma": "XGBoost",
        "ROC-AUC": "0.8699",
        "Precision": "0.4225",
        "Recall": "0.4788",
        "F1-Score": "0.4489",
        "Optimal Eşik": "0.81"
    }
    for key, value in model_info.items():
        st.metric(label=key, value=value)
    st.caption("Not: Bu metrikler eğitim/validasyon aşamasında elde edilen model performansını gösterir.")
    
    st.markdown("---")
    
    # Sistem Durumu
    st.markdown("#### 🔌 Sistem Durumu")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🟢 **Model:** Aktif")
    with col2:
        st.markdown("🟢 **API:** Çevrimiçi")
    
    st.caption(f"Son Güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

# ==================== ANA İÇERİK ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Başvuru İşleme",
    "📊 Risk Analiz Panosu", 
    "📈 Portföy Görünümü",
    "ℹ️ Sistem Bilgileri"
])

# ==================== TAB 1: BAŞVURU İŞLEME ====================
with tab1:
    st.markdown("## 📋 Kredi Başvurusu İşleme")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>📤 Başvuru Verisi Yükleme</h3>
                <p style='color: #64748b; margin-bottom: 1rem;'>
                    Modelle uyumlu şemaya sahip kredi başvuru verilerinizi içeren CSV dosyasını yükleyin. 
                    Sistem, her bir başvuruyu otomatik olarak işleyip risk skoru üretecektir.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card" style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);'>
                <h3 style='color: #1e40af;'>✅ Desteklenen Format</h3>
                <ul style='color: #1e40af; margin: 0;'>
                    <li>Dosya tipi: CSV</li>
                    <li>Format: Model eğitiminde kullanılan örnek veri seti ile uyumlu CSV</li>
                    <li>Boyut: Maks. 200MB</li>
                    <li>Kolon sayısı: 10–11 özellik</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    uploaded = st.file_uploader(
        "CSV Dosyası Seçin",
        type=["csv"],
        help="Kredi başvuru verilerini CSV formatında yükleyin"
    )
    
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            
            # Veri önizleme
            st.markdown("### 📊 Başvuru Verisi Önizleme")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Toplam Başvuru</div>
                        <div class="metric-value">{len(df):,}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Özellik Sayısı</div>
                        <div class="metric-value">{len(df.columns)}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Dosya Boyutu</div>
                        <div class="metric-value">{uploaded.size / 1024:.2f} KB</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Yükleme Saati</div>
                        <div class="metric-value">{datetime.now().strftime('%H:%M:%S')}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Tablo
            with st.expander("📋 Ham Veriyi Görüntüle", expanded=False):
                st.dataframe(df.head(20), use_container_width=True, height=400)
            
            with st.expander("📑 Kolon Bilgileri", expanded=False):
                col_info = pd.DataFrame({
                    "Kolon Adı": df.columns,
                    "Veri Tipi": df.dtypes.values,
                    "Null Olmayan": df.count().values,
                    "Null Sayısı": df.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)
            
            st.markdown("---")
            
            # İşleme butonu
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                process_btn = st.button(
                    "🚀 Başvuruları İşle ve Risk Skoru Üret",
                    type="primary",
                    use_container_width=True
                )
            
            if process_btn:
                with st.spinner("🔄 Başvurular işleniyor... Lütfen bekleyin..."):
                    try:
                        # Tahmin
                        y_pred_default, y_proba = predict_from_raw(df)
                        y_pred = (y_proba >= threshold).astype(int)
                        
                        # Sonuç DataFrame'i
                        result_df = df.copy()
                        result_df["Default_Probability"] = y_proba
                        result_df["Predicted_Label"] = y_pred
                        result_df["Risk_Kategorisi"] = pd.cut(
                            y_proba,
                            bins=[0.0, 0.3, 0.6, 0.9, 1.0],
                            labels=RISK_CATEGORIES
                        )
                        result_df["Karar"] = result_df["Predicted_Label"].map({
                            0: "ONAY",
                            1: "DETAYLI İNCELEME"
                        })
                        result_df["Islem_Tarihi"] = datetime.now().strftime('%d.%m.%Y')
                        result_df["Islem_Saati"] = datetime.now().strftime('%H:%M:%S')
                        
                        # Session state
                        st.session_state['result_df'] = result_df
                        st.session_state['y_proba'] = y_proba
                        st.session_state['y_pred'] = y_pred
                        st.session_state['df'] = df
                        st.session_state['threshold'] = threshold
                        
                        st.success("✅ İşleme başarıyla tamamlandı!")
                        
                        
                        st.markdown("---")
                        
                        # Özet metrikler
                        st.markdown("### 📈 İşleme Özeti")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        approved = (y_pred == 0).sum()
                        review = (y_pred == 1).sum()
                        approved_pct = approved / len(y_pred) * 100
                        review_pct = review / len(y_pred) * 100
                        avg_risk = y_proba.mean()
                        max_risk = y_proba.max()
                        
                        with col1:
                            st.markdown(f"""
                                <div class="metric-card" style="border-left-color: #10b981;">
                                    <div class="metric-label">✅ Onaylanan</div>
                                    <div class="metric-value">{approved:,}</div>
                                    <div class="metric-delta positive">{approved_pct:.1f}%</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                                <div class="metric-card" style="border-left-color: #ef4444;">
                                    <div class="metric-label">⚠️ İnceleme Gerektiren</div>
                                    <div class="metric-value">{review:,}</div>
                                    <div class="metric-delta negative">{review_pct:.1f}%</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                                <div class="metric-card" style="border-left-color: #f59e0b;">
                                    <div class="metric-label">📊 Ortalama Risk</div>
                                    <div class="metric-value">{avg_risk:.3f}</div>
                                    <div class="metric-delta">Eşik: {threshold:.2f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with col4:
                            st.markdown(f"""
                                <div class="metric-card" style="border-left-color: #ef4444;">
                                    <div class="metric-label">🔴 Maksimum Risk</div>
                                    <div class="metric-value">{max_risk:.3f}</div>
                                    <div class="metric-delta">Portföydeki en yüksek skor</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Risk dağılımı
                        st.markdown("### 📊 Risk Dağılımı Analizi")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=y_proba,
                            nbinsx=50,
                            name="Risk Dağılımı",
                            marker_color='#3b82f6',
                            opacity=0.7
                        ))
                        fig.add_vline(
                            x=threshold,
                            line_dash="dash",
                            line_color="red",
                            line_width=2,
                            annotation_text=f"Karar Eşiği: {threshold:.2f}",
                            annotation_position="top"
                        )
                        fig.update_layout(
                            title="Risk Skoru Dağılımı",
                            xaxis_title="Default Olasılığı",
                            yaxis_title="Başvuru Sayısı",
                            height=400,
                            showlegend=False,
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # İndirme seçenekleri
                        st.markdown("### 💾 Sonuçları Dışa Aktar")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Tüm Sonuçları İndir (CSV)",
                                data=csv_bytes,
                                file_name=f"kredi_risk_sonuclari_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        with col2:
                            high_risk_df = result_df[result_df["Predicted_Label"] == 1]
                            if len(high_risk_df) > 0:
                                high_risk_csv = high_risk_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="⚠️ Yüksek Riskli Başvuruları İndir (CSV)",
                                    data=high_risk_csv,
                                    file_name=f"yuksek_riskli_basvurular_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                    except Exception as e:
                        st.error(f"❌ İşleme sırasında hata oluştu: {str(e)}")
                        with st.expander("🔍 Hata Detayları"):
                            st.code(str(e))
                            st.info("""
                            Sorun Giderme İpuçları:
                            - CSV dosyasının modelle uyumlu ham formatta olduğundan emin olun
                            - Gerekli tüm kolonların mevcut olduğunu kontrol edin
                            - Veri tiplerini gözden geçirin
                            - Dosya boyutunun 200MB altında olduğundan emin olun
                            """)
        
        except Exception as e:
            st.error(f"❌ CSV dosyası okunurken hata oluştu: {str(e)}")
            st.info("Lütfen geçerli bir CSV dosyası yüklediğinizden emin olun.")
    
    else:
        st.markdown("""
            <div class="alert-box alert-info">
                <strong>ℹ️ Başlamak için</strong><br>
                Kredi başvurularını içeren CSV dosyasını yükleyerek toplu işleme sürecini başlatabilirsiniz.
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📖 Gerekli CSV Formatı ve Kolon Detayları"):
            st.markdown("""
            #### Zorunlu Kolonlar
            
            | Kolon Adı | Açıklama | Veri Tipi |
            |-----------|----------|-----------|
            | `RevolvingUtilizationOfUnsecuredLines` | Kredi kartı kullanım oranı | Float |
            | `age` | Başvuru sahibinin yaşı | Integer |
            | `MonthlyIncome` | Aylık gelir | Float |
            | `NumberOfTime30-59DaysPastDueNotWorse` | 30–59 gün gecikme sayısı | Integer |
            | `DebtRatio` | Borç / gelir oranı | Float |
            | `NumberOfOpenCreditLinesAndLoans` | Açık kredi / kart sayısı | Integer |
            | `NumberOfTimes90DaysLate` | 90+ gün ciddi gecikme sayısı | Integer |
            | `NumberRealEstateLoansOrLines` | Gayrimenkul kredisi sayısı | Integer |
            | `NumberOfTime60-89DaysPastDueNotWorse` | 60–89 gün gecikme sayısı | Integer |
            | `NumberOfDependents` | Bakmakla yükümlü olunan kişi sayısı | Integer |
            
            **Örnek Test Dosyası:** `data/test_sample_raw.csv`
            """)

# ==================== TAB 2: RISK ANALİZ PANOSU ====================
with tab2:
    st.markdown("## 📊 Risk Analiz Panosu")
    
    session = get_analysis_data()
    if session is not None:
        result_df, y_proba, y_pred, df = session
        
        # KPI'lar
        st.markdown("### 📈 Temel Risk Göstergeleri")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Portföy Büyüklüğü", f"{len(result_df):,}")
        with col2:
            st.metric("Ort. Risk", f"{y_proba.mean():.3f}")
        with col3:
            st.metric("Medyan Risk", f"{np.median(y_proba):.3f}")
        with col4:
            st.metric("Std. Sapma", f"{y_proba.std():.3f}")
        with col5:
            st.metric("Risk Aralığı", f"{y_proba.min():.2f} - {y_proba.max():.2f}")
        
        st.markdown("---")
        
        # Grafikler
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                result_df,
                x="Default_Probability",
                nbins=50,
                title="📊 Risk Skoru Dağılımı",
                labels={"Default_Probability": "Default Olasılığı", "count": "Başvuru Sayısı"},
                color_discrete_sequence=['#3b82f6']
            )
            fig_hist.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Eşik: {threshold:.2f}"
            )
            fig_hist.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            risk_counts = result_df["Risk_Kategorisi"].value_counts().sort_index()
            colors = ['#10b981', '#f59e0b', '#ef4444', '#7f1d1d']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker_colors=colors
            )])
            fig_pie.update_layout(
                title="🎯 Risk Segmentasyonu",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.caption(
            "Not: Risk kategorileri, modelden dönen default olasılığına göre 4 aralığa ayrılmıştır: "
            "0.00–0.30 Düşük, 0.30–0.60 Orta, 0.60–0.90 Yüksek, 0.90–1.00 Çok Yüksek."
        )
        
        st.markdown("---")
        
        # Çok boyutlu analiz
        if "age" in df.columns and "MonthlyIncome" in df.columns:
            st.markdown("### 🔍 Çok Boyutlu Risk Analizi")
            
            fig_scatter = px.scatter(
                result_df,
                x="age",
                y="MonthlyIncome",
                color="Risk_Kategorisi",
                size="Default_Probability",
                hover_data=["Default_Probability", "Karar"],
                title="Yaş – Gelir Dağılımı (Riske Göre Ağırlıklandırılmış)",
                color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444', '#7f1d1d'],
                category_orders={"Risk_Kategorisi": RISK_CATEGORIES}
            )
            fig_scatter.update_layout(height=500, template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("---")
        
        # En riskli başvurular
        st.markdown("### 🚨 Öncelikli İnceleme Gerektiren Başvurular (Top 10)")
        
        high_risk_df = result_df.nlargest(10, "Default_Probability")
        display_cols = ["Default_Probability", "Risk_Kategorisi", "Karar"]
        if "age" in df.columns:
            display_cols.append("age")
        if "MonthlyIncome" in df.columns:
            display_cols.append("MonthlyIncome")
        if "DebtRatio" in df.columns:
            display_cols.append("DebtRatio")
        
        st.dataframe(
            high_risk_df[display_cols].style.background_gradient(
                subset=["Default_Probability"],
                cmap="Reds"
            ),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # Özellik analizi
        if "age" in df.columns:
            st.markdown("### 📊 Risk Segmentlerine Göre Özellik Analizi")
            
            numeric_cols = ["age", "MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if available_cols:
                col1, col2 = st.columns([1, 3])
                with col1:
                    display_options = [FEATURE_LABELS.get(col, col) for col in available_cols]
                    selected_display = st.selectbox(
                        "Özellik Seçin:",
                        display_options,
                        key="feature_selector"
                    )
                    inverse_map = {FEATURE_LABELS.get(col, col): col for col in available_cols}
                    selected_feature = inverse_map[selected_display]
                with col2:
                    title_label = FEATURE_LABELS.get(selected_feature, selected_feature)
                    fig_box = px.box(
                        result_df,
                        x="Risk_Kategorisi",
                        y=selected_feature,
                        title=f"{title_label} Dağılımı (Risk Kategorilerine Göre)",
                        color="Risk_Kategorisi",
                        color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444', '#7f1d1d'],
                        category_orders={"Risk_Kategorisi": RISK_CATEGORIES}
                    )
                    fig_box.update_layout(height=400, template="plotly_white", showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)

# ==================== TAB 3: PORTFÖY GÖRÜNÜMÜ ====================
with tab3:
    st.markdown("## 📈 Portföy Görünümü ve Risk Yönetimi")
    
    session = get_analysis_data()
    if session is not None:
        result_df, y_proba, y_pred, df = session
        
        # Özet
        st.markdown("### 💼 Portföy Sağlık Özeti")
        
        col1, col2, col3, col4 = st.columns(4)
        total_apps = len(result_df)
        approved = (result_df["Predicted_Label"] == 0).sum()
        review_req = (result_df["Predicted_Label"] == 1).sum()
        avg_risk = y_proba.mean()
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Toplam Başvuru</div>
                    <div class="metric-value">{total_apps:,}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            approval_rate = approved / total_apps * 100
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #10b981;">
                    <div class="metric-label">Onay Oranı</div>
                    <div class="metric-value">{approval_rate:.1f}%</div>
                    <div class="metric-delta positive">{approved:,} onay</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            review_rate = review_req / total_apps * 100
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #ef4444;">
                    <div class="metric-label">İnceleme Oranı</div>
                    <div class="metric-value">{review_rate:.1f}%</div>
                    <div class="metric-delta negative">{review_req:,} işaretli</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            risk_status = "Sağlıklı" if avg_risk < 0.5 else "Yükselmiş"
            color = "#10b981" if avg_risk < 0.5 else "#ef4444"
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: {color};">
                    <div class="metric-label">Portföy Riski</div>
                    <div class="metric-value">{avg_risk:.3f}</div>
                    <div class="metric-delta">{risk_status}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Risk segmentasyon tablosu
        st.markdown("### 📊 Detaylı Risk Segmentasyonu")
        
        seg_stats = result_df.groupby("Risk_Kategorisi").agg({
            "Default_Probability": ["count", "mean", "min", "max"],
            "Predicted_Label": lambda x: (x == 1).sum()
        }).round(3)
        seg_stats.columns = ["Adet", "Ort. Risk", "Min Risk", "Maks Risk", "Yüksek Risk Adedi"]
        seg_stats["Yüzde"] = (seg_stats["Adet"] / total_apps * 100).round(1)
        
        st.dataframe(seg_stats, use_container_width=True)
        
        st.markdown("---")
        
        # İnteraktif filtreleme
        st.markdown("### 🔍 İnteraktif Başvuru Gezgini")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_prob = st.slider("Minimum Risk Skoru", 0.0, 1.0, 0.0, 0.01)
        with col2:
            max_prob = st.slider("Maksimum Risk Skoru", 0.0, 1.0, 1.0, 0.01)
        with col3:
            selected_categories = st.multiselect(
                "Risk Kategorileri",
                options=RISK_CATEGORIES,
                default=RISK_CATEGORIES
            )
        
        filtered_df = result_df[
            (result_df["Default_Probability"] >= min_prob) &
            (result_df["Default_Probability"] <= max_prob) &
            (result_df["Risk_Kategorisi"].isin(selected_categories))
        ]
        
        st.info(f"📋 Gösterilen kayıt: **{len(filtered_df):,} / {total_apps:,}**")
        
        st.dataframe(
            filtered_df.sort_values("Default_Probability", ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Filtrelenmiş sonuçları indirme
        csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Filtrelenmiş Sonuçları İndir (CSV)",
            data=csv_filtered,
            file_name=f"filtrelenmis_portfoy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==================== TAB 4: SİSTEM BİLGİLERİ ====================
with tab4:
    st.markdown("## ℹ️ Sistem Bilgileri ve Dokümantasyon")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>🏦 Kurumsal Kredi Risk Platformu</h3>
                <p style='color: #64748b; line-height: 1.8;'>
                    Bu platform, gelişmiş makine öğrenmesi algoritmalarını kullanarak kredi başvurularının
                    anlık risk değerlendirmesini ve operasyon ekipleri için karar desteğini sağlar.
                    XGBoost tabanlı model ve kapsamlı feature engineering ile,
                    kredi risk süreçlerinde tutarlı ve açıklanabilir skorlar üretir.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Platform Yetkinlikleri")
        
        capabilities = {
            "Otomatik İşleme": "Kredi başvurularını toplu olarak işleyip anlık risk skoru üretme",
            "Risk Segmentasyonu": "Düşük, Orta, Yüksek, Çok Yüksek şeklinde çok katmanlı sınıflandırma",
            "Gelişmiş Analitik": "Etkileşimli dashboard ve drill-down analizler",
            "Dışa Aktarım": "CSV formatında çıktı alarak diğer sistemlerle entegrasyon",
            "Eşik Yönetimi": "Risk iştahına göre ayarlanabilir karar eşiği",
            "İzlenebilirlik": "Tüm işlemler için zaman damgalı log ve kayıtlar"
        }
        for k, v in capabilities.items():
            st.markdown(f"**{k}:** {v}")
        
        st.markdown("---")
        
        st.markdown("### 🤖 Model Mimarisi ve Performans")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Teknik Özellikler:**
            - Algoritma: XGBoost (Gradient Boosting)
            - Özellik sayısı: 26 (22 sayısal + 4 kategorik, hedef hariç)
            - Eğitim verisi: 150.000 başvuru
            - Validasyon: 80/20 train–validation bölünmesi
            - Sınıf dengesizliği: scale_pos_weight = 13.96
            """)
        with col_b:
            st.markdown("""
            **Performans Metrikleri:**
            - ROC-AUC: 0.8699
            - Precision: 0.4225
            - Recall: 0.4788
            - F1-Score: 0.4489
            - Optimal Threshold: 0.81
            """)
    
    with col2:
        st.markdown("""
            <div class="info-card" style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white;'>
                <h3 style='color: white;'>📞 Destek ve İletişim</h3>
                <p style='color: #e0e7ff;'>
                    <strong>İletişim:</strong><br>
                    <a href="mailto:corumyagmuur@gmail.com" style='color:#bfdbfe;'>corumyagmuur@gmail.com</a><br><br>
                    <strong>LinkedIn:</strong><br>
                    <a href="https://www.linkedin.com/in/yagmurcorum" target="_blank" style='color:#bfdbfe;'>
                        linkedin.com/in/yagmurcorum
                    </a><br><br>
                    <strong>Medium:</strong><br>
                    <a href="https://medium.com/@corumyagmur" target="_blank" style='color:#bfdbfe;'>
                        medium.com/@corumyagmur
                    </a><br><br>
                    <strong>Platform Sürümü:</strong><br>
                    v2.0.0 (Build 2025.01)
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-card">
                <h3>📚 Dokümantasyon</h3>
                <p style='color: #64748b;'>
                    • Kullanıcı Kılavuzu<br>
                    • API Dokümantasyonu<br>
                    • Entegrasyon Rehberi<br>
                    • En İyi Uygulamalar<br>
                    • SSS & Sorun Giderme
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
        <div style='text-align: center; padding: 2rem; color: #64748b;'>
            <p>© 2025 Kurumsal Kredi Risk Platformu | Sürüm 2.0.0</p>
            <p style='font-size: 0.875rem;'>Gelişmiş Makine Öğrenmesi Teknolojisi ile Güçlendirilmiştir</p>
        </div>
    """, unsafe_allow_html=True)

