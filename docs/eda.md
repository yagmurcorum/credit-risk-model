# EDA Summary

Bu doküman `01_eda.ipynb` notebook'unun kısa özeti ve önemli bulguları içerir.

## Veri Seti Bilgileri

**- Kaynak:** data/cs-training.csv
**- Satır sayısı:** ~150.000
**- Sütun sayısı:** 12 (ham: 1 ID + 1 hedef + 10 feature)
**- Hedef değişken:** SeriousDlqin2yrs
**- 0** → Önümüzdeki 2 yıl içinde ciddi finansal gecikme yok
**- 1** → Önümüzdeki 2 yıl içinde en az bir kez ciddi finansal gecikme var


## Veri Yapısı

**Sütunlar:**
- `Unnamed: 0` - Index kolonu (modellemede kaldırılacak)
- `SeriousDlqin2yrs` - Hedef değişken
- `RevolvingUtilizationOfUnsecuredLines` - Kredi kartı limit kullanım oranı
- `age` - Müşteri yaşı
- `NumberOfTime30-59DaysPastDueNotWorse` - 30-59 gün gecikme sayısı
- `DebtRatio` - Borç/gelir oranı
- `MonthlyIncome` - Aylık gelir
- `NumberOfOpenCreditLinesAndLoans` - Açık kredi/hesap sayısı
- `NumberOfTimes90DaysLate` - 90+ gün gecikme sayısı
- `NumberRealEstateLoansOrLines` - Gayrimenkul kredisi sayısı
- `NumberOfTime60-89DaysPastDueNotWorse` - 60-89 gün gecikme sayısı
- `NumberOfDependents` - Bakmakla yükümlü kişi sayısı

## Eksik Değerler

**İki değişkende eksik değer bulunuyor:**
- `MonthlyIncome`: ~%20 eksik (29,731 satır)
- `NumberOfDependents`: ~%2.6 eksik (3,924 satır)

**Strateji:**
- `MonthlyIncome` eksikliği hem teknik hem davranışsal sinyal barındırıyor.
- **Data Cleaning** adımında MonthlyIncome ve NumberOfDependents için median imputation uygulanır 
  (eksikler ilgili veri seti üzerindeki median değerle doldurulur).
- İlk versiyonda ek bir missing flag üretilmemiştir; 
  ilerleyen iterasyonlarda `MonthlyIncomeMissingFlag` gibi ek bir feature
  oluşturmak, bu davranışsal sinyali modele taşımak için planlanan 
  geliştirme olarak not edilmiştir.

## Hedef Değişken Dağılımı

**Class Imbalance:**
- **0 (Default yok):** ~%93 (139,500 gözlem)
- **1 (Default var):** ~%7 (10,500 gözlem)

**Sonuç:**
- Veri seti belirgin şekilde dengesiz
- Sadece accuracy’ye bakmak yeterli değil
- ROC-AUC, Recall, Precision, F1 gibi metrikler daha anlamlı
- Class weight veya threshold ayarı kullanılmalı

## Dağılımlar ve Uç Değerler
### Sağa Çarpık Değişkenler

**RevolvingUtilizationOfUnsecuredLines:**  
- Değerler ağırlıklı olarak 0–2 bandında toplanmış; birkaç gözlem ise 50,000 seviyelerine kadar gidiyor.  
- Bu yapı sağa doğru uzun bir kuyruk oluşturuyor. FE aşamasında log dönüşümü ve üstten kırpma (winsorization) planlanıyor.

**DebtRatio:**  
- Müşterilerin büyük çoğunluğu 0–2 civarında; 300,000+ seviyelerinde az sayıda değer var ve bunlar muhtemelen veri hatası veya aşırı uç kayıtlar.  
- Bu değişken için de log dönüşümü ve yüksek uçlarda winsorization uygulanması düşünülüyor.

**MonthlyIncome:**  
- Yaklaşık %20 eksik değer var. Gelirlerin ana kitlesi 2k–10k bandında; bunun dışında 1M+ seviyelerinde birkaç uç değer bulunuyor.  
- Eksik değerler Data Cleaning aşamasında median ile doldurulur; FE tarafında ise `MonthlyIncome_log1p` feature’ı ile yüksek gelirlerin etkisi yumuşatılır.

### Delinquency Değişkenleri

**Gözlemler:**
- Gözlemlerin büyük çoğunluğu **0** (hiç gecikme yok)
- Maksimum değerlerin **98** olması pratik olarak mümkün değil
- 269 adet uç değer (90+ gün) tespit edildi
- Default yaşayan müşterilerde gecikme değerleri daha yüksek

**Strateji:**
- Upper capping uygulanacak (98 → 10)
- Toplam gecikme skoru türetilecek
- Gecikme flag'leri oluşturulacak

### Age Değişkeni

- 1 adet **0 yaş** değeri (hatalı kayıt)
- Default olan grupta yaş medyanı daha düşük
- Genç müşterilerde risk daha yüksek

**Strateji:**
- 0 değeri median ile doldurulacak
- Yaş segmentasyonu (binning) yapılacak

## Korelasyon Analizi

**Genel Yapı:**
- Korelasyonlar genel olarak düşük (kredi davranışları çok boyutlu)
- Gecikme değişkenleri birbirleriyle yüksek korelasyonlu (0.98–0.99)
- Hedef değişkenle korelasyonlar zayıf ama yönü tutarlı

**Öne Çıkan İlişkiler:**
- `NumberOfOpenCreditLinesAndLoans` ↔ `NumberRealEstateLoansOrLines`: 0.43
- Üç delinquency değişkeni arasında: 0.98–0.99

**Sonuç:**
- Belirgin bir multicollinearity problemi yok
- Delinquency kolonları model için güçlü sinyal taşıyor

## Default vs Feature Analizi

**Default yaşayan müşterilerde:**
- **Yaş daha düşük** (genç müşteriler daha riskli)
- **Gelir daha düşük** (ödeme kapasitesi azalıyor)
- **Kredi hattı sayısı daha az** (sınırlı kredi geçmişi)
- **Delinquency değerleri daha yüksek** (geçmiş ödeme sorunları)
- **Borç/gelir oranı daha oynak** (finansal stres)

**Sonuç:**  
Bu tabloya göre modelin en güçlü sinyali delinquency, gelir ve revolving utilization değişkenlerinden geliyor gibi duruyor.  
Feature engineering aşamasında özellikle bu değişkenler etrafında çalışılacak.


## Önemli Bulgular

1. **Veri kalitesi:** Age'de 0 değeri, delinquency'de 98 gibi uç değerler → Temizlik gerekli
2. **Class imbalance:** %7 default oranı → Class weight veya threshold tuning gerekli
3. **Sağa çarpık dağılımlar:** Bazı sayısal değişkenler için log dönüşümü ve winsorization kullanmak gerekecek
4. **Delinquency önemi:** Gecikme geçmişi en güçlü risk göstergesi
5. **Eksik değer stratejisi:** Median imputation + missing flag yaklaşımı

## Sonraki Adımlar

EDA bulgularına göre:
- **Data Cleaning:** Eksik değer doldurma, outlier yönetimi, anlamsız kolonların çıkarılması
- **Feature Engineering:** Delinquency özet değişkenleri, risk flag'leri, binning, log dönüşümleri
