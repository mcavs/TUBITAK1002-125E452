# TÜBİTAK 1002-A (Proje No: 125E452)

## Regresyon ve Sınıflandırma Modellerinde Tahmin Tutarsızlığı ve Ayarlanabilirlik: Güvenilir Makine Öğrenmesinde Hiperparametre Seçiminin Rolü

**Yürütücü:** Doç. Dr. Mustafa ÇAVUŞ  
**Kurum:** Eskişehir Teknik Üniversitesi, Fen Fakültesi, İstatistik Bölümü  
**Program:** TÜBİTAK 1002-A Hızlı Destek Programı  

---

### Proje Özeti
Bu proje, makine öğrenmesi algoritmalarında (Lojistik/Ridge Regresyon `glmnet`, Karar Ağaçları `rpart`, En Yakın Komşu `kknn`, Destek Vektör Makineleri `svm`, Rassal Orman `ranger` ve Gradyan Artırma `xgboost`) hiperparametre seçiminin modellerin başarımına ve tahmin tutarsızlığına (prediction discrepancy / Rashomon etkisi) etkilerini kapsamlı kıyaslama deneyleri, vekil modelleme ve Açıklanabilir Yapay Zeka (XAI) araçlarıyla incelemektedir.

---

### Dizin Yapısı
- **`datasets/`**: Kıyaslama deneylerinde kullanılan standart veri seti koleksiyonları (OpenML-CC18 ikili sınıflandırma ve OpenML-CTR23 regresyon paketleri).
- **`src/`**: Veri indirme, ön işleme, 10 katlı çapraz doğrulama, simülasyon, vekil modelleme ve görselleştirme betikleri (R ve Python).
- **`figures/`**: Proje kapsamında üretilen yüksek çözünürlüklü performans, ayarlanabilirlik, tutarsızlık ve XAI (PFI ve PDP) figürleri.
- **`results/summary_metrics/`**: Çapraz doğrulama ve Rashomon tolerans ($\delta$) analizleri sonucunda türetilen özet performans ve tutarsızlık tabloları.
- **`results/surrogates/` & `results/xai/`**: Vekil model doğrulama metrikleri ve XAI özet çıktıları.

---

### Kullanılan İstatistiksel Ölçütler ve Matematiksel Tanımlar

Projeden elde edilen ve `results/summary_metrics/` dizininde sunulan sonuçlar, hem sınıflandırma hem de regresyon problemleri için aşağıdaki matematiksel formüller ve istatistiksel tanımlar temelinde hesaplanmıştır:

#### 1. Sınıflandırma İçin Tahmin Tutarsızlığı (Prediction Discrepancy)

Sınıflandırma problemlerinde varsayılan hiperparametre konfigürasyonuna sahip model $f_{\theta_d}$ ile arama uzayındaki herhangi bir optimize edilmiş konfigürasyon $f_{\theta_j}$ arasındaki tahmin uyuşmazlığı, test verisindeki $n$ gözlem üzerinden hesaplanan sıfır-bir kaybı (Zero-One Loss) ile tanımlanır (Cavus vd., 2026):

$$\text{Discrepancy}(f_{\theta_d}, f_{\theta_j}) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{I}\left(f_{\theta_d}(x_i) \neq f_{\theta_j}(x_i)\right)$$

Burada $\mathbb{I}(\cdot)$ gösterge fonksiyonudur. İncelenen model için en yüksek uyuşmazlığı veren model düzeyi tutarsızlık ise arama uzayındaki tüm aday konfigürasyonlar ($\Theta$) üzerinden şu şekilde elde edilir:

$$\text{Model Discrepancy} = \max_{\theta_j \in \Theta} \text{Discrepancy}(f_{\theta_d}, f_{\theta_j})$$

Tek bir hiperparametrenin ($h$) marjinal tutarsızlığı ise, diğer tüm hiperparametrelerin varsayılan değerlerinde sabit tutulduğu alt küme $\Theta_h$ üzerinde hesaplanır:

$$\text{HP Discrepancy}(h) = \max_{\theta_j \in \Theta_h} \text{Discrepancy}(f_{\theta_d}, f_{\theta_j})$$

---

#### 2. Regresyon İçin Tahmin Tutarsızlığı ve $\delta$-Discrepancy (Predictive Multiplicity)

Sürekli yanıt değişkenine sahip regresyon problemlerinde tahmin değerleri sürekli olduğundan, tutarsızlık hedef değişkenin gözlemlenen aralığına ($y_{\text{range}} = y_{\max} - y_{\min}$) normalize edilerek tanımlanır (Serce ve Cavus, 2026):

- **Normalize Edilmiş Tahmin Tutarsızlığı (Multiplicity):**
$$\text{Multiplicity}(f_{\theta_d}, f_{\theta_j}) = \frac{1}{n} \sum_{i=1}^{n} \frac{|f_{\theta_d}(x_i) - f_{\theta_j}(x_i)|}{y_{\max} - y_{\min}}$$

- **Model Düzeyinde Regresyon Tutarsızlığı:**
$$\text{Model Multiplicity} = \max_{\theta_j \in \Theta} \text{Multiplicity}(f_{\theta_d}, f_{\theta_j})$$

- **Tolerans Tabanlı Tutarsızlık ($\delta$-Discrepancy):**  
Bireysel gözlemlerin ne kadarının varsayılan modele kıyasla hedef değişken aralığının en az $\delta$ toleransı kadar saptığını ölçen oran:
$$\delta\text{-Discrepancy}(f_{\theta_d}, f_{\theta_j}) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{I}\left(\frac{|f_{\theta_d}(x_i) - f_{\theta_j}(x_i)|}{y_{\max} - y_{\min}} \ge \delta\right)$$

Çalışmada standart eşik değeri $\delta = 0.05$ (hedef aralığının %5'i kadar sapma) olarak belirlenmiştir.

---

#### 3. Model Ayarlanabilirliği (Tunability)

Probst vd. (2019) yaklaşımı temel alınarak ayarlanabilirlik, bir algoritmanın varsayılan konfigürasyonuna ($f_{\theta_d}$) kıyasla hiperparametre optimizasyonu sonucunda sağladığı nesnel başarım artışı olarak tanımlanmıştır:

- **Sınıflandırma Görevlerinde ($F_1$-Skoru Kazancı):**
$$\text{Tunability}_{\text{classif}} = \max\left(0, \,\max_{\theta_j \in \Theta} F_1(f_{\theta_j}) - F_1(f_{\theta_d})\right)$$

- **Regresyon Görevlerinde (Determinasyon Katsayısı $R^2$ Artışı):**
$$\text{Tunability}_{\text{regr}} = \max\left(0, \,\max_{\theta_j \in \Theta} R^2(f_{\theta_j}) - R^2(f_{\theta_d})\right)$$

---

### Kıyaslama Veri Setleri Koleksiyonları

Projede kullanılan kıyaslama verileri, makine öğrenmesi literatüründe yaygın olarak kabul gören iki temel küratörlü koleksiyondan derlenmiştir:

1. **İkili Sınıflandırma Koleksiyonu (OpenML-CC18):** Pratik makine öğrenmesi problemlerini temsil eden, dengeli ve dengesiz dağılımlara sahip 34 adet standart ikili sınıflandırma veri seti kullanılmıştır (Bischl vd., 2021).
2. **Tabular Regresyon Koleksiyonu (OpenML-CTR23):** Regresyon modellerinin kıyaslanması amacıyla oluşturulmuş, farklı boyut ve değişken yapılarına sahip 26 adet tabular regresyon veri seti kullanılmıştır (Fischer vd., 2023).

---

#### 1. İkili Sınıflandırma Veri Setleri (OpenML-CC18 — 34 Veri Seti)

|   Görev Kimliği (Task ID) | Veri Seti Adı                    |   Gözlem Sayısı |   Değişken Sayısı |
|--------------------------:|:---------------------------------|----------------:|------------------:|
|                         3 | kr-vs-kp                         |            3196 |                37 |
|                        15 | breast-w                         |             699 |                10 |
|                        31 | credit-g                         |            1000 |                21 |
|                        37 | diabetes                         |             768 |                 9 |
|                        43 | spambase                         |            4601 |                58 |
|                        49 | tic-tac-toe                      |             958 |                10 |
|                       219 | electricity                      |           45312 |                 9 |
|                      3021 | sick                             |            3772 |                30 |
|                      3902 | pc4                              |            1458 |                38 |
|                      3903 | pc3                              |            1563 |                38 |
|                      3904 | jm1                              |           10885 |                22 |
|                      3913 | kc2                              |             522 |                22 |
|                      3917 | kc1                              |            2109 |                22 |
|                      3918 | pc1                              |            1109 |                22 |
|                      7592 | adult                            |           48842 |                15 |
|                      9910 | Bioresponse                      |            3751 |              1777 |
|                      9946 | wdbc                             |             569 |                31 |
|                      9952 | phoneme                          |            5404 |                 6 |
|                      9957 | qsar-biodeg                      |            1055 |                42 |
|                      9971 | ilpd                             |             583 |                11 |
|                      9976 | madelon                          |            2600 |               501 |
|                      9977 | nomao                            |           34465 |               119 |
|                      9978 | ozone-level-8hr                  |            2534 |                73 |
|                     10093 | banknote-authentication          |            1372 |                 5 |
|                     10101 | blood-transfusion-service-center |             748 |                 5 |
|                     14952 | PhishingWebsites                 |           11055 |                31 |
|                     14954 | cylinder-bands                   |             540 |                38 |
|                     14965 | bank-marketing                   |           45211 |                17 |
|                    125920 | dresses-sales                    |             500 |                13 |
|                    146819 | climate-model-simulation-crashes |             540 |                19 |
|                    146820 | wilt                             |            4839 |                 6 |
|                    167120 | numerai28.6                      |           96320 |                22 |
|                    167125 | Internet-Advertisements          |            3279 |              1559 |
|                    167141 | churn                            |            5000 |                21 |

---

#### 2. Regresyon Veri Setleri (OpenML-CTR23 — 26 Veri Seti)

|   Görev Kimliği (Task ID) | Veri Seti Adı                       |   Gözlem Sayısı |   Değişken Sayısı |
|--------------------------:|:------------------------------------|----------------:|------------------:|
|                    167210 | Moneyball                           |            1232 |                15 |
|                    233211 | diamonds                            |           53940 |                10 |
|                    233212 | Allstate_Claims_Severity            |          188318 |               131 |
|                    233214 | Santander_transaction_value         |            4459 |              4992 |
|                    233215 | Mercedes_Benz_Greener_Manufacturing |            4209 |               377 |
|                    359930 | quake                               |            2178 |                 4 |
|                    359932 | socmob                              |            1156 |                 6 |
|                    359933 | space_ga                            |            3107 |                 7 |
|                    359934 | tecator                             |             240 |               125 |
|                    359935 | wine_quality                        |            6497 |                12 |
|                    359936 | elevators                           |           16599 |                19 |
|                    359937 | black_friday                        |          166821 |                10 |
|                    359938 | Brazilian_houses                    |           10692 |                13 |
|                    359939 | topo_2_1                            |            8885 |               267 |
|                    359940 | yprop_4_1                           |            8885 |               252 |
|                    359941 | OnlineNewsPopularity                |           39644 |                60 |
|                    359942 | colleges                            |            7063 |                45 |
|                    359944 | abalone                             |            4177 |                 9 |
|                    359945 | us_crime                            |            1994 |               127 |
|                    359946 | pol                                 |           15000 |                49 |
|                    359948 | SAT11-HAND-runtime-regression       |            4440 |               117 |
|                    359949 | house_sales                         |           21613 |                22 |
|                    359950 | boston                              |             506 |                14 |
|                    359951 | house_prices_nominal                |            1460 |                80 |
|                    359952 | house_16H                           |           22784 |                17 |
|                    360945 | MIP-2016-regression                 |            1090 |               145 |

---

### Kaynaklar

- Bischl, B., Casalicchio, G., Feurer, M., Gijsbers, P., Hutter, F., Lang, M., Mantovani, R. G., van Rijn, J. N., & Vanschoren, J. (2021). OpenML benchmarking suites. *arXiv preprint arXiv:1708.03731*.
- Cavus, M., Woźnica, K., & Biecek, P. (2026). The role of hyperparameters in predictive multiplicity. *arXiv preprint arXiv:2412.03541*.
- Fischer, S., Harutyunyan, L., Feurer, M., & Bischl, B. (2023). OpenML-CTR23 – A curated tabular regression benchmarking suite. *Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023) Datasets and Benchmarks Track*.
- Probst, P., Boulesteix, A. L., & Bischl, B. (2019). Tunability: Importance of hyperparameters of machine learning algorithms. *Journal of Machine Learning Research*, 20(53), 1-32.
- Serce, M. E., & Cavus, M. (2026). Predictive multiplicity in regression models: Concepts, metrics, and empirical evaluation. *Working Paper / Preprint*.

---
*Tüm hakları saklıdır. Kod ve analiz verileri TÜBİTAK Açık Bilim Politikası ve açık bilim ilkelerine uygun olarak paylaşılmaktadır.*
