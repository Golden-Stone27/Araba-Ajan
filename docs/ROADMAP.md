# 🗺️ RaceAgent: Uzun Vadeli Yol Haritası ve Gelecek Vizyonu

Bu doküman, RaceAgent projesinin temel eğitim aşamaları (M1–M6) sonrasındaki vizyonunu, bağımsız yapay zeka dağıtım stratejisini ve üst düzey sürüş tekniklerine (drift, GT yarışları, heel-and-toe) yönelik geliştirme adımlarını içerir.

---

## 🏛️ Temel Felsefe: Simülatörden Bağımsız Sürücü Beyni
RaceAgent'ın nihai hedefi bir oyun motoruna (Unity/Unreal) mahkûm olmak değil; **herhangi bir simülasyon veya robotik araca bağlanabilen, tak-çalıştır, bağımsız bir otonom yarış pilotu (Autonomous Driving Agent)** olmaktır.

* **Sim-to-Sim Transfer Stratejisi:** 
  1. Hızlı ve hafif simülasyonda (Unity) yüksek veri toplama hızıyla (~14.000 SPS) temel refleksleri eğit.
  2. Ağırlıkları dışa aktar.
  3. Gerçekçi fizik motorunda (Unreal Engine 5 Chaos Vehicles / Özel Lastik Modeli) ince ayar (fine-tuning) yap.

---

## Faz 1: Model Dağıtımı, SDK ve Hafif Demo Platformu
Yapay zekanın Unity ortamından tamamen koparılarak herkesin kullanabileceği ve test edebileceği taşınabilir bir yazılım kütüphanesine dönüştürülmesi.

* **Bağımsız Çıkarım Motoru (ONNX Entegrasyonu):**
  - Eğitilmiş PyTorch modellerinin (`.pt`) standart `.onnx` formatına dondurulması.
  - Ağır kütüphanelere (PyTorch, CUDA) ihtiyaç duymadan, yalnızca hafif `onnxruntime` (CPU) ile çalışabilen çıkarım katmanı.
* **Gömülü Normalizasyon Katmanı:**
  - Eğitim sırasında toplanan Welford RMS (ortalama ve varyans) parametrelerinin model mimarisine gömülmesi.
  - Dışarıdan bağlanan kullanıcının sadece ham telemetri verisi (km/h, raycast metreleri) vermesini sağlayan otomatik ölçekleyici.
* **Python SDK Paketi (`raceagent`):**
  - Tek satırda çalışan API mimarisi: `driver = PretrainedDriver.load("champion-v1")`.
  - Kullanıcının kendi aracından aldığı 26 float girdiye karşılık `[steer, throttle_brake]` aksiyonu üreten hafif arayüz.
* **Hafif Görselleştirici (Demo Viewer):**
  - Eğitilen modellerin agresif yarış çizgisini ve telemetrisini (hız, g-kuvveti, raycast ışınları) kod yazmadan tek tıkla gösteren bağımsız demo penceresi.

---

## Faz 2: Fiziksel Gerçekçilik ve Araç Dinamiği Yükseltmesi
Mevcut standart tekerlek collider fiziğinin ötesine geçerek gerçek otomobil dinamiklerini yakalamak.

* **Non-Lineer Lastik Dinamiği (Pacejka 'Magic Formula'):**
  - Lastiğin tutunma limitinde (slip angle $8^\circ - 35^\circ$ kayma açısı aralığı) lineer olmayan sürtünme eğrilerinin simülasyona eklenmesi.
  - Yanal ve boyuna tutunma kaybının modellenmesi.
* **Ağırlık Transferi ve Süspansiyon Geometrisi:**
  - Sert frenajda ön aksa binen yük (pitch), viraj içi yanal yatma (roll) ve arka lastiklerin boşa çıkması dinamiği.
  - Ağırlık transferine bağlı aşırı arkadan kayma (oversteer) ve önden kayma (understeer) simülasyonu.
* **Genişletilmiş Ayrık Aksiyon Uzayı:**
  - Tek birleşik eksenden gerçekçi 5 bağımsız pedal/kumanda eksenine geçiş:
    $$\text{Aksiyon} = [\text{Steer}, \text{Throttle}, \text{Brake}, \text{Clutch}, \text{Handbrake}]$$
  - Gaz ve frenin aynı anda basılabilmesi (left-foot braking / trail braking altyapısı).

---

## Faz 3: Sürüş Disiplinleri ve Özelleşmiş Yapay Zeka Karakterleri
Farklı sürüş hedefleri için ödül fonksiyonlarının özelleştirilmesi ve farklı pilot profillerinin eğitilmesi.

### 1. Teknik Drift Ajanı
* **Fizik:** Arka lastiklerin çekiş kaybı (power-over ve clutch-kick ile koparma).
* **Aksiyon:** Anlık debriyaj tokatlama (`Clutch`), el freniyle ağırlık aktarma (`Handbrake`) ve direksiyonu ters kırma (`Counter-Steering`).
* **Ödül Fonksiyonu:** En hızlı gitmek yerine; $\text{Hız} \times \sin(\text{Kayma Açısı}) \times \text{Apeks Yakınlığı} - \text{Spin Cezası}$.

### 2. GT / Pist Yarışı Ajanı (Track Attack)
* **Fizik:** Maksimum yol tutuşu (grip driving), minimum lastik kayması.
* **Aksiyon:** Apeks öncesi kademeli fren bırakma (trail braking), pürüzsüz viraj çıkışı ve erken apeks/geç apeks optimizasyonu.
* **Ödül Fonksiyonu:** Sektör tamamlama süresi, apeks teğetliği ve direksiyon pürüzsüzlüğü (`steer_smoothness`).

### 3. Heel-and-Toe & Şanzıman Senkronizasyonu
* **Fizik:** Vites küçültme anında motor devri (RPM) ile şanzıman hızının eşleşmemesinden doğan "şanzıman şoku" ve arka tekerlek kilitlenmesi.
* **Aksiyon:** Sağ ayak ucu frendeyken topukla gaza ara gazı (rev-match blip) verilmesi, debriyajın eşzamanlı bırakılması.
* **Ödül Fonksiyonu:** Vites geçişlerindeki devir sapması ve şanzıman darbe katsayısına ceza uygulanması.

---

## Faz 4: Çoklu Ajan, Yarış Zekası ve Sokak Dinamikleri (Multi-Agent RL)
Pistte tek başına tur atmaktan çıkıp rakiplerle temas halinde yarışma.

* **Self-Play (Kendi Kopyalarına Karşı Yarış):**
  - Modelin kendi geçmiş sürümleriyle aynı anda piste sürülmesi.
* **Yarış Zekası (Racecraft):**
  - Düzlükte öndeki aracın hava koridoruna girme (slipstream / drafting).
  - Viraj girişinde geç frenajla (late braking) iç çizgiyi kapatarak sollama yapma.
  - Temas etmeden yan yana viraj alma ve defansif çizgi savunması.

---

## Faz 5: Sim-to-Sim Transfer (Unity $\rightarrow$ Unreal Engine 5)
Eğitimin son durağı olan fotogerçekçi grafikler ve ileri fizik simülasyonu.

* **Unreal Engine 5 Chaos Vehicles Entegrasyonu:**
  - Unreal tarafında C++ tabanlı binary TCP köprüsünün kurulması.
  - Gelişmiş diferansiyel kilitleri (LSD), tork vektörleme ve aerodinamik bastırma kuvveti (downforce) parametreleri.
* **Fine-Tuning:**
  - Unity'de eğitilmiş temel modelin (`checkpoint`), Unreal Engine'in yüksek gerçeklikli Chaos fiziğinde kısa süreli pekiştirmeli öğrenme ile kalibre edilmesi.