# Görüntüleyiciyi çalıştırma

Komutları depo kökünde çalıştırın. Görüntüleyici yereldir ve yalnız doğrulama parçalarını kullanır. Kayıtlı oynatma, özgün deney verileri olmayan bir klonda da çalışır. Canlı mod ek girdiler gerektirir. Hiçbir mod eğitim yapmaz veya deney kayıtlarını güncellemez; test bölmesi kapalıdır.

## Kayıtlı oynatıcı: sunum için önerilen yol

```powershell
uv sync --locked --extra demo
uv run --extra demo python scripts/live_demo.py --recorded --port 8080
```

Terminalde yazan yerel adresi tarayıcıda açın; varsayılan adres `http://127.0.0.1:8080` olur. Terminal açık kalmalıdır. Durdurmak için Ctrl+C kullanın. Port doluysa başka bir `--port` seçin ve terminalin bildirdiği gerçek adrese bağlanın.

Oynatıcı, `results/demo/manifest.json` içindeki SHA-256 ve boyutları yüklemeden önce doğrular. Uyuşmazlıkta yüklemeyi reddeder. Paketi değiştirmez, yeniden üretmez ve politika çağırmaz. Özgün mesh, görünürlük/robot önbelleği veya eğitim checkpoint'i gerekmez. Robotun görsel ağları ve lisansları depoda bulunur; çalışma sırasında robot modeli indirilmez.

Parça ve yöntemi seçin. Kayıtlı paket yalnız varyant sıfır ve seed sıfır içerir. Yöntemler: Greedy-IG/maliyet, MCTS, ILP + TSP, ortak planlayıcı, doğrusal potansiyelli SARSA, değer tahminli SARSA ve RegQ.

| Kontrol | İşlev |
|---|---|
| Başlat / Duraklat | Oynat veya duraklat |
| Adım | Bir tarama pası ilerle; son pasın ardından eve dönüşü oynat |
| Sıfırla | Başlangıç durumuna dön |
| Hız | Yalnız animasyon hızını değiştir |
| Simüle zaman | Kayıtlı bölümde zamanı seç |
| Genel / Parça / Yan | Hazır kamera konumu |
| Aday paslar | Erişilebilir, IK ile erişilemez ve çarpışma maskeli pasları göster |
| Çarpışma modeli | Görsel UR5 ağından ayrı kapsül/kutu modelini aç |

Sürükleyerek kamerayı döndürün, fare tekeriyle yakınlaşın. Sahne yerel istemciler arasında paylaşılır; sunumda tek operatör kullanın. Hedef kapsama görüldüğünde hemen durmayın: eve dönüş de toplam süreye dahildir.

## Karşılaştırma

Kayıtlı modda **Karşılaştır** sekmesinden iki hücreyi etkinleştirin ve ikinci yöntemi seçin. **Oynat** sekmesine dönüp iki bölümü aynı simüle zamanla ilerletin. Daha kısa bölüm bitince ev konumunda bekler. Hücrelerin yan yana yerleştirilmesi yalnız görüntüleme dönüşümüdür.

Grafikte seçili yöntem vurgulanır, aynı varyantın diğer kayıtlı yöntemleri soluk gösterilir. Aday renkleri, yöntem çizgileri ve nokta durumları ortak temadan gelir. Paket içindeki eski stil metadatası tarihsel olarak korunur; güncel görüntüleyici renkleri ayrı ortak temadan alır.

## Canlı mod: özgün doğrulama girdileri gerekir

```powershell
uv run --extra demo python scripts/live_demo.py --weights weights/linear_learners.json --port 8080
```

Canlı modun ek önkoşulları:

- Özgün doğrulama görünürlük ve robot önbellekleri.
- Özgün doğrulama bölüm kayıtları, referans planlar ve kaynak checkpoint doğrulama dosyaları.
- Paylaşılan kuyruk ortalamaları için `weights/linear_learners.json`; dosyanın hash'i kaynak metadatasına ve sayısal değerleri özgün checkpoint'lere karşı kontrol edilir.

Kaynak-only klon bu girdileri içermez. Ağırlık dosyası bunların yerini almaz. Eksik deney girdileri sırf sunum açmak için yeniden üretilmemeli; kayıtlı mod kullanılmalıdır. Yalnız özgün önkoşullar mevcutsa canlı doğrulama yapılır.

Canlı seçenekler doğrulama parçaları, kayıtlı varyantları ve resmi seed'lerdir. Greedy, MCTS, SARSA ve RegQ donmuş ayarlarla karar verir. Keşif yoktur. Değer tahmini potansiyeli eğitimde kullanılmıştır; donmuş doğrusal çıkarım sırasında yeniden hesaplanmaz. ILP + TSP ve ortak planlayıcı **kayıtlı plan** olarak etiketlenir ve yeniden çözülmez.

Daha sonraki bir varyantta MCTS'nin kayıtlı rastgele sayı akışını yeniden kurması hazırlık süresi gerektirebilir. Canlı karşılaştırma ikinci bir politika çalıştırmaz; iki hücre karşılaştırması kayıtlı modda kullanılır.

## Doğrulama, kapsam ve süre

Bölüm sonunda canlı dizi, çevrim süresi, normalize süre ve başarı özgün doğrulama kaydıyla tam hassasiyette karşılaştırılır. **Kayıtla birebir** etiketi eşleşmeyi, **uyuşmazlık** etiketi farklı alanları gösterir. Uyarıyı gizlemeyin; kaydı veya manifesti uydurmak için değiştirmeyin.

Kayıtlı mod özgün paket doğrulamasını ve bölümün önceden yapılmış tam/kısmi oracle kontrolünü taşır. Öğrenenlerin nihai kapsama ve eklem durumları yeniden oynatmadan türetilmiştir; tarihsel adım izleri bağımsız olarak saklanmamıştır. Bu nedenle etiket, kısmi oracle ile tutarlılıktır.

Birincil kapsama yalnız erişilebilir yüzey birleşimine göredir. Gri kalan yüzeylerin varlığıyla yüksek kapsama birlikte görülebilir. Noktalar önce ölçülmüş, bu pasta yeni ölçülmüş, erişilebilir ama ölçülmemiş ve ölçülemez sınıflarını ayırır. Renk açıklaması **Bilgi** sekmesindedir.

UR5'in görsel ağları çarpışma modelinden ayrıdır. Eklem durumları arasındaki enterpolasyon yalnız animasyondur. Önceden yoğun örneklenmiş bir çarpışma aralığında ilgili gövde kırmızı gösterilir; uyarı yolu onarmaz. Gösterilen derinlik, kayıtlı aralığın en büyük örneklenmiş penetrasyonudur; yeni bir kare-bazlı geometrik denetim değildir.

Canlı süre **bu makine, görüntüleme yükü dahil** olarak yorumlanır; ayrı sıcak-önbellek planlama benchmark'ıyla doğrudan karşılaştırılmaz. Kayıtlı oynatma yeni karar süresi ölçümü değildir.

## Atıf ve lisans

Atıf panelini görünür tutun. Parça adı, yazar, kaynak, kayıtlı lisans ailesi, sürüm durumu ve yapılan değişiklikleri içerir. Nominal mesh ile varyant noktaları küçük geometrik farklılıklar taşır. BY-SA türevleri kendi lisans koşullarını korur. Kaynak sayfa erişimiyle doğrulanamayan sürümler tahmin edilmez; güncel durum ve parça bağlantıları [araştırma raporundadır](../README.md#11-veri-ve-lisans-atıfları).

## Sorun giderme

| Belirti | Yapılacak işlem |
|---|---|
| Görüntüleyici modülü bulunamıyor | Kurulumda ve çalıştırmada `--extra demo` kullanın |
| Canlı modda önbellek/checkpoint eksik | `--recorded` ile kayıtlı oynatıcıya geçin |
| Hash veya kimlik uyuşmazlığı | Yüklemeyi durdurun; özgün kanıtı değiştirmeyin |
| Boş sahne | GUI hata metnini, terminali ve tarayıcının WebGL desteğini kontrol edin |
| MCTS hazırlığı yavaş | Özgün akışın hazırlanmasını bekleyin veya kayıtlı modu kullanın |
| Kapsama hedefi görüldü, son doğrulama yok | Eve dönüş adımını tamamlayın |
| Dışa aktarıcı mevcut hedefi reddediyor | Dışa aktarma oynatma değildir; paketi yeniden üretmeyin |

Sunum öncesi kayıtlı oynatıcıyı ilgili makinede deneyin. Ağ tüneli veya public sunucu başlatılmaz; hizmet yalnız yerel arayüze bağlanır.
