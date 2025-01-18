# UAV_PACK

## Video Object Detection and Tracking

Bu proje, bir video dosyasındaki nesneleri tespit etmek için YOLOv8 modellerini kullanan bir Python uygulamasıdır. Uygulama, video üzerinde nesne algılaması yapar, tespit edilen nesnelerin konumlarını kaydeder ve yeni bir video dosyasına tespit edilen nesneleri işaretler. Ayrıca, tespit edilen nesnelerin koordinatları metin dosyalarına yazılır.

## Özellikler

- **Video İşleme**: Verilen video dosyasındaki her bir kare işlenir.
- **Nesne Algılama**: YOLOv8 modelleri kullanılarak her çerçevede nesne tespiti yapılır.
- **Bölgesel Tespit**: Video görüntülerinin belirli bölümlerinde nesneler tespit edilir.
- **Sonuçların Kaydedilmesi**: Tespit edilen nesnelerin koordinatları metin dosyalarına kaydedilir ve tespit edilen kareler yeni bir video dosyası olarak kaydedilir.

## Kullanım

### Gereksinimler

Proje aşağıdaki Python kütüphanelerine ihtiyaç duyar:

- `opencv-python` : Video işleme ve görüntü gösterimi için.
- `numpy` : Matematiksel işlemler ve verilerin işlenmesi için.
- `ultralytics` : YOLOv8 modelini kullanabilmek için.

Gerekli kütüphaneleri yüklemek için aşağıdaki komutları kullanabilirsiniz:

```bash
pip install opencv-python numpy ultralytics
```


# Video Frame Extraction

Bu proje, bir video dosyasındaki belirli bir zaman aralığındaki kareleri (frames) çıkarmak ve bu kareleri PNG formatında kaydetmek için kullanılan bir Python uygulamasıdır. Kullanıcıdan alınan video yolu, başlangıç zamanı ve bitiş zamanına göre kareler çıkarılır ve her bir kare, belirtilen dizine kaydedilir.

## Özellikler

- **Video Okuma**: Verilen video dosyasındaki kareler okunur.
- **Zaman Aralığı Seçimi**: Başlangıç ve bitiş zamanına göre video kareleri seçilir.
- **Kareleri Kaydetme**: Seçilen kareler belirtilen dizine PNG formatında kaydedilir.
- **İlerleme Takibi**: İşlem sırasında kalan süre ve tamamlanma yüzdesi kullanıcıya bildirilir.

## Kullanım

### Gereksinimler

Proje aşağıdaki Python kütüphanelerine ihtiyaç duyar:

- `opencv-python` : Video işleme ve görüntü kaydetme için.
- `os` ve `time` : Dosya yönetimi ve zaman hesaplamaları için.

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:

```bash
pip install opencv-python
```


# Image and Label Management Tool

Bu Python aracı, görüntüleri ve etiketlerini belirli klasörlere taşıma, etiketlerdeki sınırlayıcı kutuları (bounding boxes) düzenleme ve kaldırma işlemleri için kullanılır. Araç, kullanıcıya görüntüler üzerinde etkileşimli olarak çalışmayı ve belirli etiketleri silmeyi sağlar.

## Özellikler

- **Görüntü ve Etiket Yönetimi**: Görüntüler ve etiket dosyaları arasında taşıma işlemleri yapılır.
- **Etiket Düzenleme**: Kullanıcı, bir görüntüye ait etiketleri tıklayarak kaldırabilir.
- **Görüntüleme ve Etkileşim**: Görüntüler üzerinde sınırlayıcı kutuları çizilir ve bu kutulara tıklayarak etiketler kaldırılabilir.
- **Kısayol Tuşları**:
    - `d`: Bir sonraki görüntüye geç.
    - `a`: Bir önceki görüntüye geç.
    - `w`: Mevcut görüntü ve etiketleri "remove" klasörlerine taşı.
    - `s`: Mevcut görüntü ve etiketleri "revizyon" klasörlerine taşı.
    - `q`: Uygulamadan çık.

## Gereksinimler

Proje aşağıdaki Python kütüphanelerine ihtiyaç duyar:

- `opencv-python` : Görüntü işleme ve kullanıcı etkileşimi için.
- `os` ve `shutil` : Dosya ve klasör yönetimi için.

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:

```bash
pip install opencv-python
```
