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
