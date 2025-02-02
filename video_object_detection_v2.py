import os
import cv2
import argparse
import time
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import glob

# Ekran bölgeleri
EKRANLAR = [
    (0, 0, 680, 640),
    (620, 0, 1300, 640),
    (1240, 0, 1920, 640),
    (0, 440, 680, 1080),
    (620, 440, 1300, 1080),
    (1240, 440, 1920, 1080)
]

class RealTimeVideoProcessor:
    def __init__(self, video_path, model_path, split_model_path, device='cpu'):
        self.video_path = video_path
        self.device = device
        self.model = YOLO(model_path).to(self.device)
        self.split_model = YOLO(split_model_path).to(self.device)
        
        # Dosya yapılandırması
        self.base_dir = os.path.dirname(video_path)
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.setup_directories()
        
        # Sayaçlar
        self.frame_counter = 0
        self.total_detections = 0
        
    def setup_directories(self):
        self.output_dir = os.path.join(self.base_dir, self.video_name)
        self.dirs = {
            'with_detections': os.path.join(self.output_dir, 'images_with_detections'),
            'without_detections': os.path.join(self.output_dir, 'images_without_detections'),
            'labels': os.path.join(self.output_dir, 'labels'),
            'predict': os.path.join(self.output_dir, 'predict')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video yazıcıyı başlat
        writer = cv2.VideoWriter(
            os.path.join(self.dirs['predict'], 'processed_video.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            int(cap.get(cv2.CAP_PROP_FPS)),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        progress = tqdm(total=total_frames, desc=f"İşleniyor: {self.video_name}", unit='frame')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, has_detection = self.process_frame(frame)
            writer.write(processed_frame)
            self.save_results(frame, has_detection)
            
            progress.update(1)
            self.frame_counter += 1

        cap.release()
        writer.release()
        progress.close()
        
        print(f"\nİşlem tamamlandı! Toplam: {self.total_detections} tespit kaydedildi.")

    def process_frame(self, frame):
        # Ana model ile tespit
        results = self.model(frame, verbose=False)
        has_detection = len(results[0].boxes) > 0
        
        # Tespit yoksa ekran bölgelerini kontrol et
        if not has_detection:
            has_detection = self.check_screen_regions(frame)

        # Görselleştirme
        visualized_frame = results[0].plot()
        return visualized_frame, has_detection

    def check_screen_regions(self, frame):
        for region in EKRANLAR:
            x1, y1, x2, y2 = region
            roi = frame[y1:y2, x1:x2]
            
            results = self.split_model(roi, verbose=False)
            if len(results[0].boxes) > 0:
                return True
        return False

    def save_results(self, frame, has_detection):
        # Dosya isimlendirme
        file_id = f"{self.frame_counter:06d}"
        
        if has_detection:
            img_path = os.path.join(self.dirs['with_detections'], f"{file_id}.jpg")
            self.total_detections += 1
        else:
            img_path = os.path.join(self.dirs['without_detections'], f"{file_id}.jpg")

        # Görseli ve etiketi kaydet
        cv2.imwrite(img_path, frame)
        self.save_labels(file_id, frame.shape)

    def save_labels(self, file_id, frame_shape):
        # Etiket dosyasını oluştur (örnek implementasyon)
        label_path = os.path.join(self.dirs['labels'], f"{file_id}.txt")
        with open(label_path, 'w') as f:
            # Bu kısım gerçek tespit verileriyle doldurulmalı
            f.write("0 0.5 0.5 0.3 0.3\n")  # Örnek YOLO formatı

def main():
    parser = argparse.ArgumentParser(description="Gerçek Zamanlı Video İşleme")
    parser.add_argument("--input", required=True, help="Giriş videosu veya dizini")
    parser.add_argument("--model", required=True, help="Ana model yolu (.pt)")
    parser.add_argument("--split-model", required=True, help="Bölge modeli yolu (.pt)")
    parser.add_argument("--device", choices=['cpu', 'cuda'], default='cpu', help="İşlemci seçimi")
    
    args = parser.parse_args()

    # Dosyaları bul
    video_files = []
    if os.path.isfile(args.input):
        video_files.append(args.input)
    elif os.path.isdir(args.input):
        video_files = [f for f in glob.glob(os.path.join(args.input, "*.mp4"))]

    # İşlemleri başlat
    start_time = time.time()
    for video_path in video_files:
        processor = RealTimeVideoProcessor(
            video_path=video_path,
            model_path=args.model,
            split_model_path=args.split_model,
            device=args.device
        )
        processor.process_video()

    print(f"\nToplam süre: {time.time()-start_time:.2f} saniye")

if __name__ == "__main__":
    main()