import glob
import os
from ultralytics import YOLO
import cv2
import numpy as np

# Ekran koordinatları listesi
ekranlar = [
    (0, 0, 680, 640),
    (620, 0, 1300, 640),
    (1240, 0, 1920, 640),
    (0, 440, 680, 1080),
    (620, 440, 1300, 1080),
    (1240, 440, 1920, 1080)
]

# Dilim algılama sınıfı
class slice_detect:
    def __init__(self, video_path, write_count, coco_model="yolov8x.pt", coco_split_model="yolov8x.pt", my_model="best.pt", my_split_model="best.pt"):
        # Video dosyasını açma
        self.video = cv2.VideoCapture(video_path)
        self.path, name = video_path.split("\\")[1], video_path.split("\\")[-1]
        self.name = name[:-4]
        self.write_count = write_count
        self.frame_count = 1
        self.to_frame_count = 1

        # Klasör oluşturma
        if len(glob.glob(self.path)) == 0:
            os.mkdir(self.path)
        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video.get(cv2.CAP_PROP_FPS))

        # Modelleri yükleme
        self.coco_model = YOLO(coco_model)
        self.coco_split_model = YOLO(coco_split_model)
        # self.my_model = YOLO(my_model)
        # self.my_split_model = YOLO(my_split_model)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.folder_control()

        to_path = f"{self.path}/{self.name}/predict/{self.name}.mp4"
        self.output = cv2.VideoWriter(to_path, fourcc, fps, (frame_width, frame_height))

        self.detect()

    def folder_control(self):
        # Klasörleri kontrol etme ve oluşturma
        if len(glob.glob(f"{self.path}/{self.name}")) == 0:
            os.mkdir(f"{self.path}/{self.name}")

        if len(glob.glob(f"{self.path}/{self.name}/images_with_detections")) == 0:
            os.mkdir(f"{self.path}/{self.name}/images_with_detections")
        if len(glob.glob(f"{self.path}/{self.name}/images_without_detections")) == 0:
            os.mkdir(f"{self.path}/{self.name}/images_without_detections")
        if len(glob.glob(f"{self.path}/{self.name}/labels")) == 0:
            os.mkdir(f"{self.path}/{self.name}/labels")

        if len(glob.glob(f"{self.path}/{self.name}/predict")) == 0:
            os.mkdir(f"{self.path}/{self.name}/predict")

    def frame_true_false(self, frame):
        # Frame'in belirli bir gri tonuna sahip olup olmadığını kontrol etme
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        unique_elements, counts = np.unique(img, return_counts=True)
        img_true_false = False
        for num, count in zip(unique_elements, counts):
            if num == 128:
                if count > 100000:
                    img_true_false = True
        return img_true_false

    def detect(self):
        # Algılama işlemi
        self.detectno = False
        calisma = 1
        img_list = []
        while True:
            ret, frame = self.video.read()

            if not ret:
                break
            frame_new = frame.copy()
            if not self.detectno:
                if len(img_list) > 20:
                    img_list = []
                img_list.append(frame)
            if (calisma == 1) & self.detectno:
                for i in img_list:
                    boxes = []
                    self.save_boxes_and_frame(frame, boxes)
                    self.frame_count += 1
                    self.output.write(frame_new)
                calisma = 0
            results = self.coco_model([frame], classes=0, device=1)
            for result in results:
                if len(result.boxes.xywhn) > 0:
                    self.detectno = True
                    boxes = result.boxes.xyxy
                    self.save_boxes_and_frame(frame, boxes)
                else:
                    frame_new = self.detect2(frame)
                if len(result.boxes.xywhn) > 0:
                    for bbox in result.boxes.xyxy:
                        frame_new = cv2.rectangle(frame_new, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
            self.frame_count += 1
            self.output.write(frame_new)
        cv2.destroyAllWindows()
        self.video.release()
        self.output.release()

    def detect2(self, frame):
        # Algılama işleminin ikinci aşaması
        print(self.to_frame_count)
        frame_new = frame.copy()
        bboxs_list = self.label_to_control(frame)
        bboxs = self.io(bboxs_list)

        if len(bboxs_list) > 0:
            self.detectno = True
            self.save_boxes_and_frame(frame, bboxs)

            for bbox in bboxs:
                frame_new = cv2.rectangle(frame_new, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
        else:
            if self.detectno:
                boxes = []
                self.save_boxes_and_frame(frame, boxes)

        return frame_new

    def save_boxes_and_frame(self, frame, boxes):
        # Frame'i ve algılama sonuçlarını kaydetme
        if len(boxes) > 0:
            frame_path = f"{self.path}/{self.name}/images_with_detections/%0.6d.jpg" % self.to_frame_count
        else:
            frame_path = f"{self.path}/{self.name}/images_without_detections/%0.6d.jpg" % self.to_frame_count
        label_path = f"{self.path}/{self.name}/labels/%0.6d.txt" % self.to_frame_count

        w, h = frame.shape[1], frame.shape[0]

        with open(label_path, "w") as file:
            if len(boxes) > 0:
                for box in boxes:
                    print(box)
                    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                    x_center = (xmin + xmax) / (2 * w)
                    y_center = (ymin + ymax) / (2 * h)
                    bbox_width = (xmax - xmin) / w
                    bbox_height = (ymax - ymin) / h
                    line = f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n"
                    file.write(line)
        cv2.imwrite(frame_path, frame)
        self.to_frame_count += 1

    def label_to_control(self, frame):
        # Frame'i kontrol etme ve kutuları listeleme
        bboxs_list = []
        for control in range(1, 7):
            secilen_ekran = ekranlar[control - 1]
            image = frame[secilen_ekran[1]:secilen_ekran[3], secilen_ekran[0]:secilen_ekran[2]]

            results = self.coco_split_model([image], classes=0, device=1)
            for result in results:
                if len(result.boxes.xywhn) > 0:
                    for boxes in result.boxes.xyxy:
                        xmin, ymin, xmax, ymax = int(boxes[0] + secilen_ekran[0]), int(boxes[1] + secilen_ekran[1]), int(boxes[2] + secilen_ekran[0]), int(boxes[3] + secilen_ekran[1])
                        bboxs_list.append([xmin, ymin, xmax, ymax])
        return bboxs_list

    def label_to_control2(self, frame):
        # Frame'i kontrol etme ve kutuları listeleme (ikinci yöntem)
        bboxs_list = []
        for control in range(1, 7):
            secilen_ekran = ekranlar[control - 1]
            image = frame[secilen_ekran[1]:secilen_ekran[3], secilen_ekran[0]:secilen_ekran[2]]

            results = self.my_split_model([image], classes=0, device=1)
            for result in results:
                if len(result.boxes.xywhn) > 0:
                    for boxes in result.boxes.xyxy:
                        xmin, ymin, xmax, ymax = int(boxes[0] + secilen_ekran[0]), int(boxes[1] + secilen_ekran[1]), int(boxes[2] + secilen_ekran[0]), int(boxes[3] + secilen_ekran[1])
                        bboxs_list.append([xmin, ymin, xmax, ymax])
        return bboxs_list

    def iou(self, box1, box2):
        # Intersection Over Union (IOU) hesaplama
        (x1_1, y1_1, x2_1, y2_1) = box1
        (x1_2, y1_2, x2_2, y2_2) = box2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = box1_area + box2_area - inter_area
        iou_val = inter_area / float(union_area)

        return iou_val

    def io(self, bboxs):
        # Kutuların birleştirilmesi
        for item in range(len(bboxs)):
            count = item + 1
            while count < len(bboxs):
                bbox1 = bboxs[item]
                bbox2 = bboxs[count]
                if self.iou(bbox1, bbox2) > 0.3:
                    bboxs_ek = np.array(bboxs)
                    bboxs_ek = bboxs_ek[[item, count]]
                    xmin = bboxs_ek[:, 0].min()
                    ymin = bboxs_ek[:, 1].min()
                    xmax = bboxs_ek[:, 2].max()
                    ymax = bboxs_ek[:, 3].max()
                    bboxs[item] = [xmin, ymin, xmax, ymax]
                    bboxs.pop(count)
                else:
                    count += 1
        return bboxs

if __name__ == "__main__":
    # Video dosyalarını işleme
    for path in glob.glob("C:/Users/beu/Desktop/SAKARYA/Mesut/WORK_3/**/**.avi"):
        print(path)
        slice_detect(path, write_count=15, coco_model="best.pt", coco_split_model="best.pt")
