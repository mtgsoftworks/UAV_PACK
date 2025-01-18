import cv2,os,shutil

# Görüntü ve etiket klasör yollarını tanımla
directir=r""
image_folder = f'{directir}/images_with_detections'
label_folder = f'{directir}/labels'
# directir=r"C:\Users\Gedik\Desktop\Train_v1-001\102"
# image_folder = f'{directir}/remove-images'
# label_folder = f'{directir}/remove-labels'


# Görüntü ve etiket dosyalarını sırala
images = sorted([img for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')])
labels = sorted([lbl for lbl in os.listdir(label_folder) if lbl.endswith('.txt')])

current_image_index = 0
total_images = len(images)

def is_point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2
def move_image_and_label(image_index, dest_image_folder, dest_label_folder):
    global images, labels

    source_image_path = os.path.join(image_folder, images[image_index])
    source_label_path =f"{label_folder}/{images[image_index].replace('.jpg','.txt')}"

    # Hedef klasörleri oluştur (varsa, bu adımı atlar)
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_label_folder, exist_ok=True)

    # Dosyaları hedef klasörlere taşı
    shutil.move(source_image_path, os.path.join(dest_image_folder, images[image_index]))
    try:
        shutil.move(source_label_path, os.path.join(dest_label_folder,images[image_index].replace('.jpg','.txt')))
        labels.remove(images[image_index].replace('.jpg','.txt'))

    except FileNotFoundError:
        pass 
    # Listeleri güncelle

    del images[image_index]


def remove_label_from_file(label_path, box_to_remove, image_width, image_height):
    new_lines = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, w, h = map(float, line.split())
            box = calculate_box_coordinates(x_center, y_center, w, h, image_width, image_height)
            # Eğer kutu, kaldırılacak kutu değilse, listeye ekle
            if not is_point_in_box((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box_to_remove):
                new_lines.append(line)
    
    # Güncellenmiş etiketleri dosyaya yaz
    with open(label_path, 'w') as file:
        file.writelines(new_lines)

def mouse_callback(event, x, y, flags, param):
    global current_image_index, width, height
    if event == cv2.EVENT_MBUTTONDOWN:
        image, label_path = param
        for box in bounding_boxes:
            if is_point_in_box(x, y, box):
                remove_label_from_file(label_path, box, width, height)
                display_image(current_image_index)  # Refresh the image display
                break

def calculate_box_coordinates(x_center, y_center, w, h, image_width, image_height):
    x1 = int((x_center - w / 2) * image_width)
    y1 = int((y_center - h / 2) * image_height)
    x2 = int((x_center + w / 2) * image_width)
    y2 = int((y_center + h / 2) * image_height)
    return x1, y1, x2, y2

def draw_bounding_boxes(image, label_path):
    global width, height, bounding_boxes
    height, width, _ = image.shape
    bounding_boxes = []

    # Etiket dosyasını oku ve sınırlayıcı kutuları çiz

    try:
        with open(label_path, 'r') as file:
            for line in file:
                print("linee",line)
                class_id, x_center, y_center, w, h = map(float, line.split())
                print(class_id)
                box = calculate_box_coordinates(x_center, y_center, w, h, width, height)
                bounding_boxes.append(box)
                if class_id==0:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
                else:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
    except FileNotFoundError:
        pass
def display_image(index):
    global current_image_index, total_images

    image_path = os.path.join(image_folder, images[index])
    label_path = f"{label_folder}/{images[index].replace('.jpg','.txt')}"

    #label_path = os.path.join(label_folder, labels[index])

    # Etiketli ve etiketsiz iki ayrı görüntü yükle
    print(image_path)
    print(label_path)

    image_with_labels = cv2.imread(image_path)
    image_without_labels = image_with_labels.copy()
    draw_bounding_boxes(image_with_labels, label_path)
    # İndeks bilgisini etiketli görüntüye yaz
    info_text = f'Image {index + 1} of {total_images}'
    cv2.putText(image_with_labels, info_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #cv2.rectangle(image_with_labels,(320,0),(960,640),(0,0,0),2)
    # Her iki görüntüyü de göster
    
    cv2.imshow('Image with Bounding Boxes',cv2.resize(image_with_labels,(1920,1080))) #cv2.resize(image_with_labels,(1920,1080)))
    #cv2.imshow('Image without Bounding Boxes', cv2.resize(image_without_labels,(1920,1080)))

    cv2.setMouseCallback('Image with Bounding Boxes', mouse_callback, [image_with_labels, label_path])


cv2.namedWindow('Image with Bounding Boxes')

while True:
    display_image(current_image_index)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('d'):
        current_image_index = (current_image_index + 1) % total_images
    elif key == ord('a'):
        current_image_index = (current_image_index - 1) % total_images
    elif key == ord('w'):
        move_image_and_label(current_image_index, f'{directir}/remove-images', f'{directir}/remove-labels')
        total_images -= 1
        if current_image_index == total_images:
            current_image_index -= 1
    elif key == ord('s'):
        move_image_and_label(current_image_index, f'{directir}/images-revizyon', f'{directir}/labels-revizyon')
        total_images -= 1
        if current_image_index == total_images:
            current_image_index -= 1
    elif key == ord("q"):
        break
