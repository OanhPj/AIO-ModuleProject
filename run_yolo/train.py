from ultralytics import YOLOv10

MODEL_PATH = 'D:/AIO_2024/Module_1/project_yolov10/yolov10/ultralytics/yolov10n.pt'
model = YOLOv10(MODEL_PATH)

YAML_PATH = 'D:/AIO_2024/Module_1/Data/yolov10/Safety_Helmet_Dataset/data.yaml'
EPOCHS = 20
IMG_SIZE = 640
BATCH_SIZE = 16

model.train(data=YAML_PATH, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH_SIZE)
