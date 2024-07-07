from ultralytics import YOLOv10

# Model
model_path = 'D:/AIO_2024/Module_1/project_yolov10/yolov10/ultralytics/yolov10n.pt'
model = YOLOv10(model_path)

# Image predict
video_path = 'D:/AIO_2024/Module_1/Data/yolov10/video/How to cross a street in Ho Chi Minh City (Saigon), Vietnam.mp4'
results = model(source=video_path, show=True)
