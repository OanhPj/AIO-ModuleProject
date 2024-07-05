from ultralytics import YOLOv10

# Model
model_path = 'D:/AIO_2024/Module_1/project_yolov10/yolov10/ultralytics/yolov10n.pt'
model = YOLOv10(model_path)

# Image predict
image_path = 'D:/AIO_2024/Module_1/Data/yolov10/Safety_Helmet_Dataset/train/images/helmet-13-_jpg.rf.40efdceb104dcedeb1d02fc8ecd350b8.jpg'
results = model(source=image_path)[0]

# Predict
results.save('D:/AIO_2024/Module_1/predict/predict.jpg')
