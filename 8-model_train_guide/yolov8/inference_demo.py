from ultralytics import YOLO


# train
# Load a model
# model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# # Train the model
# model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)


# inference
# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
print(results)
