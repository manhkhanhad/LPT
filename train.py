from ultralytics import YOLO

# Load a model  
model = YOLO("yolo11m.pt")

# Train the model  
model.train(data="/home/khanh/data/LPT/yolo_config.yaml", #path to yaml file  
           batch=32, #number of batch size  
           epochs=50, #number of epochs  
           device=1) #device ‘0’ if gpu else ‘cpu’