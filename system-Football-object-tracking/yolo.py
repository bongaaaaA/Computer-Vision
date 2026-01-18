from ultralytics import YOLO
# using model training for dataset 
model = YOLO('Models/best.pt')

reuslts = model.predict('Cinput_vedio/test.mp4', save=True)
print(reuslts[0])

for box in reuslts[0].boxes:
    print(box)
