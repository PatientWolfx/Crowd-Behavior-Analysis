from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_people(frame):

    results = model(frame)

    person_boxes = []

    for r in results:
        for box in r.boxes:

            class_id = int(box.cls)

            if class_id == 0:  # person class
                person_boxes.append(box.xyxy)

    return person_boxes