import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
cap = cv2.VideoCapture("Video/cars.mp4")
total_Counts = []
model = YOLO('Yolo-Weights/yolov8n.pt')
mask = cv2.imread('mask.png')
limits = [272, 244, 472, 246]
# def mousePoints(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x,y)
#Tracking
tracker = Sort(max_age=20, min_hits = 3, iou_threshold = 0.3)
                            # max_age -> Agar koi
                            # car/person detect hua aur phir
                            # 20 consecutive frames tak
                            # detector usay na pakray → tracker sochega
                            # object scene se chala gaya → tracking stop

while(True):
    res, frame = cap.read()
    frame = cv2.resize(frame, (900, 600))  # width, height
    ImageRegion = cv2.bitwise_and(frame,  mask)
    results = model(ImageRegion, stream = True)
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            conf = (math.ceil(box.conf[0]* 100))/100

            cls_id = int(box.cls[0])
            classname = model.names[cls_id]
            if classname == "car" or classname=='truck' or classname=='motorbike' or classname=='bus' and conf > 0.4:
                # cvzone.putTextRect(frame, f'{classname} {conf}', (max(0,x1),max(0,y1-20)), scale=0.75, thickness=1,
                #                offset = 1)
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=10)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections,currentArray))

    trackerResult = tracker.update(detections)
    for result in trackerResult:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = map(int, [x1, y1, x2, y2, id])
        w, h = x2 - x1, y2 - y1

        cx,cy =x1 + w // 2, y1 + h // 2
        if limits[0] < cx < limits[2] and limits[1]-15 < cy < limits[3]+15:
            if total_Counts.count(id) ==0:
                total_Counts.append(id)

        cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(255,255,0))
        cvzone.putTextRect(frame, f'{id}', (max(0, x1), max(0, y1 - 20)), scale=0.6, thickness=1,
                          offset=3)
        cv2.line(frame,(limits[0], limits[1]), (limits[2], limits[3]), (0,0, 255), thickness=5)
        cv2.circle(frame, (cx, cy), 3,(255, 255, 0), cv2.FILLED)
        cvzone.putTextRect(frame, f"Counts: {len(total_Counts)}", (50,50),
                           scale=2, thickness=3, offset=3)

    cv2.imshow('Image',frame)
    # cv2.setMouseCallback("Image", mousePoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()