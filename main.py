# pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: # Si recibe un frame vacío, salir del bucle (hubo error)
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # for result in results:
    #    boxes = result.boxes
    #    for box in boxes:
    #        x1, y1, x2, y2 = map(int, box.xyxy[0])
    #        conf = box.conf[0]
    #        cls = int(box.cls[0])
    #        label = model.names[cls]
    #        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # Presiona 'q' o 'Esc' para salir
        break

# Librar todos los recursos
cap.release()
cv2.destroyAllWindows()