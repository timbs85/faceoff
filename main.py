import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

scaling_factor = 0.2
cap = cv2.VideoCapture('media/RNC_outfall_from_first_amendment_to_security_cameras-_WMNF_News.webm')

expand = 0.5

app = FaceAnalysis(providers=['CPUExecutionProvider']) # CPU only
app.prepare(ctx_id=0, det_size=(int(640 * scaling_factor), int(640 * scaling_factor))) # lower=fast, higher=robust

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)  # returns faces with .bbox = [x1,y1,x2,y2]

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        bw, bh = x2 - x1, y2 - y1
        ex, ey = int(bw*expand), int(bh*expand)
        x1 = max(0, x1 - ex)
        y1 = max(0, y1 - ey)
        x2 = min(frame.shape[1], x2 + ex)
        y2 = min(frame.shape[0], y2 + ey)
        frame[y1:y2, x1:x2] = (0, 0, 0)  # black rectangle

        cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()