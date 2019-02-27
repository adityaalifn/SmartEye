import time

import cv2

from face_capture import FaceCapture

cap = cv2.VideoCapture("assets/test_video.mp4")
fps_time = 0
frame_rate = int(cap.get(5))
fc = FaceCapture(model="cnn")
fps_count = 0
while cap.isOpened():
    fps_count += 1
    ret, frame = cap.read()
    RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if fps_count >= 5:
        fps_count = 0
        fc.capture_all_faces(RGB_img)

    cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.imshow("SmartEye", frame)

    fps_time = time.time()
    if cv2.waitKey(frame_rate) == 27:
        break

cap.release()
cv2.destroyAllWindows()
