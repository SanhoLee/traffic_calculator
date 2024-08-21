from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # initialize model
source="video/traffic_video.avi"
# results = model.track(source=source, vid_stride=1, show=True, save=True)  # perform inference


# when you use opencv......
cap = cv2.VideoCapture(source)
# cap_x, cap_y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(f'width : {cap_x}, height : {cap_y}')

# pStart = (0,cap_y-10)
# pEnd = (cap_x-1, cap_y-10)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         results = model.track(frame, persist=True)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()
#         cv2.line(annotated_frame, pStart, pEnd, (0,0,255), 3)


#         xyxyList = results[0].boxes.xyxy.tolist()[0]
#         if int(xyxyList[0]) > int(cap_x/2) & int(xyxyList[1]) > cap_y-10:
#             print("counted a car running in the right side...")
            

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()




# another version...track line...
track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()