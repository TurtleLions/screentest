import datetime
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import ImageGrab, Image, ImageTk
import tkinter as tk
img_height, img_width = 1440, 2560
n_channels = 4
transparent_img = np.zeros((img_height, img_width, n_channels))

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

# load the pre-trained YOLOv8n model
model = YOLO("best.pt")
while True:
    # start time to compute the fps
    start = datetime.datetime.now()
    screen = np.array(ImageGrab.grab(bbox=(0,0,2560,1440)))
    screen = screen[:, :, ::-1].copy()
    #cv2.imshow('Python Window', screen)
    transparent_img = np.zeros((img_height, img_width, n_channels))
    
    # run the YOLO model on the frame
    detections = model.predict(screen)
    result = detections[0]
    # loop over the detections
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        cv2.rectangle(transparent_img, (x1, y1) , (x2, y2), GREEN, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(screen, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    cv2.imwrite("hi.jpg", transparent_img)
    # show the frame to our screen
    #cv2.imshow("Python Window", transparent_img)
    #if cv2.waitKey(1) == ord("q"):
    #    break
#cv2.destroyAllWindows()