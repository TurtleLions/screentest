import datetime
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import ImageGrab, Image, ImageTk
import tkinter as tk


img_height, img_width = 1440, 2560
n_channels = 4
transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)

from tkinter import Tk, Canvas, PhotoImage, NW

#244L x 245H
#0,1194

root = Tk()

root.attributes('-transparentcolor','white')
root.attributes('-fullscreen',True)
root.attributes('-topmost', 'true')
# Canvas
canvas = Canvas(root, width=2560, height=1440)
canvas.pack()

# Image
img = ImageTk.PhotoImage(image=Image.fromarray(transparent_img))

# Positioning the Image inside the canvas
canvas.create_image(0, 0, anchor=NW, image=img)


# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

# load the pre-trained YOLOv8n model
model = YOLO("best.pt")
def update():
    global photoim
    # start time to compute the fps
    start = datetime.datetime.now()
    #screen = np.array(ImageGrab.grab(bbox=(0,0,2560,1440)))
    #screen = screen[:, :, ::-1].copy()
    screenshot = ImageGrab.grab(bbox=(0,1194,244,1439))
    screenshot.save("ss.png")
    #cv2.imshow('Python Window', screen)
    transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
    # run the YOLO model on the frame
    detections = model.predict("ss.png")
    result = detections[0]
    # loop over the detections
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        cv2.rectangle(transparent_img, (x1, y1+1194) , (x2, y2+1194), GREEN, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
    cv2.imwrite("hi.jpg", transparent_img)
    # show the frame to our screen
    img= Image.open("hi.jpg")
    np_img = np.array(img)
    imagemask = cv2.inRange(np_img, (0,0,0), (50,50,50))
    photoim =  ImageTk.PhotoImage(image=Image.fromarray(imagemask))
    #photoim = ImageTk.PhotoImage(Image.open("hi.jpg"))
    canvas.create_image(0, 0, anchor=NW, image=photoim)
    canvas.update()
    root.attributes('-topmost', 'true')
    root.after(500,update)
    
#cv2.destroyAllWindows()
root.after(500,update)
root.mainloop()