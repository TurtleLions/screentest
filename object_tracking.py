import datetime
import easyocr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import ImageGrab, Image, ImageTk
import tkinter as tk
import pyautogui

mapoffsetx = 800
mapoffsety = 0
mapsize = 245

reader = easyocr.Reader(['en'], detector='dbnet18', gpu = True)

game = []

img_height, img_width = 1440, 2560
n_channels = 4
transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)

from tkinter import Tk, Canvas, PhotoImage, NW

#244L x 245H
#0,1194

ingame=False

root = Tk()

root.attributes('-transparentcolor','white')
root.attributes('-fullscreen',True)
root.attributes('-topmost', 'true')
# Canvas
canvas = Canvas(root, width=2560, height=1440)
canvas.pack()

# Image
img = ImageTk.PhotoImage(image=Image.fromarray(transparent_img))

# whiteimg = np.zeros([1440,2560,3],dtype=np.uint8)
# whiteimg.fill(255)
# whitephotoim =  ImageTk.PhotoImage(image=Image.fromarray(whiteimg))


# blueboxwhiteimg = np.zeros([1440,2560,3],dtype=np.uint8)
# blueboxwhiteimg.fill(255)
# cv2.rectangle(blueboxwhiteimg,(mapoffsetx,mapoffsety),(mapoffsetx+mapsize,mapoffsety+mapsize),(255,0,0),2)
# blueboxwhitephotoim =  ImageTk.PhotoImage(image=Image.fromarray(blueboxwhiteimg))

# Positioning the Image inside the canvas
canvas.create_image(0, 0, anchor=NW, image=img)


# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# load the pre-trained YOLOv8n model
model = YOLO("best.pt")

x = 0
def update():
  global ingame
  global x
  global photoim
  print(ingame)
  transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
  if ingame==False:
    cv2.imwrite("overlayimg.jpg", transparent_img)
    img= Image.open("overlayimg.jpg")
    np_img = np.array(img)
    imagemask = cv2.inRange(np_img, (0,0,0), (50,50,50))
    np_img[imagemask>0]=[255,255,255]
    photoim =  ImageTk.PhotoImage(image=Image.fromarray(np_img))
    canvas.create_image(0, 0, anchor=NW, image=photoim)
    canvas.update()
    root.attributes('-topmost', 'true')
    objectlocation = pyautogui.locateCenterOnScreen('randomdice.png', confidence = 0.7)#If the file is not a png file it will not work
    print(objectlocation)
    if str(objectlocation)!='None':
      if(objectlocation.x>700 and objectlocation.x<800 and objectlocation.y>400 and objectlocation.y<500):
        screenshot = ImageGrab.grab(bbox=(800,405,1300,470))
        screenshot.save("ocr.png")
        result = reader.readtext("ocr.png",batch_size = 1)
        for text in result:
          game.append(text[1])
        ingame=True
  if ingame==True:
    swordslocation = pyautogui.locateCenterOnScreen('endswords.png', confidence = 0.5)
    settingslocation = pyautogui.locateCenterOnScreen('settings.png', confidence = 0.5)
    if str(swordslocation)!='None':
      if(swordslocation.x>2320 and swordslocation.x<2360 and swordslocation.y>30 and swordslocation.y<60):
        ingame=False
    if str(settingslocation)!='None':
      if(settingslocation.x>20 and settingslocation.x<65 and settingslocation.y>870 and settingslocation.y<920):
        ingame=False
    # start time to compute the fps
    start = datetime.datetime.now()
    screenshot = ImageGrab.grab(bbox=(0,1194,244,1439))
    screenshot.save("ss.png")
    
    cv2.rectangle(transparent_img,(mapoffsetx,mapoffsety),(mapoffsetx+mapsize,mapoffsety+mapsize),(255,0,0),2)
    cv2.putText(transparent_img,game[0]+ ' '+ game[1], (mapoffsetx-100,mapoffsety+50),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    # run the YOLO model on the frame
    detections = model.predict("ss.png",stream=True)
    # loop over the detections
    for result in detections:
      boxes = result.boxes
      masks = result.masks
      probs = result.probs
      boxarray = boxes.cpu().xyxy.numpy()
      clsarray = boxes.cpu().cls.numpy()
      #loop over all boxes and write rectangles over them
      for index in range(0, len(boxarray)):
        x1, y1, x2, y2 = [
          round(x) for x in boxarray[index].tolist()
        ]
        if(clsarray[index]==0): 
          cv2.rectangle(transparent_img, (x1+mapoffsetx, y1+mapoffsety) , (x2+mapoffsetx, y2+mapoffsety), RED, 2)
        if(clsarray[index]==1):
          cv2.rectangle(transparent_img, (x1+mapoffsetx, y1+mapoffsety) , (x2+mapoffsetx, y2+mapoffsety), GREEN, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
    cv2.imwrite("overlayimg.jpg", transparent_img)
    # show the frame to our screen
    img= Image.open("overlayimg.jpg")
    np_img = np.array(img)
    imagemask = cv2.inRange(np_img, (0,0,0), (50,50,50))
    np_img[imagemask>0]=[255,255,255]
    photoim =  ImageTk.PhotoImage(image=Image.fromarray(np_img))
    canvas.create_image(0, 0, anchor=NW, image=photoim)
    canvas.update()
    root.attributes('-topmost', 'true')
  root.after(1,update)
    
#cv2.destroyAllWindows()
root.after(1,update)
root.mainloop()