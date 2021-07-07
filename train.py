import cv2 #for face detection
import os
from PIL import Image #to get that image
import numpy as np #to covert images to array
import pickle #to convert to bytes and store on disk

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() #using cv2 recognizer(local binary patter histogram)

y_labels=[]
x_train = []
current_id =0
label_ids = {}


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #directory where my train.py is located
img_DIR = os.path.join(BASE_DIR,"images")

for root, dirs, files in os.walk(img_DIR): #just to get images from img directory
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file) #find path
            label = os.path.basename(root).replace(" ","-").lower() #label
            #print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            
            _id = label_ids[label]
            #print(label_ids)

            pil_image = Image.open(path).convert("L") # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image,"uint8") # convert image to numpy array
            #print(image_array)

            faces = face_cascade.detectMultiScale(image_array , scaleFactor=1.5, minNeighbors=5)

            for(x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi) #verify image,turn it into numpy array,gray
                y_labels.append(_id) #for id of pictures

with open("pickles/face-labels.pickle","wb") as f: #wb write bytes
    pickle.dump(label_ids, f)#just to dump those label ids in a file
           

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizer/face-trainner.yml")

            