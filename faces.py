import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizer/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()} #reverse the values



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

        _id, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<85:
            #print(_id)
            #print(labels[_id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[_id]
            color =(255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


        img_item ="7.png"
        cv2.imwrite(img_item, roi_color)

        color=(0,0,255) #BGR 0-255
        stroke =2
        endX = x + w #end coordinates of x
        endY = y + h #end coord of y
        cv2.rectangle(frame,(x,y),(endX , endY), color, stroke)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

