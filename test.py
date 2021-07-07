
import cv2

cap = cv2.VideoCapture(0)
#CAP_PROP_FRAME_WIDTH =3, //!< Width of the frames in the video stream.
#CAP_PROP_FRAME_HEIGHT =4, //!< Height of the frames in the video stream.
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

#make_720p()
#change_res(1280, 720)
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while(True):

    ret, frame = cap.read()
    #frame75 = rescale_frame(frame, percent=75)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   

    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()