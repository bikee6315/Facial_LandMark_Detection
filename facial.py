'''
Detecting facial landmarks is a two step process:
Step 1:  Localize the face in the image.
Step 2: Detect the key facial structures on the face ROI(Region of Image).

The facial landmark detector included in the dlib library is an implementation of the One Millisecond
Face Alignment with an Ensemble of Regression Trees. 

The pre-trained facial landmark detector inside the dlib library is used to estimate the location
of 68 (x, y)-coordinates that map to facial structures on the face.

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])
'''
import cv2
import numpy as np
import dlib

#We define a function "rectangle to bounding box". It accepts an argument "rectangle" which is assumed to be
#bounding box rectangle produced by a dlib detector.
def rectangle_to_boundbox(rectangle):
    #take a bounding detected by dlib and converting it into rectangle of format(x,y,w,h)
    x=rectangle.left()
    y=rectangle.top()
    w=rectangle.right()-x
    h=rectangle.bottom()-y
    #return a tuple of (x,y,w,h)
    return(x,y,w,h)


#The dlib face landmark detector will return a shape object containing the 68 (x, y)-coordinates
#of the facial landmark regions. Using the shape_to_array  function, we can convert this object to a
#NumPy array,
def shape_to_array(shape,dtype="int"):
    #initialize list of (x,y) coordinates
    coordinates=np.zeros((68,2),dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0,68):
        coordinates[i]=(shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates


PREDICTOR_PATH="shape_predictor_68_face_landmarks.dat"

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
predictor=dlib.shape_predictor(PREDICTOR_PATH)
detector=dlib.get_frontal_face_detector()

cap=cv2.VideoCapture(0)
while 1:
    #capture frame by frame
    ret,frame=cap.read()

    #Changing the frame in grayscale
    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    rects=detector(grayscale,1)
    
    # loop over the face detections
    for (i,rect) in enumerate(rects):

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape=predictor(grayscale,rect)
        shape=shape_to_array(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x,y,w,h)=rectangle_to_boundbox(rect)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        # show the face number
        cv2.putText(frame,"Face{}".format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255))

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw circle on the image
        for(x,y) in shape:
            cv2.circle(frame,(x,y),1,(0,0,255),-1)
    

    # Display the resulting frame
    cv2.imshow("Image",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break;
    
#Releasing the capture
cap.release()
cv2.destroyAllWindows()





    
