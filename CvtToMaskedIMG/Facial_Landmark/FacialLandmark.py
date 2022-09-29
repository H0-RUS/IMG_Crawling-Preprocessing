from typing import OrderedDict
from imutils import face_utils
import dlib
import argparse
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

faceCascade = cv2.CascadeClassifier('IMG_Preprocessing\ConvertToMasked\haarcascade_frontalface_default.xml')
predictor =  dlib.shape_predictor(args["shape_predictor"])
# PATH: IMG_Preprocessing\ConvertToMasked\shape_predictor_68_face_landmarks.dat

FACIAL_LANDMARKS_PNTS = OrderedDict([
    ("nose", (27, 35)),
    ("borderOfMask", (1, 16))
])

def visualize_facial_landmarks(img, shape, colors=None, alpha=0.7):
    overlay = img.copy()
    output = img.copy()
    
    if colors is None:
        colors = [(252, 3, 152), (252, 3, 152)]
        
    for (i, name) in enumerate(FACIAL_LANDMARKS_PNTS.keys()):
        j, k = FACIAL_LANDMARKS_PNTS[name] # pointnum의 min, max를 각각 j, k에 저장
        points = shape[j:k]
        
        if name == "jaw":
            for l in range(1, len(points)):
                ptX, ptY = tuple(points[l-1]), tuple(points[l])
                cv2.line(overlay, ptX, ptY, colors[i], 3)
        else:
            hull = cv2.convexHull(points)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
        
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output
                
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            
            # <IMG_DATA 수집할 시>
            # file_name_path = os.path.join(dir, str(cnt) + '.jpg')
            # cv2.imwrite(file_name_path, face) # 수집한 사진을 설정한 경로에 저장
            # cv2.putText(frame, str(cnt), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # cnt+=1
        else:
            if len(faces) == 0:
                msg = "no face"
            elif len(faces) > 1:
                msg = "too many face"
            cv2.putText(frame, msg, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        
        cv2.imshow('face record', frame)
        if cv2.waitKey(1) == 27:
            break