import cv2
import numpy as np
import os
base_dir = './IMG/train/'
target_cnt = 400 # 수집할 사진 수
cnt = 0 # 사진 촬영 수

face_classifier = cv2.CascadeClassifier('./Cascades/haarcascade_frontalface_default.xml')

name = input("Insert User Name(Only Alphabet): ")
id = input("Insert User Id(Non-Duplicate number): ")
dir = os.path.join(base_dir, name+'_'+id)

if not os.path.exists(dir):
    os.mkdir(dir)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            file_name_path = os.path.join(dir, str(cnt) + '.jpg')
            cv2.imwrite(file_name_path, face) # 수집한 사진을 설정한 경로에 저장
            cv2.putText(frame, str(cnt), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cnt+=1
        else:
            if len(faces) == 0:
                msg = "no face"
            elif len(faces) > 1:
                msg = "too many face"
            cv2.putText(frame, msg, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        
        cv2.imshow('face record', frame)
        if cv2.waitKey(1) == 27 or cnt == target_cnt:
            break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Completed.")