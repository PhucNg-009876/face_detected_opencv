import pickle
import cvzone
import cv2
import face_recognition
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage

# firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "",
    'storageBucket': "" # remove gs://
})
bucket = storage.bucket()


# Mở camera
cap = cv2.VideoCapture(0)

# Đặt chiều rộng và chiều dài cho frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Importing the mode images into a list as a numpy list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
print(len(imgModeList))


if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Tải ảnh background
background = cv2.imread('Resources/background.png')

# Đặt kích thước cho background (giả sử background có kích thước 1280x720)
background_height, background_width = 720, 1280
background = cv2.resize(background, (background_width, background_height))

# load encoding file
file = open("Encode.p","rb")
encodelistKnowWithIds = pickle.load(file)
file.close()
# extract data from file
encodelistKnow,studentIds =encodelistKnowWithIds
print(studentIds)

# state
modeType =0
couter =0
id=0
imgStudent=[]
# chạy camera
while True:
    # Đọc frame từ camera
    ret, frame = cap.read()

    # smaller img
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # BGR to RGB
    # take face form frame
    faceCurFrame = face_recognition.face_locations(imgS)
    # compare to encode img
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
    if not ret:
        print("Không thể nhận frame. Thoát...")
        break

    # Lấy kích thước của frame
    frame_height, frame_width, _ = frame.shape

    # Chèn frame vào background
    background[162:162+frame_height, 55:55+frame_width] = frame
    #chèn các ảnh trong folder modes
    background[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    # xử lí nhận diện
        # zip: using encodeCurFrame & faceCurFrame at the same time
    for encodeFace,faceLocation in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodelistKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnow,encodeFace)
        # matches [True, False, False] do khi encoding có 3 ảnh và ảnh đầu trùng với ảnh trong cam
        #print("matches",matches)
        #print("facedis",faceDis)
        # take index tại giá trị min <=> giá trị true
        matchIndex = np.argmin(faceDis)
        # check ảnh if true => detected
        if matches[matchIndex]:
            #print("face Detected")
           # print(studentIds[matchIndex])
            # draw a rectangle around the face
            y1,x2,y2,x1 = faceLocation
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            bbox = 55 +x1 ,162+y1,x2-x1,y2-y1
            background  =  cvzone.cornerRect(background,bbox,rt=0)
            id=studentIds[matchIndex]
            # if it detected ->active
            if couter ==0:
                couter =1
                # change img state
                modeType=1

    #if it detected and active -> show info
    if couter !=0:
        if couter ==1:
            # dowload data from database
            studentInfo = db.reference(f'Students/{id}').get()
            print("student info ",studentInfo)
            # get img from storage
            blob=bucket.get_blob(f'Images/{id}.png')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
            #update attendance
            ref = db.reference(f'Students/{id}')
            studentInfo['total_attendance'] +=1
            ref.child('total_attendance').set(studentInfo['total_attendance'])
        if 10<couter<=20:
            # đổi trạng thái detect
            modeType=2
            # chèn các ảnh trong folder modes
            background[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
        if couter <=10:
            # add text info
            cv2.putText(background,str(studentInfo['total_attendance']),(861,125),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            cv2.putText(background, str(studentInfo['major']), (1006, 550),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(background, str(id), (1006, 493),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(background, str(studentInfo['standing']), (910, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(background, str(studentInfo['year']), (1025, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(background, str(studentInfo['starting_year']), (1125, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            #add name
            (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            offset = (414 - w) // 2
            cv2.putText(background, str(studentInfo['name']), (808 + offset, 445),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
            # add student img ( phải đúng tỉ lệ 216x216)
            background[175:175 + 216, 909:909 + 216] = imgStudent

        couter +=1
        # reset trạng thái detect
        if couter >=20:
            couter =0
            modeType =0
            studentInfo =[]
            imgStudent =[]
            # chèn các ảnh trong folder modes
            background[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    # Hiển thị background với frame đã được chèn
    cv2.imshow('Camera with Background', background)
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

# còn thiếu
# code resize ảnh khi thêm ảnh mới
