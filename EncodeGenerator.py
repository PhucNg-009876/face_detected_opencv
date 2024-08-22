import cv2
import face_recognition
import pickle # convert to byte for encoding
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


# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []
studentIds =[]
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    #get id mane and remove ".png"
    studentIds.append(os.path.splitext(path)[0])
    # upload to database
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print(studentIds)

def findEncoding(imgList):
    encodeList =[]
    for img in imgList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodelistKnow = findEncoding(imgList)
encodelistKnowWithIds =[encodelistKnow,studentIds]

# save in a file
file = open("Encode.p","wb")
pickle.dump(encodelistKnowWithIds,file)
file.close()