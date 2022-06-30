import face_recognition
import cv2
import os
from google. colab. patches import cv2_imshow

#image preprocessing
def img_resize(path):
    img = cv2.imread(path)
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h*width)
    return cv2.resize(img, (width, height))

#list to store the face encodings
train_enc = []
#list to store the names of person
train_names = []

#train the model
train_img = 'train'
for file in os.listdir(train_img):
    img = img_resize(train_img + '/' + file)
    img_enc = face_recognition. face_encodings(img)[0]
    train_enc. append(img_enc)
    train_names. append(file.split('.'[0]))

#test the model
test_img = 'test'
for file in os.listdir(test_img):
    img = img_resize(test_img + '/' + file)
    img_enc = face_recognition. face_encodings(img)[0]
    outputs = face_recognition. compare_face(train_enc, img_enc)

#results displayin
for i in range(len(outputs)):
    if outputs[i]:
        name = train_names[i]
        (top, right, bottom, left) = face_recognition.face_locations(img)[0]
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(img, name, (left+2, bottom+20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2_imshow(img)