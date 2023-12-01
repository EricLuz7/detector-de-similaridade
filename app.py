import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

Tk().withdraw()
load_image = askopenfilename()

target_image = fr.load_image_file(load_image)
target_encodings = fr.face_encodings(target_image)

def encode_faces(folder):
    list_people_encoding = []

    for filename in os.listdir(folder):
        known_image = fr.load_image_file(os.path.join(folder, filename))
        known_encodings = fr.face_encodings(known_image)

        if known_encodings:
            list_people_encoding.append((known_encodings[0], filename))

    return list_people_encoding

def find_target_face():
    face_locations = fr.face_locations(target_image)

    if not face_locations:
        print("No face found in the target image.")
        return

    for person in encode_faces('people/'):
        encoded_face, filename = person

        is_target_face = fr.compare_faces(encoded_face, target_encodings, tolerance=0.6)

        for i, location in enumerate(face_locations):
            if is_target_face[i]:
                label = filename
                create_frame(location, label)

def create_frame(location, label):
    top, right, bottom, left = location

    cv.rectangle(target_image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv.rectangle(target_image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
    cv.putText(target_image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

def render_image():
    rgb_img = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
    cv.imshow('Face Recognition', rgb_img)
    cv.waitKey(0)

find_target_face()
render_image()