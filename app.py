import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt

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

def find_target_faces():
    face_locations = fr.face_locations(target_image)

    if not face_locations:
        print("Nenhuma pessoa foi identificada.")
        return None

    target_faces = []

    for person in encode_faces('people/'):
        encoded_face, filename = person

        is_target_face = fr.compare_faces(encoded_face, target_encodings, tolerance=0.6)

        for i, location in enumerate(face_locations):
            if is_target_face[i]:
                label = os.path.splitext(filename)[0]
                target_faces.append((label, location))

    if not target_faces:
        print("Nenhuma pessoa foi identificada.")
        return None

    return target_faces

def create_frames(image, faces):
    for label, location in faces:
        top, right, bottom, left = location

        cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv.rectangle(image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
        cv.putText(image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

def render_images(target_faces):
    if target_faces:
        create_frames(target_image, target_faces)

        plt.imshow(target_image)
        plt.title('Pessoas identificadas')
        plt.axis('off')
        plt.show()

target_faces = find_target_faces()
render_images(target_faces)
