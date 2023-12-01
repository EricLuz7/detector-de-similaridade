import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt

# Abre uma janela de diálogo para o usuário escolher a imagem alvo
Tk().withdraw()
load_image = askopenfilename()

# Carrega a imagem alvo e suas codificações faciais
target_image = fr.load_image_file(load_image)
target_encodings = fr.face_encodings(target_image)

# Função para codificar as faces presentes em uma pasta
def encode_faces(folder):
    list_people_encoding = []

    for filename in os.listdir(folder):
        known_image = fr.load_image_file(os.path.join(folder, filename))
        known_encodings = fr.face_encodings(known_image)

        if known_encodings:
            # Armazena a codificação da primeira face encontrada na imagem
            list_people_encoding.append((known_encodings[0], filename))

    return list_people_encoding

# Função para encontrar as faces correspondentes na imagem alvo
def find_target_faces():
    face_locations = fr.face_locations(target_image)

    if not face_locations:
        print("A pessoa não foi identificada na imagem.")
        return None

    target_faces = []

    # Itera sobre as faces codificadas das imagens conhecidas
    for person in encode_faces('people/'):
        encoded_face, filename = person

        # Compara as faces da imagem alvo com as faces conhecidas
        is_target_face = fr.compare_faces(encoded_face, target_encodings, tolerance=0.6)

        # Adiciona as faces identificadas à lista target_faces
        for i, location in enumerate(face_locations):
            if is_target_face[i]:
                label = os.path.splitext(filename)[0]
                target_faces.append((label, location))

    return target_faces

# Função para criar frames ao redor das faces identificadas na imagem alvo
def create_frames(image, faces):
    for label, location in faces:
        top, right, bottom, left = location

        cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv.rectangle(image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
        cv.putText(image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

# Função para renderizar a imagem alvo com os frames ao redor das faces identificadas
def render_images(target_faces):
    if target_faces:
        # Se faces foram identificadas, cria os frames
        create_frames(target_image, target_faces)
        title = 'Pessoas identificadas'
    else:
        # Se nenhuma face foi identificada, mostra mensagem no console
        title = 'A pessoa não foi identificada'
        print(title)

    # Mostra a imagem alvo com os frames
    plt.imshow(target_image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Chama as funções principais
target_faces = find_target_faces()
render_images(target_faces)
