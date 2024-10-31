import cv2
import torch
import json
import os
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')  # Modelo YOLO mais robusto

# Diretórios das imagens
img_dir_annotated = './data/imagens'
img_dir_test = './data/imagens_teste'

# Variáveis para controle de navegação
current_index = 0
use_annotations_images = False
selected_images = []  # Lista de imagens do conjunto de teste

# Dimensões para redimensionamento da imagem
IMG_WIDTH, IMG_HEIGHT = 600, 400


# Função para carregar imagens dos diretórios anotado e de teste
def load_images():
    global selected_images, current_index
    current_index = 0  # Resetar o índice ao carregar novas imagens

    if(use_annotations_images):
        # Carregar imagens anotadas do JSON
        with open('./data/annotations.json', 'r') as f:
            annotations = json.load(f)

        # Armazenar todas as imagens anotadas e de teste para detecção
        annotated_images = [{'file_name': img['file_name']} for img in annotations['images']]
        test_images = [{'file_name': f} for f in os.listdir(img_dir_test) if f.endswith(('.jpg', 'JPG', '.png', '.jpeg'))]

        # Concatenar as listas para detecção
        all_images = annotated_images + test_images

        # Realizar a detecção em todas as imagens
        for img_info in all_images:
            img_path = os.path.join(img_dir_annotated if img_info in annotated_images else img_dir_test,
                                    img_info['file_name'])
            if os.path.exists(img_path):
                process_image(img_path)  # Detecção nas imagens, mas exibirá apenas do conjunto de teste

    # Filtrar apenas as imagens do conjunto de teste para exibição
    selected_images = [{'file_name': f} for f in os.listdir(img_dir_test) if f.endswith(('.jpg', 'JPG', '.png', '.jpeg'))]
    show_image()


# Função para processar e exibir a imagem
def process_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)  # Realiza a detecção com YOLO

    # Renderizar as detecções sem rótulos de classe
    results_img = img.copy()  # Começa com uma cópia da imagem original
    for *box, conf, cls in results.xyxy[0]:  # Itera sobre cada detecção
        x1, y1, x2, y2 = map(int, box)  # Coordenadas da caixa delimitadora
        cv2.rectangle(results_img, (x1, y1), (x2, y2), (0, 255, 0), 30)  # Desenha apenas o retângulo

    # Redimensionar a imagem mantendo a proporção
    h, w = results_img.shape[:2]
    scale = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    resized_img = cv2.resize(results_img, (int(w * scale), int(h * scale)))

    # Converter a imagem para exibir no tkinter
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Exibir a imagem na interface
    img_label.config(image=img_tk)
    img_label.image = img_tk
    update_status()


# Função para exibir a imagem atual do conjunto de teste
def show_image():
    if selected_images:
        file_name = selected_images[current_index]['file_name']
        img_path = os.path.join(img_dir_test, file_name)  # Exibir apenas imagens de teste
        if os.path.exists(img_path):
            process_image(img_path)


# Atualizar o status da navegação
def update_status():
    status_label.config(text=f"{current_index + 1} / {len(selected_images)}")


# Função para mostrar a próxima imagem
def show_next_image():
    global current_index
    if current_index < len(selected_images) - 1:
        current_index += 1
        show_image()


# Função para mostrar a imagem anterior
def show_previous_image():
    global current_index
    if current_index > 0:
        current_index -= 1
        show_image()


# Configurar a interface Tkinter
root = tk.Tk()
root.title("Detecção de Lixo com YOLO")

# Frame para a imagem
img_frame = Frame(root)
img_frame.pack(pady=10)

# Label para exibir a imagem
img_label = Label(img_frame)
img_label.pack()

# Label para mostrar o status (índice da imagem)
status_label = Label(root, text="")
status_label.pack()

# Frame para botões de navegação
button_frame = Frame(root)
button_frame.pack(pady=10)

prev_button = Button(button_frame, text="Anterior", command=show_previous_image)
prev_button.grid(row=0, column=0, padx=5)

next_button = Button(button_frame, text="Próximo", command=show_next_image)
next_button.grid(row=0, column=1, padx=5)

# Carregar as primeiras imagens ao iniciar
load_images()

root.mainloop()