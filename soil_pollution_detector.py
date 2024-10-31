import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk
from functools import partial

# ======================================================
# Função para carregar todas as imagens e máscaras do diretório
# ======================================================
def carregar_imagens_e_mascaras(diretorio_imagens, diretorio_mascaras):
    imagens = []
    mascaras = []
    nomes_imagens = []

    arquivos_imagens = sorted(os.listdir(diretorio_imagens))
    arquivos_mascaras = sorted(os.listdir(diretorio_mascaras))

    for img_file, mask_file in zip(arquivos_imagens, arquivos_mascaras):
        # Carregar imagem
        caminho_imagem = os.path.join(diretorio_imagens, img_file)
        imagem_original = cv2.imread(caminho_imagem)

        # Carregar máscara
        caminho_mascara = os.path.join(diretorio_mascaras, mask_file)
        mascara = cv2.imread(caminho_mascara, cv2.IMREAD_GRAYSCALE)

        # Padronizar a máscara (binária: 0 e 255)
        _, mascara_binaria = cv2.threshold(mascara, 0, 255, cv2.THRESH_BINARY)

        # Adicionar imagem, máscara e nome do arquivo à lista
        imagens.append(imagem_original)
        mascaras.append(mascara_binaria)
        nomes_imagens.append(os.path.splitext(img_file)[0])  # Nome da imagem sem a extensão

    return imagens, mascaras, nomes_imagens

# ======================================================
# Função principal de processamento
# ======================================================
def detectar_poluicao(imagem, mascara):
    # Aplicar a máscara na imagem original para destacar a poluição
    resultado = cv2.bitwise_and(imagem, imagem, mask=mascara)

    # Detectar contornos nas áreas poluídas
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copiar a imagem original para desenhar os contornos
    imagem_com_contornos = imagem.copy()
    cv2.drawContours(imagem_com_contornos, contornos, -1, (0, 0, 255), 20)  # Desenhar contornos em vermelho

    return imagem_com_contornos, resultado

# ======================================================
# Função para redimensionar imagens mantendo proporção
# ======================================================
def redimensionar_imagem2(imagem, largura_max=1000, altura_max=500):
    altura_original, largura_original = imagem.shape[:2]

    # Calcula a proporção baseada nos limites de largura e altura
    fator_escala = min(largura_max / largura_original, altura_max / altura_original)

    # Calcula novas dimensões baseadas na proporção
    nova_largura = int(largura_original * fator_escala)
    nova_altura = int(altura_original * fator_escala)

    # Redimensiona a imagem
    imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura))
    return imagem_redimensionada

# ======================================================
# Função para converter imagem OpenCV para Tkinter
# ======================================================
def imagem_para_tk(imagem_cv):
    imagem_rgb = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2RGB)
    imagem_pil = Image.fromarray(imagem_rgb)
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    return imagem_tk

# ======================================================
# Função para exibir slides de uma imagem selecionada
# ======================================================
class SlideShowApp:
    def __init__(self, root, imagens):
        self.root = root
        self.root.title("Exibição de Poluição - Slides")
        self.imagens = imagens
        self.indice_atual = 0

        # Label para exibir a imagem
        self.label_imagem = tk.Label(root)
        self.label_imagem.pack()

        # Frame para organizar os botões e o índice
        self.frame_controles = tk.Frame(root)
        self.frame_controles.pack()

        # Botão "Anterior"
        self.btn_anterior = tk.Button(self.frame_controles, text="Anterior", command=self.anterior_slide)
        self.btn_anterior.pack(side=tk.LEFT)

        # Label para exibir o índice (ex: "1/3")
        self.label_indice = tk.Label(self.frame_controles, text="")
        self.label_indice.pack(side=tk.LEFT)

        # Botão "Próximo"
        self.btn_proximo = tk.Button(self.frame_controles, text="Próximo", command=self.proximo_slide)
        self.btn_proximo.pack(side=tk.LEFT)

        # Exibir o primeiro slide
        self.exibir_slide()

    def exibir_slide(self):
        # Atualizar a imagem no label
        imagem_tk = self.imagens[self.indice_atual]
        self.label_imagem.config(image=imagem_tk)
        self.label_imagem.image = imagem_tk  # Necessário para manter a referência

        # Atualizar o texto do índice
        self.label_indice.config(text=f"{self.indice_atual + 1}/{len(self.imagens)}")

    def anterior_slide(self):
        if self.indice_atual > 0:
            self.indice_atual -= 1
        self.exibir_slide()

    def proximo_slide(self):
        if self.indice_atual < len(self.imagens) - 1:
            self.indice_atual += 1
        self.exibir_slide()

# ======================================================
# Função para criar o menu de seleção de imagens com paginação
# ======================================================
class MenuApp:
    def __init__(self, root, imagens, mascaras, nomes_imagens, imagens_por_pagina=8):
        self.root = root
        self.root.title("Menu de Imagens de Poluição")
        self.imagens = imagens
        self.mascaras = mascaras
        self.nomes_imagens = nomes_imagens
        self.imagens_por_pagina = imagens_por_pagina
        self.pagina_atual = 0
        self.total_paginas = (len(imagens) - 1) // imagens_por_pagina + 1

        # Criar um título para o menu
        self.label_titulo = tk.Label(root, text="Resultados das Imagens", font=("Arial", 16, "bold"))
        self.label_titulo.pack(pady=10)

        # Frame para organizar o menu
        self.frame_menu = tk.Frame(root)
        self.frame_menu.pack()

        # Frame para os botões de navegação
        self.frame_navegacao = tk.Frame(root)
        self.frame_navegacao.pack(pady=10)

        # Botão "Página Anterior"
        self.btn_anterior = tk.Button(self.frame_navegacao, text="Página Anterior", command=self.pagina_anterior)
        self.btn_anterior.pack(side=tk.LEFT)

        # Label de página
        self.label_pagina = tk.Label(self.frame_navegacao, text="")
        self.label_pagina.pack(side=tk.LEFT)

        # Botão "Próxima Página"
        self.btn_proximo = tk.Button(self.frame_navegacao, text="Próxima Página", command=self.proxima_pagina)
        self.btn_proximo.pack(side=tk.LEFT)

        # Exibir a primeira página
        self.exibir_pagina()

    def exibir_pagina(self):
        # Limpar o conteúdo atual do frame de menu
        for widget in self.frame_menu.winfo_children():
            widget.destroy()

        # Calcular o índice inicial e final das imagens para a página atual
        inicio = self.pagina_atual * self.imagens_por_pagina
        fim = min(inicio + self.imagens_por_pagina, len(self.imagens))

        # Atualizar o texto do índice de página
        self.label_pagina.config(text=f"Página {self.pagina_atual + 1} de {self.total_paginas}")

        # Criar botões de seleção para cada imagem na página atual
        for i, (imagem, nome_imagem) in enumerate(zip(self.imagens[inicio:fim], self.nomes_imagens[inicio:fim])):
            # Criar o frame para conter a imagem e o nome
            frame_imagem = tk.Frame(self.frame_menu)
            frame_imagem.grid(row=i // 4, column=i % 4, padx=10, pady=10)

            # Label para o nome da imagem
            label_nome = tk.Label(frame_imagem, text=nome_imagem, font=("Arial", 10, "bold"), fg="blue")
            label_nome.pack()

            # Miniatura da imagem
            imagem_tk = imagem_para_tk(redimensionar_imagem2(imagem, largura_max=150, altura_max=150))
            btn = tk.Button(frame_imagem, image=imagem_tk, command=partial(self.abrir_slides, inicio + i))
            btn.image = imagem_tk  # Necessário para manter a referência
            btn.pack()

    def proxima_pagina(self):
        if self.pagina_atual < self.total_paginas - 1:
            self.pagina_atual += 1
            self.exibir_pagina()

    def pagina_anterior(self):
        if self.pagina_atual > 0:
            self.pagina_atual -= 1
            self.exibir_pagina()

    def abrir_slides(self, indice):
        # Criar uma nova janela para os slides da imagem selecionada
        top = tk.Toplevel(self.root)

        # Processar a imagem selecionada
        imagem_com_contornos, resultado_poluicao = detectar_poluicao(self.imagens[indice], self.mascaras[indice])

        # Redimensionar as imagens para exibição no slide
        imagem_original_redimensionada = redimensionar_imagem2(self.imagens[indice])
        imagem_com_contornos_redimensionada = redimensionar_imagem2(imagem_com_contornos)
        resultado_poluicao_redimensionada = redimensionar_imagem2(resultado_poluicao)

        # Converter as imagens redimensionadas para o formato Tkinter
        imagens_tk = [
            imagem_para_tk(imagem_original_redimensionada),
            imagem_para_tk(imagem_com_contornos_redimensionada),
            imagem_para_tk(resultado_poluicao_redimensionada)
        ]

        # Exibir as imagens em formato de slides
        SlideShowApp(top, imagens_tk)

# ======================================================
# Script principal
# ======================================================
if __name__ == "__main__":
    # Definir diretórios
    diretorio_imagens = './data/imagens/batch_1'
    diretorio_mascaras = './data/mascaras/batch_1'

    # Carregar todas as imagens, máscaras e nomes
    imagens, mascaras, nomes_imagens = carregar_imagens_e_mascaras(diretorio_imagens, diretorio_mascaras)

    # Criar a janela principal do menu
    root = tk.Tk()

    # Criar o menu de imagens com paginação
    MenuApp(root, imagens, mascaras, nomes_imagens, imagens_por_pagina=8)

    # Iniciar a interface gráfica
    root.mainloop()