import os
import shutil
import cv2
import numpy as np
from PIL import Image
import imagehash

DIRETORIO_IMAGENS = r"/home/andressa/Documents/UFPEL/Projeto_Visao/código/inat_extra/Pleroma_mutabile"
DIRETORIO_LIXEIRA = os.path.join(DIRETORIO_IMAGENS, "lixeira_imagens")

LIMIAR_BORRAO = 20.0    # Quanto menor, mais borrada
LIMIAR_ESCURO = 30      # Média de pixels próxima de 0 é totalmente preto
LIMIAR_CLARO = 225      # Média de pixels próxima de 255 é totalmente branco


def setup_lixeira():
    if not os.path.exists(DIRETORIO_LIXEIRA):
        os.makedirs(DIRETORIO_LIXEIRA)

def mover_para_lixeira(caminho_arquivo, motivo):
    nome_arquivo = os.path.basename(caminho_arquivo)
    destino = os.path.join(DIRETORIO_LIXEIRA, f"{motivo}_{nome_arquivo}")
    shutil.move(caminho_arquivo, destino)
    print(f"[{motivo.upper()}] Movido: {nome_arquivo}")

def limpar_dataset():
    setup_lixeira()
    hashes_vistos = {} 
    
    arquivos = [f for f in os.listdir(DIRETORIO_IMAGENS) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Total de imagens encontradas: {len(arquivos)}")
    
    for arquivo in arquivos:
        caminho_completo = os.path.join(DIRETORIO_IMAGENS, arquivo)
        
        try:
            with Image.open(caminho_completo) as img:
                img.verify() 
        except (IOError, SyntaxError) as e:
            mover_para_lixeira(caminho_completo, "corrompida")
            continue

        img_cv = cv2.imread(caminho_completo)
        if img_cv is None:
            mover_para_lixeira(caminho_completo, "leitura_falhou")
            continue
            
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        media_pixels = np.mean(gray)
        if media_pixels < LIMIAR_ESCURO:
            mover_para_lixeira(caminho_completo, "escura")
            continue
        elif media_pixels > LIMIAR_CLARO:
            mover_para_lixeira(caminho_completo, "branca")
            continue
            
        variancia = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variancia < LIMIAR_BORRAO:
            mover_para_lixeira(caminho_completo, "borrada")
            continue
            
        hash_img = str(imagehash.phash(Image.open(caminho_completo)))
        if hash_img in hashes_vistos:
            mover_para_lixeira(caminho_completo, "duplicada")
            continue
        else:
            hashes_vistos[hash_img] = arquivo

    print("\nLimpeza concluída!")

if __name__ == "__main__":
    limpar_dataset()