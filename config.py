"""
=============================================================================
Arquivo de Configuração Global e Hiperparâmetros (config.py)
=============================================================================

Centraliza todos os caminhos, hiperparâmetros de treinamento e opções de
arquiteturas para facilitar a reprodução e experimentação em sala de aula.
"""

import os
from pathlib import Path

# =============================================================================
# 1. DIRETÓRIOS E ARQUIVOS DE DADOS (Imagens Médicas de Raio-X)
# =============================================================================
DATASET_DIR = r'C:\Users\estela.ribeiro\JupyterNotebooks\Projeto_Bayer_HP\bayer_hp_gerar_dataset\outputs\hp\xray_pngs'
OUTPUT_DIR = r'C:\Users\estela.ribeiro\JupyterNotebooks\estudando_cnns\outputs'
CSV_DIR = r'C:\Users\estela.ribeiro\JupyterNotebooks\Projeto_Bayer_HP\bayer_hp_gerar_dataset\outputs\hp\metadata_final_xray_new.csv'

# =============================================================================
# 2. ESCOLHA DA ARQUITETURA E MODO DE CRIAÇÃO
# =============================================================================
# Opções disponíveis para MODEL_NAME:
#  - 'alexnet'          (2012: Ruptura com GPU + ReLU + Dropout)
#  - 'vgg11', 'vgg16'   (2014: Pilhas profundas e homogêneas de filtros 3x3)
#  - 'resnet18', 'resnet50' (2015: Conexões Residuais e Fim da Degradação)
#  - 'densenet121'      (2017: Reuso Massivo de Features via Concatenação)
#  - 'inception'        (2014: Processamento Paralelo Multi-Escala + Convs 1x1)
#  - 'mobilenet_v1', 'mobilenet_v2' (2017-18: Convolução Separável e Resíduo Invertido)
#  - 'efficientnet_b0'  (2019: Escalonamento Composto e MBConv com SE)
#  - 'vit_tiny'         (2020: Quebra de paradigma com Atenção pura em Patches)
#  - 'convnext_tiny'    (2022: Convolução modernizada para os anos 2020)
MODEL_NAME = 'resnet18'

# Modo de Instanciação:
# - USE_TIMM = False: Usa a classe didática implementada em `models/<modelo>.py` (ideal para entender por dentro).
# - USE_TIMM = True:  Usa a biblioteca `timm` (ideal para usar pesos pré-treinados do ImageNet).
USE_TIMM = False

# Se True e USE_TIMM=True, baixa os pesos pré-treinados no ImageNet:
PRETRAINED = False

# =============================================================================
# 3. HIPERPARÂMETROS DE IMAGEM E TREINAMENTO
# =============================================================================
RANDOM_SEED = 13
IN_CHANNELS = 3       # 3 para RGB
NUM_CLASSES = 1       # 1 para Classificação Binária (Normal vs Doença) com BCEWithLogitsLoss
IMG_SIZE = (224, 224) # Resolução padrão (224x224 para ViT/AlexNet/ResNet, ou 384x384 para alta resolução)
BATCH_SIZE = 8        # Tamanho do lote de imagens
LR = 1e-4             # Taxa de aprendizado inicial (Learning Rate)
EPOCHS = 20           # Número total de épocas de treinamento
N_SPLITS = 5          # Número de folds para validação cruzada

# =============================================================================
# 4. CONFIGURAÇÕES INTERNAS DE ESTÁGIOS DA VGG (Retrocompatibilidade)
# =============================================================================
VGG_CONFIGS = {
    "vgg11": [(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)],
    "vgg13": [(64, 2), (128, 2), (256, 2), (512, 2), (512, 2)],
    "vgg16": [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)],
    "vgg19": [(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)],
}
