"""
=============================================================================
Dataset para Imagens Médicas de Raio-X de Tórax (generator.py)
=============================================================================

Este módulo implementa a classe `ChestXRayDataset` derivada de `torch.utils.data.Dataset`,
responsável por:
1. Ler os exames de Raio-X a partir do disco.
2. Converter imagens médicas para o padrão RGB de 3 canais exigido pelas CNNs e ViTs.
3. Aplicar Data Augmentation clinicamente plausível (rotações e translações sutis).
4. Normalizar os tensores com a média e desvio padrão do ImageNet.
"""

import os
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXRayDataset(Dataset):
    """
    Dataset PyTorch para classificação binária de radiografias de tórax.
    
    Args:
        df: DataFrame do pandas com os metadados (colunas obrigatórias: 'saved_filename' e 'label').
        img_dir (str): Caminho da pasta contendo os arquivos de imagem PNG/JPEG.
        img_size (tuple): Resolução (altura, largura) para redimensionamento (ex: (224, 224) ou (384, 384)).
        augment (bool): Se True, aplica transformações aleatórias para enriquecer o treino e evitar overfitting.
    """
    def __init__(self, df, img_dir: str, img_size: Tuple[int, int] = (224, 224), augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment

        # Médias e desvios padrão oficiais do ImageNet (canais R, G, B)
        # Necessários para garantir que os pesos pré-treinados funcionem na mesma escala estatística
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        # Pipeline de transformações determinísticas básicas (para validação e teste)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Pipeline de Data Augmentation para treinamento:
        # ATENÇÃO CLÍNICA:
        # Em Raio-X, o flip horizontal deve ser usado com moderação (p=0.1 a 0.2)
        # porque o coração fica naturalmente à esquerda (dextrocardia é rara).
        # Flips verticais são estritamente proibidos (pacientes não ficam de cabeça para baixo).
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.05,
                contrast=0.05
            )
        ])

    def __len__(self) -> int:
        """Retorna o número total de radiografias no conjunto."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carrega e processa sob demanda a i-ésima imagem do dataset.
        
        Returns:
            image (torch.Tensor): Tensor [3, H, W] normalizado.
            label (torch.Tensor): Tensor [1] com valor float (0.0 para negativo, 1.0 para positivo).
        """
        row = self.df.iloc[idx]
        img_name = row['saved_filename']
        label_val = float(row['label'])
        label = torch.tensor([label_val], dtype=torch.float32)

        img_path = os.path.join(self.img_dir, img_name)

        # Abre a imagem e converte para 3 canais RGB
        # Mesmo sendo raio-X monocromático, a conversão para RGB replica o canal
        # em 3 dimensões, compatibilizando a entrada com modelos pré-treinados
        image = Image.open(img_path).convert('RGB')

        if self.augment:
            image = self.aug_transform(image)

        image = self.transform(image)

        return image, label
