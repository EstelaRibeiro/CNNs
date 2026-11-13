"""
=============================================================================
AlexNet: O Marco Zero do Deep Learning Moderno (Krizhevsky et al., 2012)
=============================================================================

Contexto Histórico & O Problema:
--------------------------------
Até 2012, o reconhecimento de padrões em imagens dependia de atributos manuais
(SIFT, HOG) com classificadores rasos (SVMs). A AlexNet venceu a competição
ImageNet (ILSVRC 2012) com uma margem avassaladora de quase 10% de acurácia sobre
o segundo colocado, inaugurando a era moderna do Aprendizado Profundo.

Inovações Fundamentais:
---------------------------------
1. Treinamento Acelerado por GPUs:
   - Primeira grande rede treinada com código CUDA em duas GPUs NVIDIA GTX 580.
2. Função de Ativação ReLU (Rectified Linear Unit: f(x) = max(0, x)):
   - Substituiu a tangente hiperbólica e a sigmoide. Por não saturar para valores
     positivos, acelerou a convergência via SGD em até 6 vezes!
3. Regularização por Dropout (p=0.5):
   - "Desliga" aleatoriamente metade dos neurônios a cada passo, impedindo a co-adaptação
     excessiva e combatendo o overfitting nas camadas densas massivas.
4. Convoluções Iniciais com Kernels Grandes:
   - Filtros de 11x11 com stride 4 na entrada para resumir rapidamente a alta resolução espacial.
5. Max Pooling com Sobreposição (Overlapping Pooling):
   - Janelas de tamanho 3x3 com passo (stride) 2, diminuindo a degradação de detalhes espaciais.

Relevância para Raio-X de Tórax:
--------------------------------
Excelente como modelo de partida (*baseline*) didático para demonstrar
como uma CNN lê bordas brutas, costelas e contornos antes de sintetizar diagnósticos.
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Implementação Didática da AlexNet em PyTorch.
    
    Args:
        in_channels (int): Número de canais de entrada (3 para RGB, 1 para monocromático).
        num_classes (int): Quantidade de saídas da rede (1 para classificação binária).
        dropout (float): Probabilidade de descarte de conexões no classificador MLP.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, dropout: float = 0.5):
        super().__init__()

        # Extrator de Características Convolucionais
        self.features = nn.Sequential(
            # Camada 1: Filtro gigante 11x11 com stride 4 para captura ampla e redução rápida
            # Entrada: [B, in_channels, 224, 224] -> Saída: [B, 64, 55, 55]
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # MaxPool sobreposto (3x3, stride 2) -> Saída: [B, 64, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Camada 2: Filtro 5x5 para texturas intermediárias
            # Entrada: [B, 64, 27, 27] -> Saída: [B, 192, 27, 27] -> Pool -> [B, 192, 13, 13]
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Camadas 3, 4 e 5: Sequência profunda com filtros 3x3
            # Entrada: [B, 192, 13, 13] -> Saída: [B, 384, 13, 13]
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Entrada: [B, 384, 13, 13] -> Saída: [B, 256, 13, 13]
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Entrada: [B, 256, 13, 13] -> Saída: [B, 256, 13, 13] -> Pool -> [B, 256, 6, 6]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # AdaptiveAvgPool2d((6, 6)): Garante estabilidade da saída espacial em 6x6
        # mesmo se a imagem médica fornecida tiver resoluções maiores (ex: 384x384 ou 512x512)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Cabeçote Classificador (Multi-Layer Perceptron com Dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)  # Saída em logits para BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fluxo de tensores passo a passo:
        # Entrada: [B, C, H, W]
        x = self.features(x)            # -> [B, 256, H', W']
        x = self.avgpool(x)             # -> [B, 256, 6, 6]
        x = torch.flatten(x, 1)         # -> [B, 256 * 6 * 6 = 9216]
        x = self.classifier(x)          # -> [B, num_classes]
        return x


def alexnet(num_classes: int = 1, in_channels: int = 3, dropout: float = 0.5) -> AlexNet:
    """Função construtora para a AlexNet."""
    return AlexNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
