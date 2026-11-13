"""
=============================================================================
MobileNet: Eficiência Extrema (v1 - Howard et al., 2017 & v2 - Sandler et al., 2018)
=============================================================================

Contexto Histórico & O Problema:
--------------------------------
Modelos como VGG e ResNet exigiam dezenas de milhões de parâmetros e bilhões de
operações de ponto flutuante (FLOPs), tornando impossível rodar inferência em
dispositivos móveis, tablets de radiologia ou aparelhos médicos portáteis de Raio-X.
A família MobileNet nasceu para responder:
"Como manter alta acurácia reduzindo drasticamente o consumo computacional e de bateria?"

1. MobileNetV1: Convolução Separável em Profundidade (Depthwise Separable Conv)
-------------------------------------------------------------------------------
Uma convolução padrão 2D faz duas tarefas conjuntas e caras:
  1. Filtra pixels espacialmente (H x W)
  2. Combina os canais (C)

A MobileNetV1 separa isso em duas etapas simples:
  a) Depthwise Convolution:
     Aplica um filtro 3x3 para CADA canal separadamente (`groups = in_channels`).
     Custo: D_K * D_K * M * D_F * D_F
  b) Pointwise Convolution (1x1):
     Aplica convoluções 1x1 tradicionais para combinar os canais.
     Custo: M * N * D_F * D_F

Economia Matemática:
--------------------
Razão = (Custo Separável) / (Custo Padrão) = 1/N + 1/(D_K^2)
Para kernels 3x3 (D_K = 3) e N grande, a economia é de aproximadamente 8x a 9x menos operações!

2. MobileNetV2: Resíduos Invertidos e Gargalos Lineares (Inverted Residuals & Linear Bottlenecks)
------------------------------------------------------------------------------------------------
A MobileNetV2 introduziu dois conceitos revolucionários:
  a) Inverted Residuals (Gargalos Invertidos):
     Ao contrário da ResNet (Largo -> Estreito -> Largo), a MobileNetV2 conecta os
     atalhos residuais entre as representações ESTREITAS (economizando memória RAM).
     A expansão (fator 6x) ocorre internamente no bloco.
  b) Linear Bottleneck:
     A última camada 1x1 do bloco NÃO tem função de ativação ReLU.
     Motivo matemático: aplicar ReLU (que zera valores negativos) em espaços estreitos
     de baixa dimensão destrói informação útil de forma irreversível ("manifold collapse").

Relevância para Raio-X de Tórax:
--------------------------------
Ideal para aplicações de "Point-of-Care" (PoC), onde um algoritmo de triagem precisa
rodar em tempo real direto na máquina de raio-X sem necessidade de GPU em nuvem.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Bloco Convolucional Separável em Profundidade (MobileNetV1):
    1. Depthwise: Convolução 3x3 por canal (groups = in_channels).
    2. Pointwise: Convolução 1x1 combinando os canais.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            # 1. Depthwise 3x3
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # 2. Pointwise 1x1
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    Bloco Residual Invertido com Gargalo Linear (MobileNetV2):
    1. Expansão 1x1 (expande canais por expand_ratio, padrão 6x).
    2. Depthwise Conv 3x3.
    3. Projeção Pointwise 1x1 LINEAR (sem ReLU!).
    4. Conexão Residual (apenas se stride == 1 e in_channels == out_channels).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 6):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = in_channels * expand_ratio
        layers = []

        # 1. Expansão 1x1
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # ReLU6 limita o valor máximo em 6, facilitando quantização em bits inteiros (int8) para celulares
                nn.ReLU6(inplace=True)
            ])

        # 2. Depthwise 3x3
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # 3. Projeção Linear 1x1 (Sem ativação! Preserva o manifold de baixa dimensão)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNetV1(nn.Module):
    """
    Arquitetura MobileNetV1.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        # Configuração: (canais_saída, stride)
        cfg = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1)
        ]

        # Stem inicial (Convolução 3x3 padrão)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        layers = []
        curr_in = 32
        for out_c, s in cfg:
            layers.append(DepthwiseSeparableConv(curr_in, out_c, stride=s))
            curr_in = out_c

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(curr_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV2(nn.Module):
    """
    Arquitetura MobileNetV2.
    
    Args:
        in_channels (int): Canais da imagem de entrada.
        num_classes (int): Classes de saída.
        width_mult (float): Multiplicador de largura (alpha) para regular o tamanho do modelo.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, width_mult: float = 1.0):
        super().__init__()

        # Configuração dos estágios: (expand_ratio, out_channels, num_blocks, stride)
        inverted_residual_setting = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # Stem inicial
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        features = [
            nn.Conv2d(in_channels, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ]

        # Construção dos blocos residuais invertidos
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride=stride, expand_ratio=t)
                )
                input_channel = output_channel

        # Camada final
        features.extend([
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ])

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fluxo de tensores:
        # Entrada: [B, C, H, W]
        x = self.features(x)            # -> [B, 1280, H/32, W/32]
        x = self.avgpool(x)             # -> [B, 1280, 1, 1]
        x = torch.flatten(x, 1)         # -> [B, 1280]
        x = self.classifier(x)          # -> [B, num_classes] (logits)
        return x


def mobilenet_v1(num_classes: int = 1, in_channels: int = 3) -> MobileNetV1:
    """Instancia a MobileNetV1."""
    return MobileNetV1(in_channels=in_channels, num_classes=num_classes)

def mobilenet_v2(num_classes: int = 1, in_channels: int = 3) -> MobileNetV2:
    """Instancia a MobileNetV2."""
    return MobileNetV2(in_channels=in_channels, num_classes=num_classes)
