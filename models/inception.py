"""
=============================================================================
Inception / GoogLeNet: Multi-Scale Convolutions (Szegedy et al., Google - 2014)
=============================================================================

Contexto Histórico & O Problema:
--------------------------------
Vencedora da competição ImageNet (ILSVRC 2014).
O dilema clássico enfrentado pelos projetistas de redes convolucionais:
"Qual tamanho de kernel escolher para a próxima camada? 1x1, 3x3 ou 5x5?"
- Kernels 1x1 capturam detalhes locais e correlações entre canais.
- Kernels 3x3 capturam padrões espaciais médios.
- Kernels 5x5 capturam estruturas globais amplas.

Em imagens médicas como Raio-X de tórax, isso é crítico:
- Uma calcificação ou nódulo minúsculo requer análise focal fina (3x3 ou 1x1).
- A cardiomegalia (aumento do coração) ou um derrame pleural requer visão macro (5x5 ou maior).

A Solução Inception:
--------------------
"Não escolha: execute TODOS em paralelo e concatene as saídas!"

O Truque Mágico: Convoluções 1x1 como Bottlenecks
-------------------------------------------------
Executar uma convolução 5x5 diretamente sobre 256 canais é absurdamente custoso.
O bloco Inception introduziu convoluções 1x1 ANTES das convoluções 3x3 e 5x5.
Essas convs 1x1 reduzem os canais (ex: 256 -> 32), permitindo que os filtros 5x5
operem com um custo computacional muito reduzido!

4 Ramos Paralelos em cada Bloco Inception:
------------------------------------------
1. Ramo 1: Convolução 1x1 direta.
2. Ramo 2: Redução 1x1 -> Convolução 3x3.
3. Ramo 3: Redução 1x1 -> Duas convoluções 3x3 consecutivas (equivalente a 5x5, estilo Inception-v3).
4. Ramo 4: Max Pooling 3x3 -> Projeção 1x1.
Ao final, todas as saídas (que têm mesma altura H e largura W) são concatenadas
ao longo do eixo de canais.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloco auxiliar padrão: Conv2d + BatchNorm2d + ReLU."""
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    """
    Módulo Inception com 4 ramos multi-escala em paralelo.
    """
    def __init__(
        self,
        in_channels: int,
        n1x1: int,
        n3x3_reduce: int,
        n3x3: int,
        n5x5_reduce: int,
        n5x5: int,
        pool_proj: int
    ):
        super().__init__()

        # Ramo 1: Conv 1x1 direta
        self.branch1 = ConvBlock(in_channels, n1x1, kernel_size=1)

        # Ramo 2: Conv 1x1 (redução) -> Conv 3x3
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, n3x3_reduce, kernel_size=1),
            ConvBlock(n3x3_reduce, n3x3, kernel_size=3, padding=1)
        )

        # Ramo 3: Conv 1x1 (redução) -> Duas Convs 3x3 (equivalente a 5x5)
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, n5x5_reduce, kernel_size=1),
            ConvBlock(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            ConvBlock(n5x5, n5x5, kernel_size=3, padding=1)
        )

        # Ramo 4: MaxPool 3x3 -> Conv 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        # Concatena os mapas gerados pelos 4 ramos ao longo da dimensão dos canais
        return torch.cat([out1, out2, out3, out4], dim=1)


class InceptionNet(nn.Module):
    """
    Rede Neural Inception (baseada no GoogLeNet).
    
    Args:
        in_channels (int): Canais da imagem de entrada (3 para RGB).
        num_classes (int): Classes de saída (1 para binária).
        dropout (float): Taxa de dropout no classificador.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, dropout: float = 0.4):
        super().__init__()

        # Stem inicial para reduzir a resolução espacial rapidamente
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Estágio Inception 3:
        # (in_channels, n1x1, n3x3_red, n3x3, n5x5_red, n5x5, pool_proj)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)   # Saída: 64+128+32+32 = 256
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64) # Saída: 128+192+96+64 = 480

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Estágio Inception 4:
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)  # Saída: 192+208+48+64 = 512
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64) # Saída: 160+224+64+64 = 512

        # Classificador final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fluxo de tensores:
        # Entrada: [B, C, H, W]
        x = self.stem(x)                # -> [B, 192, H/8, W/8]
        x = self.inception3a(x)         # -> [B, 256, H/8, W/8]
        x = self.inception3b(x)         # -> [B, 480, H/8, W/8]
        x = self.maxpool(x)             # -> [B, 480, H/16, W/16]

        x = self.inception4a(x)         # -> [B, 512, H/16, W/16]
        x = self.inception4b(x)         # -> [B, 512, H/16, W/16]

        x = self.avgpool(x)             # -> [B, 512, 1, 1]
        x = torch.flatten(x, 1)         # -> [B, 512]
        x = self.dropout(x)
        x = self.classifier(x)          # -> [B, num_classes] (logits)
        return x


def inception(num_classes: int = 1, in_channels: int = 3) -> InceptionNet:
    """Instancia a InceptionNet."""
    return InceptionNet(in_channels=in_channels, num_classes=num_classes)
