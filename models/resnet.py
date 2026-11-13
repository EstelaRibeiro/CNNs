"""
=============================================================================
ResNet: Deep Residual Learning (He, Zhang, Ren & Sun, Microsoft Research - 2015)
=============================================================================

Contexto Histórico & O Problema:
--------------------------------
Antes da ResNet, ao aumentar a profundidade de uma CNN (ex: de 20 para 56 camadas),
o erro de treino AUMENTAVA drasticamente. Isso não era overfitting (já que o erro de
treino piorava junto com o de teste), mas sim o "Problema de Degradação":
durante o backpropagation, sucessivas multiplicações de derivadas de camadas faziam
com que os gradientes sumissem exponencialmente (*Vanishing Gradient*).

A Inovação Revolucionária: Conexões Residuais (Skip Connections)
-----------------------------------------------------------------
Em vez de forçar a rede a aprender um mapeamento completo direto H(x), a ResNet
divide a tarefa em:
    H(x) = F(x) + x
onde:
    - x é a entrada original (o atalho de identidade, sem pesos adicionais)
    - F(x) é a função residual calculada pelas convoluções

Por que o Gradiente Não Some?
-----------------------------
Pela regra da cadeia, a derivada da perda em relação à entrada do bloco é:
    d(Loss) / dx = d(Loss) / dH * (dF(x)/dx + 1)

O termo "+ 1" funciona como uma "super-rodovia" de gradientes!
Mesmo que dF(x)/dx seja minúsculo ou nulo, o sinal do erro flui diretamente (+ 1)
de volta até as primeiras camadas sem sofrer atenuação. Foi assim que a ResNet
conseguiu treinar redes estáveis com mais de 100 ou 1000 camadas!

Blocos Construtores:
--------------------
1. BasicBlock (ResNet-18 e ResNet-34):
   - Duas convoluções 3x3 com BatchNorm e ReLU.
2. Bottleneck Block (ResNet-50, 101, 152):
   - Estratégia 1x1 (reduz canais) -> 3x3 (processa) -> 1x1 (expande canais em 4x),
     economizando milhões de operações matemáticas.

Relevância para Raio-X de Tórax:
--------------------------------
ResNet (especialmente ResNet-18 e ResNet-50) é o cavalo de batalha mais consagrado
e confiável da literatura de imagem médica para classificação de patologias pulmonares.
"""

from typing import Type, Union, List
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Bloco residual padrão para redes menores (ResNet-18 e ResNet-34).
    Composto por 2 convoluções 3x3 com BatchNorm e ReLU intercaladas.
    """
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super().__init__()

        # Conv 1: Pode reduzir a resolução se stride > 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Conv 2: Mantém resolução e canais
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Atalho com projeção 1x1 caso canais ou resolução mudem
        if downsample is None and (stride != 1 or in_channels != out_channels * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Ramo residual F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Ramo de atalho (downsample se necessário)
        if self.downsample is not None:
            identity = self.downsample(identity)

        # A soma residual: H(x) = F(x) + x
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bloco Residual Gargalo ("Bottleneck") para redes mais profundas (ResNet-50, 101, 152).
    Design:
      1. Conv 1x1 comprime a quantidade de canais
      2. Conv 3x3 executa a filtragem espacial com baixo custo computacional
      3. Conv 1x1 expande os canais de volta (fator de expansão = 4)
    """
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super().__init__()

        # Passo 1: Redução 1x1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Passo 2: Convolução 3x3
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Passo 3: Expansão 1x1 (multiplica canais por 4)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        if downsample is None and (stride != 1 or in_channels != out_channels * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # Soma residual
        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Arquitetura ResNet Geral (base para ResNet-18, 34, 50, etc.).
    
    Args:
        block: Tipo de bloco residual (BasicBlock ou Bottleneck).
        layers: Lista com a quantidade de blocos em cada um dos 4 estágios.
        in_channels: Número de canais de entrada (padrão: 3 para RGB).
        num_classes: Número de saídas (1 para classificação binária).
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int = 3,
        num_classes: int = 1
    ):
        super().__init__()
        self.in_channels = 64

        # Stem inicial: Conv 7x7 com stride 2 + MaxPool 3x3 reduz a resolução em 4x
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Estágios de Blocos Residuais
        self.layer1 = self._make_layer(block, out_channels=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, out_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, out_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, out_channels=512, blocks=layers[3], stride=2)

        # Classificador final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        blocks: int,
        stride: int
    ) -> nn.Sequential:
        layers = []
        # O primeiro bloco do estágio cuida da mudança de stride e expansão
        layers.append(block(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels * block.expansion

        # Os blocos subsequentes mantêm a resolução e canais
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fluxo de tensores passo a passo:
        # Entrada: [B, C, H, W]          (ex: [8, 3, 224, 224])
        x = self.conv1(x)               # -> [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)             # -> [B, 64, 56, 56]

        x = self.layer1(x)              # -> [B, 64 * exp, 56, 56]
        x = self.layer2(x)              # -> [B, 128 * exp, 28, 28]
        x = self.layer3(x)              # -> [B, 256 * exp, 14, 14]
        x = self.layer4(x)              # -> [B, 512 * exp, 7, 7]

        x = self.avgpool(x)             # -> [B, 512 * exp, 1, 1]
        x = torch.flatten(x, 1)         # -> [B, 512 * exp]
        x = self.fc(x)                  # -> [B, num_classes] (logits)
        return x


# Funções construtoras:
def resnet18(num_classes: int = 1, in_channels: int = 3) -> ResNet:
    """Instancia a ResNet-18 (8 blocos BasicBlock, [2, 2, 2, 2])."""
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)

def resnet34(num_classes: int = 1, in_channels: int = 3) -> ResNet:
    """Instancia a ResNet-34 (16 blocos BasicBlock, [3, 4, 6, 3])."""
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

def resnet50(num_classes: int = 1, in_channels: int = 3) -> ResNet:
    """Instancia a ResNet-50 (16 blocos Bottleneck, [3, 4, 6, 3])."""
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

# Alias para facilitar o uso didático
ResNet18 = resnet18
ResNet50 = resnet50
