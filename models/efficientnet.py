"""
=============================================================================
EfficientNet: Compound Model Scaling (Tan & Le, Google Research - ICML 2019)
=============================================================================

Contexto Histórico & O Problema:
--------------------------------
Historicamente, pesquisadores escalavam redes neurais arbitrariamente variando apenas
UMA das três dimensões:
1. Profundidade (d - Depth): Adicionar mais camadas (ex: ResNet-18 -> ResNet-152).
   - Vantagem: Aprende features mais complexas.
   - Desvantagem: Dificuldade de otimização e retornos decrescentes.
2. Largura (w - Width): Aumentar a quantidade de canais por camada (ex: WideResNet).
   - Vantagem: Captura features mais finas e é fácil de treinar.
   - Desvantagem: Redes excessivamente largas e rasas falham em capturar abstrações globais.
3. Resolução (r - Resolution): Aumentar o tamanho da imagem de entrada (ex: 224 -> 480).
   - Vantagem: Detalhes mais ricos.
   - Desvantagem: Custo de FLOPs cresce ao quadrado (O(r^2)).

A Inovação: Compound Scaling (Escalonamento Composto)
---------------------------------------------------
Tan & Le provaram empiricamente que essas três dimensões são interdependentes.
Se a imagem é maior (r maior), a rede precisa de mais camadas (d maior) para que o
campo receptivo cubra as estruturas, e precisa de mais canais (w maior) para capturar
a granularidade fina!
O artigo definiu a Lei de Escalonamento Composto:
    Profundidade: d = alpha ^ phi
    Largura:      w = beta ^ phi
    Resolução:    r = gamma ^ phi
    Sujeito a:    alpha * (beta^2) * (gamma^2) ~= 2

O Bloco MBConv com Squeeze-and-Excitation (SE):
------------------------------------------------
1. Expansão 1x1 + Depthwise Conv (kernel 3x3 ou 5x5).
2. Bloco SE (Atenção de Canais):
   - Squeeze: Global Average Pooling resume a dimensão espacial: [B, C, H, W] -> [B, C, 1, 1].
   - Excitation: Dois FCs aprendem um peso dinâmico de importância entre 0 e 1 (Sigmoid) para cada canal.
   - Scale: O mapa original é multiplicado por esses pesos aprendidos, focando nos canais clinicamente relevantes!
3. Projeção Pointwise Linear 1x1 + Conexão Residual.
4. Função de Ativação SiLU / Swish: f(x) = x * sigmoid(x).

Relevância para Raio-X de Tórax:
--------------------------------
A atenção por canais (Squeeze-and-Excitation) é fabulosa em radiologia porque permite
à rede suprimir canais de ruído de artefatos externos (marcapassos, eletrodos) e
amplificar canais sensíveis a alterações vasculares ou consolidações pulmonares.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """
    Bloco Squeeze-and-Excitation (SE):
    Recalibra adaptativamente a importância de cada canal de características.
    """
    def __init__(self, in_channels: int, reduced_dim: int):
        super().__init__()
        self.se = nn.Sequential(
            # Squeeze: comprime a informação espacial global
            nn.AdaptiveAvgPool2d(1),
            # Excitation: aprende relações entre canais via gargalo MLP
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
            # Sigmoid gera pesos entre 0 (irrelevante) e 1 (muito relevante)
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale: multiplica cada canal pelo seu peso aprendido
        scale = self.se(x)
        return x * scale


class MBConvBlock(nn.Module):
    """
    Bloco Mobile Inverted Bottleneck (MBConv) com SE e SiLU.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []

        # 1. Expansão Pointwise 1x1
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        # 2. Convolução Depthwise
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        # 3. Squeeze-and-Excitation (SE)
        if se_ratio > 0:
            reduced_dim = max(1, int(in_channels * se_ratio))
            layers.append(SqueezeExcitation(hidden_dim, reduced_dim))

        # 4. Projeção Linear 1x1 (sem ativação)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class EfficientNet(nn.Module):
    """
    Arquitetura EfficientNet (Baseline B0).
    
    Args:
        in_channels (int): Canais de entrada (3 para RGB).
        num_classes (int): Classes de saída (1 para binária).
        dropout (float): Taxa de dropout no classificador.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()

        # Configuração MBConv do EfficientNet-B0:
        # (expand_ratio, channels, num_layers, kernel_size, stride)
        b0_config = [
            (1, 16, 1, 3, 1),
            (6, 24, 2, 3, 2),
            (6, 40, 2, 5, 2),
            (6, 80, 3, 3, 2),
            (6, 112, 3, 5, 1),
            (6, 192, 4, 5, 2),
            (6, 320, 1, 3, 1),
        ]

        # Stem inicial
        out_stem = 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_stem, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_stem),
            nn.SiLU(inplace=True)
        )

        # Construção dos estágios MBConv
        layers = []
        curr_in = out_stem
        for t, c, n, k, s in b0_config:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    MBConvBlock(
                        in_channels=curr_in,
                        out_channels=c,
                        kernel_size=k,
                        stride=stride,
                        expand_ratio=t
                    )
                )
                curr_in = c

        self.blocks = nn.Sequential(*layers)

        # Cabeçote convolucional de expansão final
        out_head = 1280
        self.head_conv = nn.Sequential(
            nn.Conv2d(curr_in, out_head, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_head),
            nn.SiLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(out_head, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fluxo de tensores:
        # Entrada: [B, C, H, W]
        x = self.stem(x)                # -> [B, 32, H/2, W/2]
        x = self.blocks(x)              # -> [B, 320, H/32, W/32]
        x = self.head_conv(x)           # -> [B, 1280, H/32, W/32]
        x = self.avgpool(x)             # -> [B, 1280, 1, 1]
        x = torch.flatten(x, 1)         # -> [B, 1280]
        x = self.classifier(x)          # -> [B, num_classes] (logits)
        return x


def efficientnet_b0(num_classes: int = 1, in_channels: int = 3) -> EfficientNet:
    """Instancia a EfficientNet-B0."""
    return EfficientNet(in_channels=in_channels, num_classes=num_classes)
