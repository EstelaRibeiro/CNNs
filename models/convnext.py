"""
=============================================================================
ConvNeXt: A Convolução Modernizada para os Anos 2020 (Liu et al., Meta AI - CVPR 2022)
=============================================================================

Contexto Histórico & A Pergunta Provocativa:
--------------------------------------------
Com o surgimento do Vision Transformer (ViT) e do Swin Transformer, muitos decretaram
o "fim das convoluções". No entanto, os pesquisadores da Meta AI propuseram uma pergunta:
"O sucesso estrondoso dos Transformers veio da auto-atenção em si, ou das técnicas
modernas de design, normalização e treinamento desenvolvidas na última década?"

A "Modernização" da ResNet-50 Passo a Passo:
--------------------------------------------
Para responder, os autores pegaram uma ResNet-50 padrão de 2015 e a modificaram
progressivamente usando os conceitos dos Transformers, MAS SEM USAR NENHUMA ATENÇÃO:

1. Macro Design:
   - Proporção de estágios ajustada para (3: 3: 9: 3), espelhando o Swin Transformer.
   - Stem com "Patchify": Substituiu a conv 7x7 + maxpool por uma convolução 4x4 com
     stride 4 sem sobreposição (dividindo a imagem em patches de 4x4 logo no início).
2. Inverted Bottleneck:
   - Os canais expandem no meio do bloco (fator 4x) e depois projetam de volta,
     idêntico ao bloco MLP do Transformer.
3. Kernels Grandes (7x7 Depthwise):
   - Moveram uma convolução depthwise 7x7 para o início do bloco, simulando
     as janelas de visão ampla dos Transformers.
4. Micro Design Refinado:
   - Substituição de ReLU por GELU (ativação suave).
   - Menos ativações: Apenas UMA ativação GELU por bloco (em vez de uma após cada convolução).
   - Menos normalizações: Apenas UMA normalização por bloco.
   - Substituição de BatchNorm por LayerNorm (estabilidade superior independente do batch size).
   - Camadas dedicadas de Downsampling (Conv 2x2 com stride 2 + LayerNorm).

O Resultado Histórico:
----------------------
Uma rede PURAMENTE convolucional, mais simples, muito rápida para inferência em GPU
e que compete no mesmo nível (ou supera) os Vision Transformers modernos!

Relevância para Raio-X de Tórax:
--------------------------------
ConvNeXt combina o melhor dos dois mundos: a velocidade e o viés indutivo de localidade
das convoluções com a capacidade de abstração macro dos Transformers.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormChannelsFirst(nn.Module):
    """
    LayerNorm aplicado no formato clássico de convoluções 2D: [B, C, H, W].
    Calcula média e variância sobre a dimensão dos canais (C).
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    Bloco Convolucional Moderno ConvNeXt:
    1. Depthwise Conv 7x7 (kernel amplo simulando janelas de atenção).
    2. LayerNorm (Channels First).
    3. Pointwise Conv 1x1 (expande canais 4x).
    4. Ativação GELU (única do bloco!).
    5. Pointwise Conv 1x1 (projeta de volta).
    6. Layer Scale (multiplicador aprendível gamma para estabilidade profunda).
    7. Conexão Residual: x + F(x).
    """
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()

        # 1. Depthwise 7x7
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # 2. LayerNorm
        self.norm = LayerNormChannelsFirst(dim, eps=1e-6)

        # 3 e 4. Inverted Bottleneck (1x1 expande 4x -> GELU -> 1x1 projeta)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # 5. Layer Scale
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_res = x

        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        if self.gamma is not None:
            out = self.gamma * out

        # Conexão Residual
        return input_res + out


class ConvNeXt(nn.Module):
    """
    Arquitetura ConvNeXt Didática.
    
    Args:
        in_channels (int): Canais da imagem de entrada (3 para RGB).
        num_classes (int): Classes de saída (1 para binária).
        depths (tuple): Quantidade de blocos em cada um dos 4 estágios (ex: (3, 3, 9, 3)).
        dims (tuple): Quantidade de canais em cada um dos 4 estágios (ex: (96, 192, 384, 768)).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768)
    ):
        super().__init__()

        # 1. Stem com Patchify (Conv 4x4 com stride 4)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNormChannelsFirst(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # Downsamplings dedicados entre os estágios (Conv 2x2, stride 2)
        for i in range(3):
            downsample = nn.Sequential(
                LayerNormChannelsFirst(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        # 2. Estágios ConvNeXt
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dim=dims[i])
                for _ in range(depths[i])
            ])
            self.stages.append(stage)

        # 3. Cabeçote classificador final com LayerNorm
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passa pelos 4 estágios
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)

        # Global Average Pooling espacial: [B, C, H, W] -> [B, C]
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x


def convnext_tiny(num_classes: int = 1, in_channels: int = 3) -> ConvNeXt:
    """ConvNeXt-Tiny (depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])."""
    return ConvNeXt(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768)
    )

def convnext_small(num_classes: int = 1, in_channels: int = 3) -> ConvNeXt:
    """ConvNeXt-Small (depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])."""
    return ConvNeXt(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768)
    )
