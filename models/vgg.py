"""
=============================================================================
VGG: Very Deep Convolutional Networks (Simonyan & Zisserman, Oxford - 2014)
=============================================================================

Contexto Histórico & O Problema:
--------------------------------
Após a AlexNet (2012), pesquisadores tentavam criar redes mais profundas usando
filtros variados (11x11, 7x7, 5x5). A VGG provou uma regra de ouro que simplificou
para sempre a Visão Computacional:
"Você não precisa de filtros grandes; basta empilhar pequenos filtros 3x3."

A Equivalência Matemática do Campo Receptivo:
---------------------------------------------
Dois filtros 3x3 consecutivos têm o MESMO campo receptivo de um filtro 5x5:
    Campo Receptivo = 1 + (3 - 1) + (3 - 1) = 5
Três filtros 3x3 consecutivos têm o MESMO campo receptivo de um filtro 7x7:
    Campo Receptivo = 1 + 3 * (3 - 1) = 7

Por que isso é melhor?
1. Menos Parâmetros (Economia de Memória e Computação):
   - Um filtro 7x7 com C canais usa: 7 * 7 * C^2 = 49 C^2 pesos.
   - Três filtros 3x3 usam: 3 * (3 * 3 * C^2) = 27 C^2 pesos (~45% de redução!).
2. Mais Não-Linearidade:
   - Três filtros 3x3 possuem 3 ativações ReLU intercaladas em vez de apenas 1,
     tornando a rede muito mais expressiva.

Relevância para Raio-X de Tórax:
--------------------------------
A VGG é excelente para capturar texturas visuais contínuas (como consolidações ou
infiltrados pulmonares), mas seu classificador denso final possui muitos parâmetros,
o que exige cuidado com overfitting em datasets médicos pequenos.
"""

from typing import List, Tuple, Union
import torch
import torch.nn as nn


# Configuração dos estágios da VGG
# Formato de cada tupla: (canais_de_saida, quantidade_de_convolucoes_3x3)
VGG_CONFIGS = {
    "vgg11": [(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)],
    "vgg13": [(64, 2), (128, 2), (256, 2), (512, 2), (512, 2)],
    "vgg16": [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)],
    "vgg19": [(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)],
}


class VGGBlock(nn.Module):
    """
    Bloco básico da VGG:
    Sequência de N convoluções 3x3 (com ReLU) seguidas de Max Pooling 2x2.
    
    Exemplo:
        Entrada: [B, in_channels, H, W]
        Saída:   [B, out_channels, H // 2, W // 2]
    """
    def __init__(self, in_channels: int, out_channels: int, num_convs: int):
        super().__init__()
        layers = []

        for i in range(num_convs):
            curr_in = in_channels if i == 0 else out_channels
            layers.append(
                nn.Conv2d(curr_in, out_channels, kernel_size=3, padding=1, bias=True)
            )
            layers.append(nn.ReLU(inplace=True))

        # Max Pooling reduz as dimensões espaciais (H, W) pela metade
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG(nn.Module):
    """
    Arquitetura VGG Parametrizável.
    
    Args:
        cfg (str ou lista): Nome da variante ('vgg11', 'vgg13', 'vgg16', 'vgg19')
                            ou lista customizada de estágios.
        in_channels (int): Número de canais de entrada (3 para RGB/Raio-X colorido, 1 para escala de cinza).
        num_classes (int): Classes de saída (1 para classificação binária).
        dropout (float): Taxa de dropout para regularização no cabeçote classificador.
    """
    def __init__(
        self,
        cfg: Union[str, List[Tuple[int, int]]] = "vgg16",
        in_channels: int = 3,
        num_classes: int = 1,
        dropout: float = 0.5
    ):
        super().__init__()

        if isinstance(cfg, str):
            cfg_name = cfg.lower()
            if cfg_name not in VGG_CONFIGS:
                raise ValueError(f"Configuração '{cfg}' inválida. Escolha entre: {list(VGG_CONFIGS.keys())}")
            cfg = VGG_CONFIGS[cfg_name]

        # Constrói os 5 estágios convolucionais da VGG
        blocks = []
        curr_in = in_channels

        for out_channels, num_convs in cfg:
            blocks.append(
                VGGBlock(in_channels=curr_in, out_channels=out_channels, num_convs=num_convs)
            )
            curr_in = out_channels

        self.features = nn.Sequential(*blocks)

        # AdaptiveAvgPool2d(1, 1): Garante que a rede aceite qualquer tamanho de imagem de entrada
        # (ex: 224x224, 384x384, 512x512) sem quebrar o tamanho da camada Linear final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Cabeçote de classificação adaptado para sala de aula e tarefas médicas
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(curr_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fluxo de tensores passo a passo:
        # Entrada: [B, C, H, W]          (ex: [8, 3, 384, 384])
        x = self.features(x)            # -> [B, 512, H/32, W/32]
        x = self.avgpool(x)             # -> [B, 512, 1, 1]
        x = self.classifier(x)          # -> [B, num_classes] (logits)
        return x


# Funções construtoras convenientes:
def vgg11(num_classes: int = 1, in_channels: int = 3, dropout: float = 0.5) -> VGG:
    """Instancia a VGG-11 (8 camadas conv + 3 densas)."""
    return VGG(cfg="vgg11", in_channels=in_channels, num_classes=num_classes, dropout=dropout)

def vgg13(num_classes: int = 1, in_channels: int = 3, dropout: float = 0.5) -> VGG:
    """Instancia a VGG-13 (10 camadas conv + 3 densas)."""
    return VGG(cfg="vgg13", in_channels=in_channels, num_classes=num_classes, dropout=dropout)

def vgg16(num_classes: int = 1, in_channels: int = 3, dropout: float = 0.5) -> VGG:
    """Instancia a VGG-16 clássica (13 camadas conv + 3 densas)."""
    return VGG(cfg="vgg16", in_channels=in_channels, num_classes=num_classes, dropout=dropout)

def vgg19(num_classes: int = 1, in_channels: int = 3, dropout: float = 0.5) -> VGG:
    """Instancia a VGG-19 (16 camadas conv + 3 densas)."""
    return VGG(cfg="vgg19", in_channels=in_channels, num_classes=num_classes, dropout=dropout)
