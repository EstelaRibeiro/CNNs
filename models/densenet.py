"""
=============================================================================
DenseNet: Densely Connected Networks (Huang, Liu & Weinberger - CVPR 2017)
=============================================================================

Contexto Histórico & O Problema:
--------------------------------
Se a ResNet (2015) descobriu que conectar camadas anteriores via SOMA (x + F(x))
ajuda o fluxo do gradiente, a DenseNet levou essa ideia ao extremo:
"E se conectássemos TODAS as camadas com TODAS as camadas subsequentes via CONCATENAÇÃO?"

Diferença Crucial: Soma (ResNet) vs Concatenação (DenseNet)
----------------------------------------------------------
- ResNet: H(x) = F(x) + x. A informação da identidade se mistura aritmeticamente
  com os resíduos.
- DenseNet: x_l = H([x_0, x_1, x_2, ..., x_{l-1}]). Os mapas de características
  são empilhados diretamente no eixo dos canais (`dim=1`), mantendo os atributos
  anteriores intactos.

Pilares da DenseNet:
------------------------------
1. Reuso Massivo de Características (Feature Reuse):
   Camadas finais não precisam "reaprender" padrões básicos de bordas e texturas;
   elas têm acesso direto às features geradas pelas primeiras camadas da rede.
2. Camadas Estreitas & Taxa de Crescimento (Growth Rate, k):
   Como cada camada recebe a riqueza de todas as anteriores, cada camada individual
   só precisa produzir uma pequena quantidade de novos canais (ex: k = 32).
   Isso torna a DenseNet muito eficiente em número total de parâmetros.
3. Camadas de Transição (Transition Layers):
   Como a concatenação contínua aumentaria indefinidamente o número de canais,
   blocos de transição aplicam uma Conv 1x1 com fator de compressão (theta = 0.5,
   reduzindo os canais pela metade) e um Average Pooling 2x2 para reduzir a resolução espacial.

Relevância para Raio-X de Tórax:
--------------------------------
A DenseNet (em especial a **DenseNet-121**) é historicamente uma das arquiteturas de
MAIOR sucesso em radiografia de tórax (adotada amplamente pelo dataset CheXNet de Stanford).
Seu reuso de features e fluxo direto de gradientes é ideal para datasets médicos
com quantidade limitada de exames.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """
    Camada densa individual com design Bottleneck:
    BN -> ReLU -> Conv 1x1 (redução) -> BN -> ReLU -> Conv 3x3 (produz k novos canais).
    Ao final, concatena os novos k canais com os canais de entrada.
    """
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super().__init__()

        # Conv 1x1 (Bottleneck) comprime para 4 * growth_rate canais
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False),

            # Conv 3x3 produz exatamente 'growth_rate' (k) novos canais
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layer(x)
        # Concatenação densa: junta as features antigas com os novos k canais
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    """
    Bloco Denso: contém 'num_layers' camadas densas encadeadas.
    A cada camada, o número de canais aumenta em 'growth_rate'.
    """
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int, bn_size: int = 4):
        super().__init__()
        layers = []
        channels = in_channels

        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels=channels, growth_rate=growth_rate, bn_size=bn_size))
            channels += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransitionLayer(nn.Module):
    """
    Camada de Transição entre blocos densos:
    Aplica Conv 1x1 para comprimir canais e AvgPool 2x2 para reduzir resolução espacial.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)


class DenseNet(nn.Module):
    """
    Arquitetura DenseNet Modular.
    
    Args:
        growth_rate (int): Quantidade de novos canais gerados por camada (k = 32).
        block_config (tuple): Quantidade de camadas em cada um dos 4 blocos densos.
        num_init_features (int): Quantidade de canais após o stem inicial (64).
        compression (float): Fator theta de redução de canais nas transições (0.5 = metade).
        in_channels (int): Canais da imagem de entrada (3 para RGB).
        num_classes (int): Classes de saída (1 para binária).
    """
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64,
        compression: float = 0.5,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        super().__init__()

        # Stem inicial (Conv 7x7 + MaxPool)
        self.features = nn.Sequential()
        self.features.add_module(
            "stem_conv",
            nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        )
        self.features.add_module("stem_bn", nn.BatchNorm2d(num_init_features))
        self.features.add_module("stem_relu", nn.ReLU(inplace=True))
        self.features.add_module("stem_pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Construção dos Blocos Densos e Transições
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                in_channels=num_features,
                growth_rate=growth_rate,
                num_layers=num_layers
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = block.out_channels

            # Adiciona camada de transição entre blocos (exceto após o último)
            if i != len(block_config) - 1:
                out_trans = int(num_features * compression)
                trans = TransitionLayer(in_channels=num_features, out_channels=out_trans)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = out_trans

        # Normalização e ativação final
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.features.add_module("relu_final", nn.ReLU(inplace=True))

        # Classificador final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fluxo de tensores:
        # Entrada: [B, C, H, W]
        features = self.features(x)     # -> [B, num_features, H', W']
        out = self.avgpool(features)    # -> [B, num_features, 1, 1]
        out = torch.flatten(out, 1)     # -> [B, num_features]
        out = self.classifier(out)      # -> [B, num_classes] (logits)
        return out


# Construtores:
def densenet121(num_classes: int = 1, in_channels: int = 3) -> DenseNet:
    """Instancia a clássica DenseNet-121 (k=32, blocos: [6, 12, 24, 16])."""
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), in_channels=in_channels, num_classes=num_classes)

def minidensenet(growth_rate: int = 32, num_classes: int = 1, in_channels: int = 3) -> DenseNet:
    """Versão leve didática para experimentação rápida em sala de aula."""
    return DenseNet(growth_rate=growth_rate, block_config=(4, 4, 4), in_channels=in_channels, num_classes=num_classes)

# Alias para compatibilidade
MiniDenseNet = minidensenet
