"""
=============================================================================
Helper de Carregamento: Modelos Pré-Treinados (timm) vs Didáticos do Zero
=============================================================================

Este módulo ilustra as duas formas fundamentais de trabalhar
com redes neurais em Visão Computacional:

1. Transfer Learning / Backbones da Indústria (via biblioteca `timm`):
   Ideal para quando você precisa de pesos pré-treinados no ImageNet para atingir
   rapidamente o estado da arte com poucos dados médicos.

2. Modelos Didáticos Artesanais (via nosso pacote `models/`):
   Ideal para quando você quer entender como a arquitetura funciona por dentro,
   inspecionar o grafo de operações, alterar conexões residuais ou analisar o fluxo
   de gradientes sem camadas "caixa-preta".
"""

from typing import Optional
import torch
import torch.nn as nn
from models.factory import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_classification_model(
    model_name: str,
    use_timm: bool = True,
    pretrained: bool = True,
    num_classes: int = 1,
    in_channels: int = 3
) -> nn.Module:
    """
    Carrega um modelo e adapta sua camada final para classificação binária ou multi-classe.

    Args:
        model_name (str): Nome da arquitetura (ex: 'vgg16', 'resnet50', 'convnext_tiny', 'vit_tiny').
        use_timm (bool): Se True, carrega o modelo através da biblioteca `timm`.
                         Se False, carrega a implementação artesanal em `models/`.
        pretrained (bool): Se True, carrega os pesos pré-treinados do ImageNet (apenas com use_timm=True).
        num_classes (int): Quantidade de classes na saída (1 para binária com BCEWithLogitsLoss).
        in_channels (int): Canais da imagem (3 para RGB).

    Returns:
        nn.Module: Modelo instanciado e transferido para o dispositivo (GPU ou CPU).
    """
    try:
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            use_timm=use_timm,
            pretrained=pretrained
        )

        origem = "timm (pré-treinado)" if (use_timm and pretrained) else ("timm (do zero)" if use_timm else "didático artesanal (PyTorch puro)")
        print(f"Modelo '{model_name}' carregado com sucesso via {origem} no device: {device}.")
        return model.to(device)

    except Exception as e:
        print(f"Erro ao instanciar o modelo '{model_name}'. Detalhes: {e}")
        raise
