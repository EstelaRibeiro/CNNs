"""
=============================================================================
Model Factory Unificado: Didático (Classes Puras) & timm (Pré-treinado)
=============================================================================

Este módulo permite carregar qualquer arquitetura da história do Deep Learning
para a tarefa de classificação binária de exames de Raio-X através de dois modos:

1. Modo Didático Artesanal (use_timm=False):
   Instancia classes puras em PyTorch (`nn.Module`) escritas diretamente nos arquivos
   do pacote `models/` (ex: alexnet.py, resnet.py, densenet.py).
   -> Ideal para sala de aula: inspecionar tensores, blocos residuais, concatenações e gradientes.

2. Modo timm / Pré-treinado (use_timm=True):
   Instancia a arquitetura utilizando a biblioteca líder da indústria `timm`,
   permitindo carregar pesos oficiais pré-treinados no ImageNet (`pretrained=True`).
   -> Ideal para benchmarks comparativos e transferência de aprendizado em radiologia.
"""

from typing import List, Dict, Any, Callable
import torch
import torch.nn as nn

# Importação dos módulos didáticos individuais
from .alexnet import AlexNet, alexnet
from .vgg import VGG, VGGBlock, VGG_CONFIGS, vgg11, vgg13, vgg16, vgg19
from .resnet import ResNet, BasicBlock, Bottleneck, resnet18, resnet34, resnet50, ResNet18, ResNet50
from .densenet import DenseNet, DenseBlock, DenseLayer, TransitionLayer, densenet121, minidensenet, MiniDenseNet
from .inception import InceptionNet, InceptionBlock, inception
from .mobilenet import DepthwiseSeparableConv, InvertedResidual, MobileNetV1, MobileNetV2, mobilenet_v1, mobilenet_v2
from .efficientnet import EfficientNet, MBConvBlock, SqueezeExcitation, efficientnet_b0
from .vision_transformer import VisionTransformer, ViT, PatchEmbedding, MultiHeadSelfAttention, TransformerEncoderBlock, vit_tiny, vit_small
from .convnext import ConvNeXt, ConvNeXtBlock, convnext_tiny, convnext_small


# Mapeamento de construtores das implementações didáticas puras
DIDACTIC_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    # 1. AlexNet (2012)
    "alexnet": alexnet,

    # 2. VGG (2014)
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,

    # 3. ResNet (2015)
    "resnet": resnet18,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,

    # 4. DenseNet (2017)
    "densenet": densenet121,
    "densenet121": densenet121,
    "minidensenet": minidensenet,

    # 5. Inception (2014)
    "inception": inception,
    "googlenet": inception,

    # 6. MobileNet (2017-2018)
    "mobilenet": mobilenet_v2,
    "mobilenet_v1": mobilenet_v1,
    "mobilenet_v2": mobilenet_v2,

    # 7. EfficientNet (2019)
    "efficientnet": efficientnet_b0,
    "efficientnet_b0": efficientnet_b0,

    # 8. Vision Transformer (2020)
    "vit": vit_tiny,
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vision_transformer": vit_tiny,

    # 9. ConvNeXt (2022)
    "convnext": convnext_tiny,
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
}


def list_available_models() -> List[str]:
    """Retorna a lista de nomes aceitos para as implementações didáticas."""
    return sorted(list(DIDACTIC_BUILDERS.keys()))


def get_model(
    model_name: str,
    num_classes: int = 1,
    in_channels: int = 3,
    use_timm: bool = False,
    pretrained: bool = False,
    **kwargs: Any
) -> nn.Module:
    """
    Instancia o modelo solicitado, permitindo alternar facilmente entre
    as implementações artesanais didáticas e os modelos prontos da biblioteca `timm`.

    Args:
        model_name (str): Nome do modelo (ex: 'alexnet', 'vgg16', 'resnet18',
                          'resnet50', 'densenet121', 'inception', 'mobilenet_v2',
                          'efficientnet_b0', 'vit_tiny', 'convnext_tiny').
        num_classes (int): Quantidade de classes na saída (1 para classificação binária).
        in_channels (int): Canais da imagem de entrada (3 para RGB, 1 para escala de cinza).
        use_timm (bool): Se True, carrega o modelo usando `timm.create_model`.
                         Se False (padrão se pretrained=False), instancia nossa classe didática em PyTorch puro.
        pretrained (bool): Se True, baixa e utiliza os pesos pré-treinados do ImageNet via `timm`.
                           (Quando True, ativa automaticamente use_timm=True).
        **kwargs: Parâmetros extras repassados ao construtor do modelo.

    Returns:
        nn.Module: O modelo PyTorch pronto para treinamento ou inferência.
    """
    clean_name = model_name.lower().strip()

    # Se o usuário solicitou pesos pré-treinados ou marcou use_timm=True, delega ao timm
    if pretrained or use_timm:
        try:
            import timm
            # Mapeamento de nomes comuns para identificadores padrão do timm
            timm_name_map = {
                "inception": "inception_v3",
                "googlenet": "inception_v3",
                "vit": "vit_tiny_patch16_224",
                "vit_tiny": "vit_tiny_patch16_224",
                "vit_small": "vit_small_patch16_224",
                "convnext": "convnext_tiny",
            }
            resolved_timm_name = timm_name_map.get(clean_name, clean_name)

            model = timm.create_model(
                resolved_timm_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
                **kwargs
            )
            return model
        except Exception as e:
            raise RuntimeError(
                f"Erro ao instanciar modelo '{model_name}' via biblioteca timm: {e}\n"
                f"Dica: verifique se a biblioteca timm está instalada e se a conexão com a internet está ativa caso pretrained=True."
            )

    # Modo Didático Puro (classes implementadas nos arquivos do pacote models/)
    if clean_name in DIDACTIC_BUILDERS:
        builder = DIDACTIC_BUILDERS[clean_name]
        return builder(num_classes=num_classes, in_channels=in_channels, **kwargs)

    raise ValueError(
        f"Modelo '{model_name}' não encontrado nas implementações didáticas.\n"
        f"Modelos disponíveis para estudo didático: {list_available_models()}\n"
        f"Para usar modelos pré-treinados ou backbones adicionais, defina use_timm=True ou pretrained=True."
    )
