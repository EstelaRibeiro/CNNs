"""
=============================================================================
Model Factory (Módulo de Compatibilidade Retroativa)
=============================================================================

Este arquivo atua como uma ponte de compatibilidade, re-exportando todas as
classes e funções do novo pacote modular `models/`.

Para novos scripts e aulas didáticas, recomenda-se importar diretamente de `models`:
    from models import get_model, ResNet18, VGG, ConvNeXt
ou importar os arquivos específicos:
    from models.alexnet import AlexNet
    from models.resnet import ResNet18
    from models.densenet import DenseNet
"""

from models import (
    # Classes e construtores
    AlexNet, alexnet,
    VGG, VGGBlock, VGG_CONFIGS, vgg11, vgg13, vgg16, vgg19,
    ResNet, BasicBlock, Bottleneck, resnet18, resnet34, resnet50, ResNet18, ResNet50,
    DenseNet, DenseBlock, DenseLayer, TransitionLayer, densenet121, minidensenet, MiniDenseNet,
    InceptionNet, InceptionBlock, inception,
    DepthwiseSeparableConv, InvertedResidual, MobileNetV1, MobileNetV2, mobilenet_v1, mobilenet_v2,
    EfficientNet, MBConvBlock, SqueezeExcitation, efficientnet_b0,
    VisionTransformer, ViT, PatchEmbedding, MultiHeadSelfAttention, TransformerEncoderBlock, vit_tiny, vit_small,
    ConvNeXt, ConvNeXtBlock, convnext_tiny, convnext_small,
    # Funções de fábrica
    get_model, list_available_models, DIDACTIC_BUILDERS
)

__all__ = [
    "AlexNet", "alexnet",
    "VGG", "VGGBlock", "VGG_CONFIGS", "vgg11", "vgg13", "vgg16", "vgg19",
    "ResNet", "BasicBlock", "Bottleneck", "resnet18", "resnet34", "resnet50", "ResNet18", "ResNet50",
    "DenseNet", "DenseBlock", "DenseLayer", "TransitionLayer", "densenet121", "minidensenet", "MiniDenseNet",
    "InceptionNet", "InceptionBlock", "inception",
    "DepthwiseSeparableConv", "InvertedResidual", "MobileNetV1", "MobileNetV2", "mobilenet_v1", "mobilenet_v2",
    "EfficientNet", "MBConvBlock", "SqueezeExcitation", "efficientnet_b0",
    "VisionTransformer", "ViT", "PatchEmbedding", "MultiHeadSelfAttention", "TransformerEncoderBlock", "vit_tiny", "vit_small",
    "ConvNeXt", "ConvNeXtBlock", "convnext_tiny", "convnext_small",
    "get_model", "list_available_models", "DIDACTIC_BUILDERS"
]
