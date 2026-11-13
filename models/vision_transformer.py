"""
=============================================================================
Vision Transformer - ViT: A Quebra do Paradigma Convolucional (Dosovitskiy et al., Google - 2020)
=============================================================================

Contexto Histórico & O Grande Salto:
------------------------------------
Durante quase uma década (2012-2020), a Visão Computacional era sinônimo exclusivo
de Convoluções (CNNs). As CNNs carregam um forte "Viés Indutivo" (Inductive Bias):
1. Localidade 2D: Assume previamente que pixels vizinhos estão fortemente correlacionados.
2. Equivariância por Translação: Um padrão tem o mesmo significado onde quer que esteja na imagem.

O artigo pioneiro "An Image is Worth 16x16 Words" fez uma aposta ousada:
"E se jogássemos fora todo o viés convolucional e tratássemos a imagem simplesmente
como uma frase de texto, aplicando o mecanismo de Atenção Pura do Transformer?"

Como Transformar uma Imagem 2D em "Palavras" (Tokens)?
------------------------------------------------------
1. Patch Partition & Embedding:
   - A imagem 2D [B, C, H, W] é recortada em pequenos ladrilhos (patches) de P x P (ex: 16x16).
   - Uma imagem 224x224 gera (224/16) * (224/16) = 14 * 14 = 196 patches!
   - Cada patch é projetado linearmente para um vetor de características de dimensão D.
2. O Token Especial [CLS] (Classification Token):
   - Um token aprendível extra é adicionado na posição zero da sequência.
   - Após passar pelas camadas de auto-atenção, o estado desse token sintetizou a informação
     global da imagem inteira e alimenta o classificador final.
3. Position Embeddings (Codificação Posicional):
   - Como a auto-atenção pura é invariante à ordem das sequências, vetores posicionais
     aprendíveis 1D são SOMADOS aos tokens, informando onde cada patch estava no espaço 2D.
4. Multi-Head Self-Attention (MSA):
   - Permite que qualquer pedaço da imagem olhe e converse diretamente com qualquer outro
     pedaço desde a primeiríssima camada!
   - Fórmula da Atenção: Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V

Relevância para Raio-X de Tórax:
--------------------------------
Permite ao modelo correlacionar anomalias distantes instantaneamente (ex: comparar o ápice
do pulmão direito diretamente com a base do pulmão esquerdo). Porém, sem viés indutivo prévio,
Transformers puros precisam de grandes volumes de dados ou pré-treinamento cuidadoso para convergir.
"""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Divide a imagem em patches P x P e os projeta linearmente para vetores de dimensão 'embed_dim'.
    Pode ser implementado eficientemente como uma convolução 2D com kernel_size=patch_size e stride=patch_size.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entrada:  [B, C, H, W]
        # Convolui: [B, embed_dim, H/P, W/P]
        # Achata:   [B, embed_dim, N_patches]
        # Transpõe: [B, N_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Mecanismo de Auto-Atenção Multi-Cabeça (Multi-Head Self-Attention - MSA).
    Calcula matrizes de afinidade entre todos os pares de patches da imagem.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert embed_dim % num_heads == 0, "embed_dim deve ser divisível por num_heads"

        # Projeção linear conjunta para Query (Q), Key (K) e Value (V)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Projeta e divide em cabeças de atenção
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product: softmax( (Q @ K^T) / sqrt(d_k) )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Multiplica por V e recombina as cabeças
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    """
    Camada Feed-Forward Network (FFN) com ativação GELU e expansão de dimensão.
    """
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Bloco Encoder do Transformer com arquitetura Pre-LayerNorm e conexões residuais:
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Arquitetura Vision Transformer.
    
    Args:
        img_size (int): Resolução da imagem de entrada (ex: 224 ou 384).
        patch_size (int): Tamanho de cada patch quadrado (ex: 16).
        in_channels (int): Canais de entrada (3 para RGB).
        num_classes (int): Classes de saída (1 para binária).
        embed_dim (int): Dimensão dos vetores de incorporação (tokens).
        depth (int): Quantidade de blocos Encoder repetidos.
        num_heads (int): Quantidade de cabeças de atenção.
        mlp_ratio (float): Fator de expansão da camada MLP.
        drop_rate (float): Taxa de dropout.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Token especial de classificação aprendível [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings aprendíveis para os (num_patches + 1) tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Pilha de blocos Transformer
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate
            )
            for _ in range(depth)
        ])

        # Cabeça final
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Inicialização com trunc_normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fluxo de tensores:
        # Entrada: [B, C, H, W]
        B = x.shape[0]

        # 1. Projeção de patches: [B, N, embed_dim]
        x = self.patch_embed(x)

        # 2. Concatena o [CLS] token no início: [B, N+1, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Adiciona posições espaciais
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Passa pelos blocos de atenção global
        x = self.blocks(x)

        # 5. Extrai a saída do token [CLS] (índice 0)
        x = self.norm(x)
        cls_out = x[:, 0]

        # 6. Predição final
        out = self.head(cls_out)
        return out


def vit_tiny(img_size: int = 224, patch_size: int = 16, num_classes: int = 1) -> VisionTransformer:
    """Configuração leve do ViT (depth=6, heads=3, dim=192)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=192,
        depth=6,
        num_heads=3,
        num_classes=num_classes
    )

def vit_small(img_size: int = 224, patch_size: int = 16, num_classes: int = 1) -> VisionTransformer:
    """ViT-Small (depth=12, heads=6, dim=384)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes
    )

# Alias
ViT = VisionTransformer
