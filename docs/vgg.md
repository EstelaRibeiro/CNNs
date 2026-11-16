# 🧱 VGG (2014) — A Elegância e a Homogeneidade dos Filtros 3x3

> **Artigo Original:** *Very Deep Convolutional Networks for Large-Scale Image Recognition*  
> **Autores:** Karen Simonyan, Andrew Zisserman (Visual Geometry Group, Universidade de Oxford)  
> **Conferência:** ICLR 2015 (2º lugar no ILSVRC 2014)  
> **Código Didático:** [`models/vgg.py`](../models/vgg.py)

---

## 1. Contexto Histórico e O Problema

Após o sucesso estrondoso da AlexNet (2012), a comunidade de Visão Computacional deparou-se com uma pergunta crucial de engenharia:
> *"Se redes mais profundas performam melhor, como devemos escolher a geometria dos filtros? Devemos misturar kernels 11x11, 7x7, 5x5 e 3x3 em cada estágio?"*

As redes anteriores misturavam tamanhos de filtros de forma empírica e desordenada. A **VGG** respondeu a essa pergunta formulando uma das regras de ouro mais elegantes do Deep Learning:
**"Descarte filtros grandes. Use exclusivamente pequenas convoluções $3 \times 3$ empilhadas sequencialmente."**

---

## 2. A Descoberta Matemática: Fatoração do Campo Receptivo

A inovação conceitual central da VGG é demonstrar que o **campo receptivo efetivo** de múltiplos filtros pequenos consecutivos é matematicamente idêntico ao de um único filtro grande.

### 2.1 A Prova do Campo Receptivo
O campo receptivo $R_k$ após aplicar sucessivas camadas convolucionais com kernel $k$ e stride $s=1$ é dado por:
$$R = 1 + \sum_{i=1}^{L} (k_i - 1)$$

- **Duas convoluções $3 \times 3$ consecutivas:**
  $$R = 1 + (3 - 1) + (3 - 1) = 5$$
  *Duas camadas $3 \times 3$ enxergam a mesma área espacial que uma única camada $5 \times 5$!*
- **Três convoluções $3 \times 3$ consecutivas:**
  $$R = 1 + (3 - 1) + (3 - 1) + (3 - 1) = 7$$
  *Três camadas $3 \times 3$ enxergam a mesma área espacial que uma camada $7 \times 7$!*

```text
Visualização do Campo Receptivo:
Camada 3 (3x3):       [ * ]
                     /     \
Camada 2 (3x3):    [ * * * * * ]
                  /             \
Camada 1 (Entrada): [* * * * * * *]  <-- Cobre 7x7 pixels com 3 camadas 3x3!
```

### 2.2 As Duas Grandes Vantagens em Relação a Filtros Grandes

1. **Drástica Redução de Parâmetros:**
   Suponha uma entrada com $C$ canais de entrada e saída:
   - Um único filtro $7 \times 7$:
     $$\text{Parâmetros} = 7 \times 7 \times C \times C = 49 C^2$$
   - Três filtros $3 \times 3$:
     $$\text{Parâmetros} = 3 \times (3 \times 3 \times C \times C) = 27 C^2$$
   **Resultado:** Uma redução impressionante de **$\approx 45\%$** no número de pesos para realizar a mesmíssima cobertura espacial!
2. **Maior Capacidade de Discriminação (Mais Não-Linearidades):**
   Um filtro $7 \times 7$ aplica apenas uma função de ativação ReLU. Três filtros $3 \times 3$ possuem **três ReLUs intercaladas**, permitindo que a rede aprenda funções de decisão muito mais complexas e não-lineares.

---

## 3. Dissecando o Código em `models/vgg.py`

A implementação em [`models/vgg.py`](../models/vgg.py) estrutura a arquitetura em dois componentes fundamentais: o bloco construtor [`VGGBlock`](../models/vgg.py#L31-L51) e a classe principal [`VGG`](../models/vgg.py#L53-L102).

### 3.1 O Bloco Construtor Modular: `VGGBlock`
No arquivo [`models/vgg.py`](../models/vgg.py), o `VGGBlock` encapsula a repetição de $N$ convoluções $3 \times 3$ com `padding=1` (para preservar o tamanho espacial entre convs), seguidas por um `MaxPool2d(2, stride=2)` que reduz a resolução espacial pela metade:

```python
# Trecho didático de models/vgg.py:
for i in range(num_convs):
    curr_in = in_channels if i == 0 else out_channels
    layers.append(
        nn.Conv2d(curr_in, out_channels, kernel_size=3, padding=1, bias=True)
    )
    layers.append(nn.ReLU(inplace=True))

layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
```

### 3.2 As Configurações dos Estágios (`VGG_CONFIGS`)
A VGG padronizou a regra de **dobrar o número de canais** toda vez que a resolução espacial é reduzida pela metade via MaxPool:
$$64 \to 128 \to 256 \to 512 \to 512$$

As variantes consagradas são definidas no dicionário:
- `vgg11`: `[(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)]` (8 camadas conv)
- `vgg13`: `[(64, 2), (128, 2), (256, 2), (512, 2), (512, 2)]` (10 camadas conv)
- `vgg16`: `[(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]` (13 camadas conv)
- `vgg19`: `[(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)]` (16 camadas conv)

### 3.3 Fluxo de Tensores (*Forward Pass*)
```text
Entrada:            [B, 3, 224, 224]
Estágio 1 (64):     [B, 64, 112, 112]
Estágio 2 (128):    [B, 128, 56, 56]
Estágio 3 (256):    [B, 256, 28, 28]
Estágio 4 (512):    [B, 512, 14, 14]
Estágio 5 (512):    [B, 512, 7, 7]
AdaptiveAvgPool:    [B, 512, 1, 1]
Classifier:         [B, num_classes] (logits)
```

> **Ajuste Didático Moderno:** Na implementação original da VGG em 2014, utilizava-se um achatamento direto de $512 \times 7 \times 7 = 25.088$ conexões para o classificador denso. No nosso código didático em [`models/vgg.py`](../models/vgg.py), incluímos `AdaptiveAvgPool2d((1, 1))`. Isso permite que a rede processe radiografias de **qualquer resolução** (como $224 \times 224$, $384 \times 384$ ou $512 \times 512$) sem estourar o tamanho da matriz Linear final.

---

## 4. Relevância em Radiografias de Tórax

1. **Captura Fina de Texturas Pulmonares:**
   Por utilizar uma cascata profunda de convoluções $3 \times 3$ com receptive field crescente, a VGG destaca-se na detecção de **padrões intersticiais e opacidades alveolares** (como infiltrados pneumônicos difusos).
2. **Custo de Memória em Treinamento:**
   Mesmo com a fatoração $3 \times 3$, a VGG é uma rede pesada para treinar do zero (a VGG-16 clássica possui ~138 milhões de parâmetros na versão original). Em sala de aula, ela ilustra a transição histórica entre modelos clássicos pesados e os modelos residuais que vieram a seguir.

---

## 5. Como Instanciar e Testar

```python
import torch
from models.vgg import vgg11, vgg16, VGG

# 1. Instanciar a VGG-16 para classificação binária médica
model = vgg16(num_classes=1, in_channels=3, dropout=0.5)

# 2. Teste com um lote fictício de Raio-X
xray_input = torch.randn(2, 3, 224, 224)
output_logits = model(xray_input)

print("Saída da VGG-16:", output_logits.shape)  # torch.Size([2, 1])
```
