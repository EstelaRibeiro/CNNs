"""
=============================================================================
Pipeline Didático de Treinamento e Avaliação (main.py)
=============================================================================

Este script orquestra todo o ciclo de vida do treinamento de modelos convolucionais
e vision transformers aplicados à classificação binária de Raio-X de Tórax:

Etapas:
1. Configuração do ambiente e inicialização de sementes (reprodutibilidade).
2. Carregamento e particionamento dos dados médicos (prevenção de vazamento por paciente).
3. Compensação do desbalanceamento de classes com `pos_weight` na perda BCEWithLogitsLoss.
4. Instanciação flexível do modelo:
   - Modo Didático: escrito diretamente como classe em `models/<modelo>.py`.
   - Modo timm: backbones pré-treinados no ImageNet com transferência de aprendizado.
5. Loop de treinamento com backpropagation.
6. Avaliação em conjunto de teste cego com métricas clínicas: Loss, Acurácia, AUC-ROC e F1-Score.
7. Salvamento do melhor ponto de verificação (*checkpoint*).
"""

import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import config
from utils import create_folders, prepare_data
from models.factory import get_model, list_available_models


# =============================================================================
# 1. DISPOSITIVO DE EXECUÇÃO (GPU CUDA vs CPU) E REPRODUTIBILIDADE
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("🧠 PROJETO: EVOLUÇÃO DAS ARQUITETURAS DE CNN EM RAIO-X")
print("=" * 65)
print(f"Dispositivo de processamento: {device}")
print(f"Versão do PyTorch:            {torch.__version__}")

if torch.cuda.is_available():
    print(f"Versão CUDA:                  {torch.version.cuda}")
    print(f"Placa de Vídeo (GPU):         {torch.cuda.get_device_name(0)}")
    torch.cuda.manual_seed(config.RANDOM_SEED)
else:
    print("Aviso: CUDA indisponível. O treinamento será executado em CPU.")

# Fixação de sementes para garantir resultados reproduzíveis
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

# Cria a pasta de saída para salvar os modelos
create_folders(config.OUTPUT_DIR)


# =============================================================================
# 2. PREPARAÇÃO DOS DADOS MÉDICOS (Prevenção de Vazamento por Paciente)
# =============================================================================
train_loader, test_loader, train_df, test_df = prepare_data(
    csv_path=config.CSV_DIR,
    img_dir=config.DATASET_DIR,
    img_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    seed=config.RANDOM_SEED
)

# Cálculo dinâmico do peso da classe positiva (doença):
# Em datasets médicos, casos normais (0) costumam ser mais frequentes que patologias (1).
# O pos_weight amplifica a perda gerada pela classe minoritária:
class_counts = train_df['label'].value_counts()
neg_count = class_counts.get(0, 1)
pos_count = class_counts.get(1, 1)
pos_weight_value = neg_count / pos_count
pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

print(f"Fator de balanceamento de classes (pos_weight): {pos_weight_value:.2f}")


# =============================================================================
# 3. CRIAÇÃO E INSTANCIAÇÃO DO MODELO
# =============================================================================
print("-" * 65)
print(f"ARQUITETURA SELECIONADA: {config.MODEL_NAME.upper()}")
print(f"Modo de criação:         {'Biblioteca timm' if config.USE_TIMM else 'Classe didática artesanal (models/)'}")
print(f"Pesos pré-treinados:     {config.PRETRAINED}")
print("-" * 65)

# -----------------------------------------------------------------------------
# Método A (Via Factory Flexível):
#   model = get_model(config.MODEL_NAME, use_timm=config.USE_TIMM, pretrained=config.PRETRAINED)
#
# Método B (Importação Direta da Classe do Arquivo):
#   from models.alexnet import AlexNet
#   from models.resnet import ResNet18
#   from models.densenet import DenseNet, densenet121
#   from models.vgg import VGG, vgg16
#   model = ResNet18(num_classes=1, in_channels=3)
# -----------------------------------------------------------------------------

model = get_model(
    model_name=config.MODEL_NAME,
    num_classes=config.NUM_CLASSES,
    in_channels=config.IN_CHANNELS,
    use_timm=config.USE_TIMM,
    pretrained=config.PRETRAINED
)
model.to(device)

# Contagem de parâmetros para discussão de eficiência computacional
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total de parâmetros do modelo:       {total_params:,}")
print(f"Parâmetros treináveis:               {trainable_params:,}\n")


# =============================================================================
# 4. FUNÇÃO DE PERDA, OTIMIZADOR E TAXA DE APRENDIZADO
# =============================================================================
# BCEWithLogitsLoss: combina uma camada Sigmoid implícita com Binary Cross-Entropy
# É numericamente muito mais estável do que aplicar Sigmoid seguido de BCELoss.
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

# Otimizador AdamW: gradiente adaptativo com decaimento de peso desvinculado (Weight Decay)
optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-2)

# Scheduler: reduz o learning rate pela metade se a perda de validação estagnar por 5 épocas
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)


# =============================================================================
# 5. FUNÇÕES DE TREINAMENTO E VALIDAÇÃO
# =============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Executa uma época completa de treinamento (Forward + Backward + Passo do Otimizador).
    """
    model.train()  # Ativa comportamento de treino (Dropout ativo, BatchNorm atualiza médias)
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)

        # 1. Zera os gradientes acumulados da iteração anterior
        optimizer.zero_grad()

        # 2. Forward pass: calcula os logits da rede
        outputs = model(images)

        # 3. Calcula a perda (Loss)
        loss = criterion(outputs, labels)

        # 4. Backward pass: calcula gradientes via regra da cadeia
        loss.backward()

        # 5. Passo do otimizador: atualiza os pesos da rede (w = w - lr * grad)
        optimizer.step()

        running_loss += loss.item()

        # Imprime o progresso dentro da época
        print(
            f"Época [{epoch+1:02d}/{config.EPOCHS:02d}] "
            f"Batch [{batch_idx:03d}/{total_batches:03d}] "
            f"Loss atual: {running_loss / batch_idx:.4f}",
            end='\r'
        )

    return running_loss / total_batches


def validate(model, dataloader, criterion, device):
    """
    Avalia o modelo no conjunto de teste com métricas clínicas (AUC-ROC, F1 e Acurácia).
    """
    model.eval()  # Desativa Dropout e congela estatísticas de BatchNorm
    running_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    # Desativa cálculo de gradientes para economizar memória VRAM e acelerar inferência
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Converte logits para probabilidades entre 0 e 1 via Sigmoid
            probs = torch.sigmoid(outputs).cpu().numpy().ravel()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().ravel())

    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)

    # Tratamento caso o batch não contenha amostras de ambas as classes
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_auc = 0.5

    val_f1 = f1_score(all_labels, all_preds, zero_division=0)

    return val_loss, val_acc, val_auc, val_f1


# =============================================================================
# 6. LOOP PRINCIPAL DE TREINAMENTO E SALVAMENTO DO MELHOR MODELO
# =============================================================================
best_val_loss = float('inf')  # Inicializado com infinito para capturar o menor valor
checkpoint_filename = f"{config.MODEL_NAME}_{'timm' if config.USE_TIMM else 'didatico'}_best.pth"
best_model_path = Path(config.OUTPUT_DIR) / checkpoint_filename

print("\n" + "=" * 65)
print("🚀 INICIANDO LOOP DE TREINAMENTO")
print("=" * 65)

for epoch in range(config.EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
    val_loss, val_acc, val_auc, val_f1 = validate(model, test_loader, criterion, device)

    # Salva o modelo caso a perda de validação atinja um novo mínimo histórico
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_name': config.MODEL_NAME,
            'state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_f1': val_f1
        }, best_model_path)
        salvo_msg = f" ⭐ [Novo melhor modelo salvo em: {best_model_path.name}]"
    else:
        salvo_msg = ""

    print(
        f"Época [{epoch+1:02d}/{config.EPOCHS:02d}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val AUC: {val_auc:.4f} | "
        f"Val F1: {val_f1:.4f}"
        f"{salvo_msg}"
    )

    # Atualiza a taxa de aprendizado conforme a curva de validação
    scheduler.step(val_loss)

print("\n" + "=" * 65)
print(f" Treinamento concluído com sucesso!")
print(f"Melhor Val Loss: {best_val_loss:.4f}")
print(f"Arquivo do melhor modelo: {best_model_path}")
print("=" * 65)
