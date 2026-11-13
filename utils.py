"""
=============================================================================
Utilitários de Pré-processamento e Prevenção de Vazamento (utils.py)
=============================================================================

Este módulo contém funções essenciais para a integridade científica de modelos
de inteligência artificial aplicados à medicina:

Conceito Clínico Fundamental: Vazamento de Dados por Paciente (Patient Leakage)
-------------------------------------------------------------------------------
Um mesmo paciente costuma realizar múltiplos exames de Raio-X ao longo do tempo.
Se dividirmos as imagens aleatoriamente (random split), exames do mesmo indivíduo
podem cair simultaneamente no conjunto de TREINO e no conjunto de TESTE.
Quando isso acontece:
1. O modelo memoriza características anatômicas únicas do indivíduo (formato de costela,
   marcapassos, altura da clavícula) em vez de aprender os sinais da doença.
2. A acurácia parece artificialmente excelente (ex: 98%), mas despenca no mundo real!

Solução Implementada:
Usamos `StratifiedGroupKFold` agrupando pelo identificador do paciente (`COPA_PACI_ID`).
Isso garante que um paciente e todas as suas radiografias fiquem estritamente
ou no treino ou no teste, preservando o rigor científico.
"""

import os
from typing import Tuple
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from generator import ChestXRayDataset


def create_folders(path: str) -> None:
    """Cria o diretório de destino se ele ainda não existir."""
    Path(path).mkdir(parents=True, exist_ok=True)


def split_train_test(
    csv_path: str,
    test_size: float = 0.2,
    seed: int = 13
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o conjunto de dados em Treino (80%) e Teste Cego (20%),
    garantindo que exames do mesmo paciente NUNCA vazem entre as partições.

    Args:
        csv_path (str): Caminho do arquivo CSV de metadados.
        test_size (float): Proporção do conjunto de teste (padrão: 0.2).
        seed (int): Semente de aleatoriedade para reprodução dos experimentos.

    Returns:
        df_train (pd.DataFrame): Dados de treinamento.
        df_test (pd.DataFrame): Dados de teste.
    """
    df = pd.read_csv(csv_path)

    # Garante a presença das colunas essenciais: imagem, rótulo diagnóstico e id do paciente
    colunas_obrigatorias = ['saved_filename', 'label', 'COPA_PACI_ID']
    df = df.dropna(subset=colunas_obrigatorias)
    df['label'] = df['label'].astype(int)

    n_splits = max(2, int(1.0 / test_size))
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X = df['saved_filename'].values
    y = df['label'].values
    groups = df['COPA_PACI_ID'].values

    # Pega a primeira partição para treino e teste
    train_idx, test_idx = next(sgk.split(X, y, groups))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    return df_train, df_test


def prepare_kfold_data(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 13
) -> pd.DataFrame:
    """
    Atribui folds de validação cruzada ao DataFrame de treino
    respeitando os grupos de pacientes.
    """
    df = df.copy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X = df['saved_filename'].values
    y = df['label'].values
    groups = df['COPA_PACI_ID'].values

    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(X, y, groups)):
        df.loc[df.index[val_idx], 'fold'] = fold

    return df


def prepare_data(
    csv_path: str,
    img_dir: str,
    img_size: Tuple[int, int],
    batch_size: int,
    seed: int
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de preparação:
    1. Divide dados em treino e teste sem vazamento por paciente.
    2. Imprime estatísticas e distribuição das classes.
    3. Instancia os Datasets e DataLoaders do PyTorch.

    Returns:
        (train_loader, test_loader, train_df, test_df)
    """
    train_df, test_df = split_train_test(csv_path, test_size=0.2, seed=seed)

    print("=" * 60)
    print("📊 RELATÓRIO DE PREPARAÇÃO DOS DADOS MÉDICOS")
    print(f"Resolução das imagens: {img_size}")
    print(f"Total no conjunto de Treino: {len(train_df)} radiografias")
    print(f"Total no conjunto de Teste:  {len(test_df)} radiografias")
    print(f"Distribuição de classes no Treino: {train_df['label'].value_counts().to_dict()}")
    print(f"Distribuição de classes no Teste:  {test_df['label'].value_counts().to_dict()}")
    print("=" * 60 + "\n")

    # Datasets
    train_dataset = ChestXRayDataset(
        df=train_df,
        img_dir=img_dir,
        img_size=img_size,
        augment=True  # Augmentation ativo apenas no treino
    )

    test_dataset = ChestXRayDataset(
        df=test_df,
        img_dir=img_dir,
        img_size=img_size,
        augment=False
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, train_df, test_df
