# Solinfitec Solix - Disease Detection & Outbreak Prediction

Deteccao de doencas em culturas e previsao de surtos para o robo Solix da Solinfitec. O modelo usa tres fontes de dados: imagens de folhas (Swin Transformer), series temporais IoT/climaticas (Temporal Transformer) e coordenadas GPS (Spatial MLP), fundidos por cross-attention.

## Arquitetura

```
Imagem (Solix) --> Swin Transformer --> Features Visuais (768-d) ---+
                                                                    |
Sensores IoT ----> Temporal Transformer --> Features Temporais -----+-- Cross-Attention
                                                                    |   + Gated Fusion (640-d)
Geo/GPS ---------> Spatial MLP ----------> Features Espaciais ------+        |
                                                                      +------+------+
                                                                      v      v      v
                                                                  Classif. Surto  Severidade
                                                                  Doenca   7 dias  Estagio
                                                                      |
                                                                      v
                                                               Sistema de Alertas
                                                               + Dashboard Streamlit
```

- Swin Classifier: `swin_tiny_patch4_window7_224` (pretrained ImageNet), cabeca `768 -> 256 -> N classes`, FocalLoss
- Temporal Encoder: Transformer Encoder, 4 camadas, 8 heads, d_model=128, saida 256-d
- Spatial MLP: lat/lon/elevacao `3 -> 64 -> 128`
- Fusao: Cross-Attention (queries=temporal, keys/values=visual+spatial) + Gated Fusion -> 640-d
- 3 cabecas de saida: classificacao de doenca, regressao de surto (7 dias), severidade ordinal (4 niveis)
- Multi-task loss com ponderacao por incerteza (Kendall et al. 2018)

## Resultados

### Swin Classifier - PlantVillage, 15 classes

~20.600 imagens, split 70/15/15:

| Metrica | Resultado |
|---------|-----------|
| Val F1-Score | 99.65% |
| Val Accuracy | 99.61% |
| Melhor epoch | 27/50 (early stopping) |

### Fusion Model - Multi-modal

Imagem + IoT + geo, treinamento multi-task:

| Metrica | Resultado |
|---------|-----------|
| Disease Accuracy | 99.64% |
| Outbreak MAE | 0.0072 |
| Melhor epoch | 60/100 (early stopping) |

Treinamento em 2 fases:
1. Epochs 0-9: stages 0,1 do Swin congelados, lr=1e-4, CosineAnnealingWarmRestarts
2. Epoch 10+: descongelamento total, lr reduzido 10x

## Datasets

### PlantVillage

15 classes de doencas em folhas (~20.600 imagens):

| Classe | Amostras |
|--------|----------|
| Pepper_bell_Bacterial_spot | ~997 |
| Pepper_bell_healthy | ~1,478 |
| Potato_Early_blight | ~1,000 |
| Potato_healthy | ~152 |
| Potato_Late_blight | ~1,000 |
| Tomato_Bacterial_spot | ~2,127 |
| Tomato_Early_blight | ~1,000 |
| Tomato_healthy | ~1,591 |
| Tomato_Late_blight | ~1,909 |
| Tomato_Leaf_Mold | ~952 |
| Tomato_Septoria_leaf_spot | ~1,771 |
| Tomato_Spider_mites | ~1,676 |
| Tomato_Target_Spot | ~1,404 |
| Tomato_mosaic_virus | ~373 |
| Tomato_YellowLeaf_Curl_Virus | ~3,209 |

Classes com menos de 500 amostras recebem augmentacao 3x. `WeightedRandomSampler` equilibra o treinamento.

### DiaMOS Plant (pera)

3.505 imagens de folhas de pera com severidade anotada (Zenodo 5557313, ~13GB). 4 classes: healthy, spot, curl, slug. Severidade de 0 a 4, mapeada para 0-3.

Download: `python scripts/download_diamos.py`

### Grape Disease (IoT)

10.000 registros IoT de vinhedos (`data/grape_disease/`). Usado para calibrar o simulador IoT -- ajusta as distribuicoes de temperatura, umidade e doenca para corresponder a dados reais.

## Estrutura do projeto

```
solinfitec-solix-disease-detection/
|
|-- configs/
|   |-- config.yaml               # Config PlantVillage (15 classes)
|   +-- config_diamos.yaml         # Config DiaMOS (4 classes, severidade real)
|
|-- src/
|   |-- data/
|   |   |-- dataset.py             # PlantVillageDataset com split estratificado
|   |   |-- diamos_dataset.py      # DiaMOS Plant dataset (severidade 0-4 -> 0-3)
|   |   |-- datamodule.py          # DataLoaders com WeightedRandomSampler
|   |   |-- preprocessing.py       # Scan de duplicatas, deteccao de corrompidas, mean/std
|   |   |-- iot_simulator.py       # Simulacao IoT (SEIR, AR(1)) + calibracao
|   |   |-- weather_client.py      # Cliente Open-Meteo API com cache
|   |   |-- multimodal_dataset.py  # Dataset multi-modal (imagem + IoT + geo)
|   |   +-- mqtt_interface.py      # Interface MQTT para Solix
|   |
|   |-- features/
|   |   |-- augmentation.py        # Albumentations + MixUp/CutMix
|   |   |-- temporal_features.py   # Medias moveis, lag features, graus-dia
|   |   |-- spatial_features.py    # Encoding geoespacial
|   |   +-- disease_rules.py       # Base de conhecimento epidemiologico
|   |
|   |-- models/
|   |   |-- swin_classifier.py     # Swin Transformer com freeze/unfreeze
|   |   |-- temporal_encoder.py    # Transformer Encoder para series temporais
|   |   |-- fusion_model.py        # Cross-Attention + Gated Fusion
|   |   |-- prediction_heads.py    # Classificacao, regressao surto, severidade
|   |   +-- losses.py              # FocalLoss, LabelSmoothing, MultiTaskLoss
|   |
|   |-- utils/
|   |   |-- config.py              # ConfigManager com dataclasses
|   |   |-- seed.py                # Reproducibilidade (torch, numpy, CUDA)
|   |   |-- metrics.py             # Precision, Recall, F1, AUROC, mAP, confusion matrix
|   |   |-- callbacks.py           # EarlyStopping, ModelCheckpoint, LRScheduler
|   |   |-- alert_system.py        # Gerador de alertas (LOW/MEDIUM/HIGH/CRITICAL)
|   |   |-- onnx_export.py         # Export ONNX (opset 14) + quantizacao
|   |   +-- logging_utils.py       # Logging estruturado
|   |
|   +-- visualization/
|       |-- gradcam.py             # Grad-CAM para Swin (Stage 3)
|       |-- evaluation_plots.py    # Confusion matrix, ROC, PR curves
|       |-- dataset_plots.py       # Distribuicao de classes, grids
|       |-- training_plots.py      # Loss/accuracy curves, LR schedule
|       +-- outbreak_plots.py      # Timeline de risco, heat maps
|
|-- notebooks/
|   |-- 01_exploratory/            # EDA do dataset e dados IoT
|   |-- 02_preprocessing/          # Limpeza e preparacao
|   |-- 03_modeling/               # Treinamento Swin e Fusao
|   +-- 04_evaluation/             # Relatorios de classificacao e surto
|
|-- scripts/
|   |-- download_diamos.py         # Download do DiaMOS (Zenodo)
|   +-- register_model.py         # Registro de checkpoints no MLflow
|
|-- tests/                         # 69 testes (pytest)
|   |-- conftest.py
|   |-- test_dataset.py
|   |-- test_augmentation.py
|   |-- test_swin_classifier.py
|   |-- test_metrics.py
|   |-- test_iot_simulator.py
|   |-- test_fusion_model.py
|   |-- test_alert_system.py
|   +-- test_onnx_export.py
|
|-- app_alerts.py                  # Dashboard Streamlit
|-- train_classifier.py            # Treino do Swin (Fase 2)
|-- train_fusion.py                # Treino da fusao (Fase 4)
|-- tune_classifier.py             # HPO Optuna para Swin
|-- tune_fusion.py                 # HPO Optuna para fusao
|-- evaluate.py                    # Avaliacao do classificador
|-- evaluate_fusion.py             # Avaliacao da fusao
|-- predict.py                     # Inferencia (PyTorch / ONNX Runtime)
|-- export_model.py                # Export ONNX para edge
+-- requirements.txt
```

## Simulacao IoT

Nao temos dados IoT suficientes, entao ha um simulador que gera series temporais sinteticas:

- Temperatura: base sazonal + ciclo diurno + ruido AR(1)
- Umidade: anticorrelacionada com temperatura
- Umidade do solo: dirigida por chuva com decaimento exponencial
- Vento: distribuicao log-normal sazonal
- Chuva: gamma zero-inflada
- Prevalencia de doenca: modelo SEIR (Susceptible-Exposed-Infectious-Recovered), beta varia com temperatura e umidade

O simulador pode ser calibrado com dados reais:

```python
from src.data.iot_simulator import IoTSimulator

simulator = IoTSimulator()
simulator.calibrate_from_real_data("data/grape_disease/grape_disease_dataset.csv")
```

Isso ajusta media, variancia e correlacoes para corresponder aos dados de vinhedos. Saida em Parquet: `data/processed/iot_simulated/field_{id}.parquet`

## Dashboard e alertas

```bash
streamlit run app_alerts.py
```

O dashboard permite upload de imagens de folhas, mostra Grad-CAM sobre as regioes afetadas, timeline de risco de surto (7 dias) e mapa de severidade.

Os alertas sao gerados em JSON:

| Nivel | Limiar | Acao |
|-------|--------|------|
| LOW | < 0.2 | Monitoramento regular |
| MEDIUM | 0.2 - 0.5 | Aumentar frequencia de inspecao |
| HIGH | 0.5 - 0.75 | Tratamento direcionado |
| CRITICAL | > 0.9 | Intervencao de emergencia |

Cada alerta traz: doenca + confianca, risco de surto por dia (7 dias), severidade, acoes recomendadas e Grad-CAM.

## Como usar

### Instalacao

```bash
pip install -r requirements.txt
```

### Treino do classificador (PlantVillage)

```bash
python train_classifier.py
```

Config em `configs/config.yaml`. Usa AdamW (weight_decay=0.05), CosineAnnealingWarmRestarts (T_0=10, T_mult=2), FocalLoss (gamma=2.0) com pesos por classe, EarlyStopping (patience=10, monitor=val_f1), MixUp (alpha=0.2) e CutMix (alpha=1.0).

### Treino do modelo de fusao

```bash
# PlantVillage (15 classes, severidade simulada)
python train_fusion.py

# DiaMOS Plant (4 classes, severidade real)
python train_fusion.py --dataset diamos --config configs/config_diamos.yaml

# Retomar treino interrompido
python train_fusion.py --resume
```

### Avaliacao

```bash
python evaluate.py            # classificador Swin
python evaluate_fusion.py     # modelo de fusao
```

### Dashboard

```bash
streamlit run app_alerts.py
```

### Inferencia

```bash
python predict.py --image path/to/leaf.jpg --backend pytorch
python predict.py --image path/to/leaf.jpg --backend onnx
```

### Export ONNX (edge)

```bash
python export_model.py
```

ONNX opset 14, validacao de saida (diff < 1e-5). Quantizacao INT8/FP16 para Jetson Xavier.

### Testes

```bash
pytest tests/ -v
```

69 testes cobrindo dataset, augmentacao, modelos, metricas, simulador IoT, fusao, alertas e export ONNX.

## MLOps

### Experiment tracking (MLflow)

Os scripts de treino logam automaticamente no MLflow: parametros, metricas por epoca e o modelo final via `mlflow.pytorch.log_model()` (com schema de serving). O modelo e registrado no Model Registry.

```bash
# Visualizar experimentos
mlflow ui --port 5000

# Registrar checkpoints existentes
python scripts/register_model.py --checkpoint models/checkpoints/best_swin_classifier.pth --model-name SwinClassifier
python scripts/register_model.py --checkpoint models/checkpoints/best_fusion_model.pth --model-name MultiModalFusion
```

### Versionamento de dados (DVC)

Datasets versionados com DVC. Os arquivos `.dvc` rastreiam hashes dos dados sem commitar os arquivos no git.

```bash
dvc status          # verificar estado dos dados
dvc push            # enviar para remote (quando configurado)
```

### Hyperparameter optimization (Optuna)

Tuning automatizado com Optuna, integrado ao MLflow para tracking de trials.

```bash
# Tuning do classificador Swin
python tune_classifier.py --n-trials 20

# Tuning do modelo de fusao
python tune_fusion.py --n-trials 20
```

Os estudos ficam em `optuna.db` (SQLite) e podem ser retomados entre sessoes.

## Tecnologias

| Categoria | Tecnologia |
|-----------|-----------|
| Backbone visual | Swin Transformer (timm) |
| Encoder temporal | Transformer Encoder (PyTorch) |
| Fusao | Cross-Attention + Gated Fusion |
| Augmentacao | Albumentations, MixUp, CutMix |
| Explicabilidade | Grad-CAM (pytorch-grad-cam) |
| Dados IoT | Simulacao SEIR + Open-Meteo API |
| Dashboard | Streamlit + Plotly |
| Comunicacao | MQTT (paho-mqtt) |
| Export | ONNX (opset 14), ONNX Runtime |
| Testes | pytest |
| Tracking | MLflow + TensorBoard |
| Versionamento de dados | DVC |
| HPO | Optuna + MLflow callback |

## Fases do projeto

1. Fundacao e pipeline de dados - dataset, augmentacao, preprocessing ✅
2. Classificador Swin Transformer - fine-tuning com freeze/unfreeze em 2 fases ✅
3. Simulacao IoT e modelo temporal - simulador SEIR + Temporal Transformer ✅
4. Fusao multi-modal - Cross-Attention + 3 cabecas de predicao ✅
5. Alertas e dashboard - geracao de alertas + dashboard Streamlit ✅
6. Integracao de dados reais - DiaMOS Plant + Grape Disease + calibracao IoT ✅
7. MLOps - MLflow tracking/registry, DVC, Optuna HPO ✅
8. Deploy edge - export ONNX + inferencia para Jetson Xavier
