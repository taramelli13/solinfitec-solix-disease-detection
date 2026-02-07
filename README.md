# Solinfitec Solix - Disease Detection & Outbreak Prediction

Sistema multi-modal de deteccao de doencas em culturas e previsao de surtos para o robo Solix da Solinfitec, combinando visao computacional (Swin Transformer), dados IoT/climaticos (Temporal Transformer) e informacoes geoespaciais (Spatial MLP) com fusao por cross-attention.

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
```

**Componentes:**

- **Swin Classifier**: `swin_tiny_patch4_window7_224` (pretrained ImageNet) com cabeca customizada `768 -> 256 -> 15 classes`, FocalLoss
- **Temporal Encoder**: Transformer Encoder com 4 camadas, 8 heads, d_model=128, saida 256-d
- **Spatial MLP**: Encoding de lat/lon/elevacao `3 -> 64 -> 128`
- **Fusao**: Cross-Attention (queries=temporal, keys/values=visual+spatial) + Gated Fusion -> 640-d
- **3 Cabecas**: Classificacao de doenca (15 classes), regressao de surto (7 dias), severidade ordinal (4 niveis)
- **Multi-task Loss**: Ponderacao por incerteza (Kendall et al. 2018)

## Resultados do Treinamento (Swin Classifier)

Treinamento no dataset PlantVillage (~20.600 imagens, 15 classes) com split 70/15/15:

| Metrica | Resultado |
|---------|-----------|
| **Val F1-Score** | 99.69% |
| **Val Accuracy** | 99.61% |
| **Val Loss** | 0.0023 |
| **Melhor Epoch** | 9/50 |

Estrategia de treinamento em 2 fases:
1. **Epochs 0-9**: Stages 0,1 do Swin congelados, lr=1e-4, CosineAnnealingWarmRestarts
2. **Epoch 10+**: Descongelamento total, lr reduzido 10x, fine-tuning completo

## Dataset

**PlantVillage** - 15 classes de doencas em folhas:

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

Desbalanceamento tratado com `WeightedRandomSampler` e augmentacao extra (3x) para classes minoritarias (<500 amostras).

## Estrutura do Projeto

```
01_Previsao_Doencas_Visao/
|
|-- configs/
|   +-- config.yaml                # Configuracao centralizada (modelo, treino, IoT, alertas)
|
|-- src/
|   |-- data/
|   |   |-- dataset.py             # PlantVillageDataset com split estratificado
|   |   |-- datamodule.py          # DataLoaders com WeightedRandomSampler
|   |   |-- preprocessing.py       # Scan de duplicatas, deteccao de corrompidas, mean/std
|   |   |-- iot_simulator.py       # Simulacao IoT (SEIR, AR(1), gamma zero-inflada)
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
|   |   |-- fusion_model.py        # Multi-Modal Fusion (Cross-Attention + Gated)
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
|-- tests/                         # 69 testes unitarios (pytest)
|   |-- test_dataset.py
|   |-- test_augmentation.py
|   |-- test_swin_classifier.py
|   |-- test_metrics.py
|   |-- test_iot_simulator.py
|   |-- test_fusion_model.py
|   |-- test_alert_system.py
|   +-- test_onnx_export.py
|
|-- train_classifier.py            # Treinamento do Swin Classifier (Fase 2)
|-- train_fusion.py                # Treinamento do modelo de fusao (Fase 4)
|-- evaluate.py                    # Avaliacao no test set
|-- predict.py                     # Inferencia (PyTorch e ONNX Runtime)
|-- export_model.py                # Exportacao ONNX para edge
+-- requirements.txt
```

## Simulacao IoT

Como nao ha dados IoT reais disponiveis, o sistema inclui um simulador realista:

- **Temperatura**: Base sazonal + ciclo diurno + ruido AR(1)
- **Umidade**: Anticorrelacionada com temperatura
- **Umidade do solo**: Dirigida por chuva com decaimento exponencial
- **Vento**: Distribuicao log-normal sazonal
- **Chuva**: Distribuicao gamma zero-inflada
- **Prevalencia de doenca**: Modelo epidemiologico **SEIR** (Susceptible-Exposed-Infectious-Recovered) onde beta varia com temperatura e umidade

Saida em formato Parquet: `data/processed/iot_simulated/field_{id}.parquet`

## Sistema de Alertas

Gera alertas estruturados em JSON com base nas predicoes do modelo:

| Nivel | Limiar | Acao |
|-------|--------|------|
| LOW | < 0.2 | Monitoramento regular |
| MEDIUM | 0.2 - 0.5 | Aumentar frequencia de inspecao |
| HIGH | 0.5 - 0.75 | Tratamento direcionado imediato |
| CRITICAL | > 0.9 | Intervencao de emergencia |

Cada alerta inclui: doenca detectada + confianca, risco de surto por dia (7 dias), estagio de severidade, acoes recomendadas e imagem Grad-CAM.

## Como Usar

### Instalacao

```bash
pip install -r requirements.txt
```

### Treinamento do Classificador

```bash
python train_classifier.py
```

Configuracoes em `configs/config.yaml`. O treinamento usa:
- AdamW (weight_decay=0.05)
- CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- FocalLoss (gamma=2.0) com pesos por classe
- EarlyStopping (patience=10, monitor=val_f1)
- MixUp (alpha=0.2) + CutMix (alpha=1.0)

### Treinamento do Modelo de Fusao

```bash
python train_fusion.py
```

### Avaliacao

```bash
python evaluate.py
```

### Inferencia

```bash
python predict.py --image path/to/leaf.jpg --backend pytorch
python predict.py --image path/to/leaf.jpg --backend onnx
```

### Exportacao ONNX (Edge Deploy)

```bash
python export_model.py
```

Exporta para ONNX opset 14 com validacao de saida (diff < 1e-5). Suporte a quantizacao INT8/FP16 para Jetson Xavier.

### Testes

```bash
pytest tests/ -v
```

69 testes cobrindo dataset, augmentacao, modelos, metricas, simulador IoT, fusao, alertas e export ONNX.

## Tecnologias

| Categoria | Tecnologia |
|-----------|-----------|
| Backbone Visual | Swin Transformer (timm) |
| Encoder Temporal | Transformer Encoder (PyTorch) |
| Fusao | Cross-Attention + Gated Fusion |
| Augmentacao | Albumentations, MixUp, CutMix |
| Explicabilidade | Grad-CAM (pytorch-grad-cam) |
| Dados IoT | Simulacao SEIR + Open-Meteo API |
| Comunicacao | MQTT (paho-mqtt) |
| Export | ONNX (opset 14), ONNX Runtime |
| Testes | pytest |
| Tracking | TensorBoard |

## Fases do Projeto

1. **Fundacao e Pipeline de Dados** - Dataset, augmentacao, preprocessing
2. **Classificador Swin Transformer** - Fine-tuning com 2 fases de freeze/unfreeze
3. **Simulacao IoT e Modelo Temporal** - Simulador SEIR + Temporal Transformer
4. **Fusao Multi-Modal** - Cross-Attention + 3 cabecas de predicao
5. **Sistema de Alertas** - Geracao de alertas + visualizacoes
6. **Deploy Edge** - Export ONNX + inferencia otimizada para Jetson Xavier
