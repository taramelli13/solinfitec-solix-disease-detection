# 🌱 Previsão de Doenças em Culturas - Visão Computacional

## 📋 Objetivo
Desenvolver modelo de deep learning para detecção automática de pragas e doenças em culturas estratégicas (cana-de-açúcar, citros, café) utilizando imagens de folhas e aplicar transfer learning em modelos YOLO/CNN.

## 🎯 Aplicação Solinfitec
- **Robô Solix**: Detecção em tempo real durante patrulhamento
- **Intervenção Precoce**: Redução de perdas com diagnóstico antecipado
- **Mapeamento**: Geração de mapas de calor de infestação

## 📊 Datasets Públicos

### 1. PlantVillage (Kaggle)
- **Link**: `https://www.kaggle.com/datasets/emmarex/plantdisease`
- **Conteúdo**: 54.000+ imagens de folhas
- **Classes**: 14 pragas/doenças em tomate, milho, soja
- **Uso**: Transfer learning para cana/citros

### 2. Embrapa Dataset (dados.gov.br)
- **Link**: `https://dados.gov.br`
- **Conteúdo**: Imagens de pragas em café e soja brasileiras
- **Uso**: Fine-tuning com dados nacionais

### 3. Crop Disease Dataset
- **Link**: `https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset`
- **Conteúdo**: 87.000 imagens RGB de folhas
- **Classes**: 38 categorias de plantas + doenças

## 🏗️ Estrutura do Projeto
```
01_Previsao_Doencas_Visao/
├── data/
│   ├── raw/              # Datasets originais (PlantVillage, Embrapa)
│   ├── processed/        # Imagens preprocessadas e augmentadas
│   └── external/         # Imagens coletadas manualmente
├── notebooks/
│   ├── 01_exploratory/   # EDA de imagens e distribuição classes
│   ├── 02_preprocessing/ # Augmentation, normalização, split
│   ├── 03_modeling/      # Treinamento YOLO/ResNet/EfficientNet
│   └── 04_evaluation/    # Métricas, confusion matrix, CAM
├── src/
│   ├── data/            # Scripts para download e limpeza
│   ├── features/        # Augmentation pipelines
│   ├── models/          # Arquiteturas customizadas
│   ├── visualization/   # Plot resultados e heatmaps
│   └── utils/           # Funções auxiliares
├── models/
│   ├── checkpoints/     # Modelos durante treinamento
│   └── final/           # Modelo final para deploy
├── reports/
│   ├── figures/         # Gráficos de performance
│   └── metrics/         # JSON com métricas (mAP, F1, etc)
├── configs/             # Hyperparâmetros e configs YOLO
├── tests/               # Testes unitários
├── logs/                # Logs de treinamento
├── requirements.txt
└── README.md
```

## 🎯 Métricas de Sucesso
- **mAP@0.5**: > 85% (detecção)
- **F1-Score**: > 90% (classificação)
- **Latência**: < 200ms (inferência em edge device)
- **Precisão**: > 92% (para implantação em produção)

## 🚀 Tecnologias
- **Frameworks**: PyTorch, YOLOv8, Ultralytics
- **Modelos**: YOLO, EfficientNet, ResNet, Vision Transformer
- **Tools**: Roboflow, Albumentations, Grad-CAM
- **Deploy**: ONNX, TensorRT (otimização para Jetson)

## 📝 Próximos Passos
1. Download e exploração dos datasets
2. Análise de desbalanceamento de classes
3. Implementação de data augmentation
4. Transfer learning com modelos pré-treinados
5. Fine-tuning em culturas brasileiras
6. Otimização para inferência em tempo real
7. Validação com imagens coletadas em campo

## 🌟 Diferenciais para Solinfitec
- Modelo específico para culturas brasileiras
- Inferência em edge (baixa latência)
- Explainabilidade com Grad-CAM
- Pipeline de retreino contínuo
