#  Trash Hero - Classification Intelligente des Déchets

**Projet de Deep Learning - Classification d'images**

##  Équipe
- **TRABELSI Syrine** 
- **SAMMOUDA Cyrine** 

##  Description
Système de classification automatique de déchets utilisant du Deep Learning (PyTorch) pour identifier 6 catégories : carton, verre, métal, papier, plastique et déchets non recyclables.

##  Objectifs
- Classification automatique de 6 types de déchets
- Accuracy > 85% sur le test set
- Modèle déployable sur mobile/web

##  Dataset
- **Source** : [Kaggle Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- **Classes** : 6 (cardboard, glass, metal, paper, plastic, trash)
- **Images** : ~2500 images
- **Split** : 70% train / 15% valid / 15% test

##  Architecture
1. **Baseline CNN** : 3 blocs convolutifs (~85% accuracy)
2. **Transfer Learning** : MobileNetV2 pré-entraîné (>90% accuracy)
3. **Data Augmentation** : Amélioration de la robustesse
4. **Optimisation** : Quantization et pruning pour déploiement

##  Structure du Projet
```
trash-hero/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_data_augmentation.ipynb
│   ├── 05_model_optimization.ipynb
│   └── 06_deployment.ipynb
├── utils/
│   ├── __init__.py
│   └── utils.py
├── models/
│   └── (modèles sauvegardés)
├── data/
│   ├── raw/
│   ├── processed/
│   └── test_images/
├── requirements.txt
└── README.md
```

##  Installation

### Prérequis
- Python 3.11+
- CUDA 11.8 (optionnel, pour GPU)

### Installation des dépendances
```bash
# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installer PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Installer les autres dépendances
pip install -r requirements.txt
```

## Notebooks

###  Data Preparation
- Exploration du dataset
- Split train/valid/test
- Création des DataLoaders

###  Baseline CNN
- Modèle CNN custom
- Entraînement de base
- Évaluation initiale

###  Transfer Learning
- MobileNetV2 pré-entraîné
- Fine-tuning
- Amélioration des performances

### Data Augmentation
- Augmentation avancée
- Amélioration de la robustesse

### Model Optimization
- Quantization
- Pruning
- Réduction de la taille

### Deployment
- Export ONNX
- API Flask/FastAPI
- Interface de démonstration

## Résultats

| Modèle | Accuracy | Params | Taille |
|--------|----------|--------|--------|
| Baseline CNN | 85.3% | 2.1M | 8.4 MB |
| MobileNetV2 | 92.7% | 3.5M | 14 MB |
| MobileNet Quantized | 91.5% | 3.5M | 3.6 MB |

##  Utilisation

### Prédiction sur une image
```python
from utils import predict_waste

predicted_class, confidence = predict_waste(
    'path/to/image.jpg',
    model,
    transforms,
    class_names,
    device
)
```


## 📚 Ressources
- [Documentation PyTorch](https://pytorch.org/docs/)
- [Paper MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Dataset Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

##  Licence
Ce projet est réalisé dans un cadre académique.