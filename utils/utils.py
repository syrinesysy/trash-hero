"""
Trash Hero - Utility Functions
Fonctions réutilisables pour le projet de tri intelligent des déchets
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ============================================================================
# DATA UTILITIES
# ============================================================================

def organize_dataset(source_dir, dest_dir, split_ratios={'train': 0.7, 'valid': 0.15, 'test': 0.15}):
    """
    Organise le dataset en train/valid/test
    
    Args:
        source_dir: Dossier source contenant les classes
        dest_dir: Dossier destination
        split_ratios: Ratios de split (doit sommer à 1.0)
    """
    import os
    import random
    from pathlib import Path
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Créer les dossiers
    for split in split_ratios.keys():
        (dest_path / split).mkdir(parents=True, exist_ok=True)
    
    # Pour chaque classe
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        # Mélanger les images
        random.shuffle(images)
        
        # Calculer les splits
        n_total = len(images)
        n_train = int(n_total * split_ratios['train'])
        n_valid = int(n_total * split_ratios['valid'])
        
        train_imgs = images[:n_train]
        valid_imgs = images[n_train:n_train + n_valid]
        test_imgs = images[n_train + n_valid:]
        
        # Copier les images
        for split, imgs in zip(['train', 'valid', 'test'], 
                                [train_imgs, valid_imgs, test_imgs]):
            split_class_dir = dest_path / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img in imgs:
                shutil.copy(img, split_class_dir / img.name)
        
        print(f"{class_name}: {len(train_imgs)} train, {len(valid_imgs)} valid, {len(test_imgs)} test")


def count_images_per_class(data_dir):
    """Compte le nombre d'images par classe"""
    from pathlib import Path
    
    data_path = Path(data_dir)
    class_counts = {}
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            n_images = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            class_counts[class_dir.name] = n_images
    
    return class_counts


# ============================================================================
# MODEL UTILITIES
# ============================================================================

class MyConvBlock(nn.Module):
    """Bloc convolutif réutilisable"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_p=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )
    
    def forward(self, x):
        return self.model(x)


def get_batch_accuracy(output, y, N):
    """Calcule l'accuracy d'un batch"""
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def save_model(model, path):
    """Sauvegarde un modèle"""
    torch.save(model.state_dict(), path)
    print(f"✅ Modèle sauvegardé: {path}")


def load_model(model, path, device):
    """Charge un modèle"""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"✅ Modèle chargé: {path}")
    return model


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_class_distribution(class_counts, title="Distribution des classes"):
    """Affiche la distribution des classes"""
    plt.figure(figsize=(12, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    colors = plt.cm.Set3(range(len(classes)))
    plt.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Classes de déchets', fontsize=12, fontweight='bold')
    plt.ylabel('Nombre d\'images', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (c, count) in enumerate(zip(classes, counts)):
        plt.text(i, count + 0.01 * max(counts), str(count), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_sample_images(dataset, class_names, n_samples=5):
    """Affiche des exemples d'images par classe"""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(15, 3*n_classes))
    
    for class_idx in range(n_classes):
        # Trouver les indices de cette classe
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
        
        # Sélectionner n_samples images aléatoires
        selected_indices = np.random.choice(class_indices, 
                                           min(n_samples, len(class_indices)), 
                                           replace=False)
        
        for col, idx in enumerate(selected_indices):
            img, _ = dataset[idx]
            
            # Convertir tensor en numpy et transposer
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            
            # Dénormaliser si nécessaire
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            axes[class_idx, col].imshow(img_np)
            axes[class_idx, col].axis('off')
            
            if col == 0:
                axes[class_idx, col].set_ylabel(class_names[class_idx], 
                                               fontsize=12, fontweight='bold')
    
    plt.suptitle('Exemples d\'images par classe', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, train_accs, valid_losses, valid_accs):
    """Affiche l'historique d'entraînement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, valid_losses, 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, valid_accs, 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Affiche la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Afficher aussi le rapport de classification
    print("\n📊 Classification Report:")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))


def show_predictions(model, dataset, class_names, device, n_samples=6):
    """Affiche des prédictions du modèle"""
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            img, true_label = dataset[idx]
            
            # Prédiction
            img_batch = img.unsqueeze(0).to(device)
            output = model(img_batch)
            pred_label = output.argmax(dim=1).item()
            
            # Afficher l'image
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            ax.imshow(img_np)
            ax.axis('off')
            
            # Titre avec couleur selon si correct ou non
            color = 'green' if pred_label == true_label else 'red'
            ax.set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}',
                        color=color, fontweight='bold', fontsize=11)
    
    plt.suptitle('Exemples de Prédictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# WASTE CLASSIFICATION HELPERS
# ============================================================================

WASTE_BINS = {
    'cardboard': {
        'bin': '📦 Recyclable (Papier/Carton)',
        'color': '#4CAF50',
        'icon': '♻️',
        'description': 'Carton propre et sec'
    },
    'glass': {
        'bin': '🍾 Recyclable (Verre)',
        'color': '#2196F3',
        'icon': '♻️',
        'description': 'Bouteilles et bocaux en verre'
    },
    'metal': {
        'bin': '🥫 Recyclable (Métal)',
        'color': '#9E9E9E',
        'icon': '♻️',
        'description': 'Canettes et conserves en métal'
    },
    'paper': {
        'bin': '📄 Recyclable (Papier)',
        'color': '#FFC107',
        'icon': '♻️',
        'description': 'Papier propre et non gras'
    },
    'plastic': {
        'bin': '🍶 Recyclable (Plastique)',
        'color': '#FF9800',
        'icon': '♻️',
        'description': 'Bouteilles et emballages plastiques'
    },
    'trash': {
        'bin': '🗑️ Déchets non recyclables',
        'color': '#F44336',
        'icon': '❌',
        'description': 'Déchets résiduels'
    }
}


def get_bin_recommendation(class_name):
    """Retourne la recommandation de poubelle pour une classe"""
    return WASTE_BINS.get(class_name.lower(), WASTE_BINS['trash'])


def display_classification_result(predicted_class, confidence):
    """Affiche le résultat de classification avec style"""
    bin_info = get_bin_recommendation(predicted_class)
    
    print("\n" + "="*60)
    print(f"{bin_info['icon']} RÉSULTAT DE CLASSIFICATION {bin_info['icon']}")
    print("="*60)
    print(f"\n🎯 Déchet détecté: {predicted_class.upper()}")
    print(f"📊 Confiance: {confidence:.1%}")
    print(f"\n{bin_info['bin']}")
    print(f"💡 {bin_info['description']}")
    print("\n" + "="*60 + "\n")