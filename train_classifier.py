"""
train_classifier.py — Entraîner un classificateur CNN sur les crops de la pipeline.

Workflow recommandé :
  1. Lancer la pipeline en mode "high recall" pour générer des crops.
  2. Corriger manuellement les labels (déplacer les crops mal classés).
  3. Lancer ce script pour entraîner un CNN.
  4. Utiliser le modèle sauvegardé dans la pipeline.

Structure attendue du dataset :
    dataset/
    ├── Component_Rect/
    │   ├── crop_001.png
    │   ├── crop_002.png
    │   └── ...
    ├── Circle_Component/
    │   ├── crop_010.png
    │   └── ...
    ├── Open_Component/
    │   └── ...
    ├── Busbar_Power/
    │   └── ...
    ├── False_Positive/    ← créer ce dossier pour les erreurs
    │   └── ...
    └── Ground_Symbol/     ← créer pour les symboles de terre
        └── ...

Usage :
    # Entraîner
    python train_classifier.py --data dataset/crops --epochs 30 --output model.pth

    # Évaluer
    python train_classifier.py --data dataset/crops --evaluate --model model.pth

    # Prédire sur une image
    python train_classifier.py --predict crop.png --model model.pth
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from typing import Optional, List, Tuple
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# ══════════════════════════════════════════════════════════════
#                   DATA PREPARATION
# ══════════════════════════════════════════════════════════════

def get_transforms(img_size: int = 64, augment: bool = True):
    """
    Transformations pour les crops de composants.
    
    Les crops sont petits (30-200px typique) et souvent en N&B.
    On resize à img_size × img_size et on normalise.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


def load_dataset(data_dir: str, img_size: int = 64, val_ratio: float = 0.2):
    """
    Charge le dataset depuis un dossier structuré par catégorie.
    
    Returns:
        (train_loader, val_loader, class_names, num_classes)
    """
    transform_train = get_transforms(img_size, augment=True)
    transform_val = get_transforms(img_size, augment=False)

    # Charger tout le dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=transform_train)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"📊 Dataset: {len(full_dataset)} images, {num_classes} classes")
    for i, name in enumerate(class_names):
        count = sum(1 for _, label in full_dataset.samples if label == i)
        print(f"   {name}: {count} images")

    # Split train/val
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Override val transform (no augmentation)
    val_dataset = datasets.ImageFolder(data_dir, transform=transform_val)
    val_indices = val_set.indices
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_subset, batch_size=32, shuffle=False, num_workers=2,
    )

    print(f"   Train: {train_size}, Val: {val_size}")
    return train_loader, val_loader, class_names, num_classes


# ══════════════════════════════════════════════════════════════
#                   MODEL DEFINITION
# ══════════════════════════════════════════════════════════════

def create_model(num_classes: int, model_name: str = "resnet18", pretrained: bool = True):
    """
    Crée un modèle CNN.
    
    Options :
      - "resnet18"  : 11M params, bon compromis (recommandé)
      - "mobilenet" : 3.4M params, plus léger
      - "simple"    : CNN custom ~200K params, très rapide
    """
    if model_name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Remplacer la dernière couche fully-connected
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes,
        )

    elif model_name == "simple":
        # CNN custom léger — bon si < 500 images
        model = nn.Sequential(
            # Conv block 1: 3 → 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv block 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv block 3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            # Classifier
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# ══════════════════════════════════════════════════════════════
#                   TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(
    data_dir: str,
    output_path: str = "component_classifier.pth",
    model_name: str = "resnet18",
    epochs: int = 30,
    lr: float = 0.001,
    img_size: int = 64,
    patience: int = 7,
):
    """
    Entraîne le classificateur et sauvegarde le modèle.
    
    Args:
        data_dir: Dossier dataset (sous-dossiers = catégories).
        output_path: Chemin de sortie pour le modèle .pth.
        model_name: Architecture ("resnet18", "mobilenet", "simple").
        epochs: Nombre d'epochs max.
        lr: Learning rate initial.
        img_size: Taille des images d'entrée.
        patience: Nombre d'epochs sans amélioration avant arrêt.
    """
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"🖥️  Device: {device}")

    # Data
    train_loader, val_loader, class_names, num_classes = load_dataset(
        data_dir, img_size,
    )

    # Model
    model = create_model(num_classes, model_name, pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model: {model_name} ({trainable_params:,} trainable / {total_params:,} total params)")

    # Optimizer + scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0.0
    no_improve_count = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_loss:>8.4f} | {val_acc:>7.1%}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0

            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "model_name": model_name,
                "img_size": img_size,
                "best_val_acc": best_val_acc,
            }, output_path)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n⏹ Early stopping (pas d'amélioration depuis {patience} epochs)")
                break

    print(f"\n✅ Meilleur modèle sauvegardé → {output_path}")
    print(f"   Val accuracy: {best_val_acc:.1%}")
    print(f"   Classes: {class_names}")

    return output_path


# ══════════════════════════════════════════════════════════════
#                   INFERENCE
# ══════════════════════════════════════════════════════════════

class ComponentClassifier:
    """
    Classificateur CNN pour valider/corriger les catégories.
    
    Utilisable en post-pipeline pour remplacer ou compléter
    l'arbre de décision géométrique.
    
    Usage :
        classifier = ComponentClassifier("model.pth")
        category, confidence = classifier.predict("crop.png")
        
        # Sur une liste de crops
        results = classifier.predict_batch(["crop1.png", "crop2.png"])
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        self.class_names = checkpoint["class_names"]
        self.num_classes = checkpoint["num_classes"]
        self.img_size = checkpoint.get("img_size", 64)
        model_name = checkpoint.get("model_name", "resnet18")

        if device is None:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = create_model(self.num_classes, model_name, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = get_transforms(self.img_size, augment=False)

    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Prédit la catégorie d'une image.
        
        Returns:
            (category_name, confidence)
        """
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(1)

        return self.class_names[pred_idx.item()], confidence.item()

    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        """Prédit les catégories pour une liste d'images."""
        return [self.predict(p) for p in image_paths]

    def predict_tensor(self, tensor: torch.Tensor) -> Tuple[str, float]:
        """Prédit à partir d'un tensor déjà transformé."""
        tensor = tensor.unsqueeze(0).to(self.device) if tensor.dim() == 3 else tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(1)

        return self.class_names[pred_idx.item()], confidence.item()


# ══════════════════════════════════════════════════════════════
#                   DATASET PREPARATION HELPER
# ══════════════════════════════════════════════════════════════

def prepare_dataset_from_pipeline(
    pdf_paths: List[str],
    output_dir: str = "dataset/crops",
    pages: Optional[List[int]] = None,
):
    """
    Génère un dataset de crops à partir de la pipeline.
    
    Crée un dossier par catégorie avec les crops PNG.
    Tu devras ensuite manuellement :
      1. Créer un dossier "False_Positive/" et y déplacer les erreurs.
      2. Créer des dossiers pour les catégories manquantes (ex: "Ground_Symbol/").
      3. Déplacer les crops mal classés dans le bon dossier.
    
    Args:
        pdf_paths: Liste de PDFs à traiter.
        output_dir: Dossier de sortie pour le dataset.
        pages: Pages à traiter (None = toutes).
    """
    from .pipeline import HybridPipeline
    from .config import PipelineConfig

    os.makedirs(output_dir, exist_ok=True)

    config = PipelineConfig()

    for pdf_path in pdf_paths:
        print(f"📄 Processing {pdf_path}...")
        pipeline = HybridPipeline(pdf_path, config)
        pipeline.run(
            output_dir=output_dir,
            pages=pages,
            export_crops_flag=True,
            export_json=False,
            export_yolo=False,
        )
        pipeline.doc.close()

    print(f"\n✅ Dataset généré → {output_dir}")
    print(f"\n📝 Prochaines étapes :")
    print(f"   1. Ouvrir {output_dir} et vérifier les crops")
    print(f"   2. Créer un dossier 'False_Positive/' pour les erreurs")
    print(f"   3. Déplacer les crops mal classés dans le bon dossier")
    print(f"   4. Optionnel: créer 'Ground_Symbol/', 'Arrow/', etc.")
    print(f"   5. Lancer: python train_classifier.py --data {output_dir} --epochs 30")


# ══════════════════════════════════════════════════════════════
#                          CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Entraîner un classificateur CNN pour les composants électriques.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── Train ──
    train_parser = subparsers.add_parser("train", help="Entraîner le modèle")
    train_parser.add_argument(
        "--data", "-d", required=True,
        help="Dossier dataset (sous-dossiers = catégories).",
    )
    train_parser.add_argument(
        "--output", "-o", default="component_classifier.pth",
        help="Chemin de sortie pour le modèle.",
    )
    train_parser.add_argument(
        "--model", "-m", default="resnet18",
        choices=["resnet18", "mobilenet", "simple"],
        help="Architecture du modèle (défaut: resnet18).",
    )
    train_parser.add_argument(
        "--epochs", "-e", type=int, default=30,
        help="Nombre d'epochs (défaut: 30).",
    )
    train_parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (défaut: 0.001).",
    )
    train_parser.add_argument(
        "--img-size", type=int, default=64,
        help="Taille des images d'entrée (défaut: 64).",
    )

    # ── Predict ──
    predict_parser = subparsers.add_parser("predict", help="Prédire une image")
    predict_parser.add_argument("image", help="Chemin de l'image.")
    predict_parser.add_argument(
        "--model", "-m", required=True,
        help="Chemin du modèle .pth.",
    )

    # ── Prepare ──
    prep_parser = subparsers.add_parser(
        "prepare", help="Générer un dataset depuis des PDFs",
    )
    prep_parser.add_argument(
        "input", nargs="+",
        help="Chemin(s) PDF ou dossier de PDFs.",
    )
    prep_parser.add_argument(
        "--output", "-o", default="dataset/crops",
        help="Dossier de sortie pour le dataset.",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            data_dir=args.data,
            output_path=args.output,
            model_name=args.model,
            epochs=args.epochs,
            lr=args.lr,
            img_size=args.img_size,
        )

    elif args.command == "predict":
        classifier = ComponentClassifier(args.model)
        category, confidence = classifier.predict(args.image)
        print(f"📋 {args.image}")
        print(f"   Catégorie: {category}")
        print(f"   Confiance: {confidence:.1%}")

    elif args.command == "prepare":
        pdf_paths = []
        for inp in args.input:
            p = Path(inp)
            if p.is_dir():
                pdf_paths.extend(sorted(str(f) for f in p.glob("*.pdf")))
            else:
                pdf_paths.append(str(p))
        prepare_dataset_from_pipeline(pdf_paths, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
