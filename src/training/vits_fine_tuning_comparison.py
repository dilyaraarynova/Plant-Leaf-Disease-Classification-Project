"""
Module for Comparison of Transfer Learning Strategies on ViTs for Plant Disease Classification
Supports: DINOv2, ViT
"""

import os

from nbconvert import export
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class Config:
    """Configurations for all experiments"""

    # Model configurations
    MODELS = {
        'dinov2_vits14': {
            'name': 'vit_small_patch14_dinov2.lvd142m',
            'reason': 'Smallest DINOv2 model for less overfitting',
            'image_size': 224,
            'num_blocks': 12,
            'partial_unfreeze': 3,
        },
        'vit_base': {
            'name': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            'reason': 'Baseline ViT',
            'image_size': 224,
            'num_blocks': 12,
            'partial_unfreeze': 3,
        }
    }

    # Transfer learning strategies
    STRATEGIES = ['feature_extraction', 'partial_finetuning', 'full_finetuning']

    # Dataset structure: CLASS_NAME/SPLIT/images
    DATA_ROOT = Path('./splitted_dataset')  # Root folder with class folders
    OUTPUT_DIR = Path('./experiments')

    # Training HPs
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5
    WARMUP_EPOCHS = 3
    GRADIENT_CLIP = 1.0

    # Learning rates
    LR_BACKBONE = 1e-5
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 0.01

    # Augmentation
    USE_ADVANCED_AUG = False

    # Loss
    USE_CLASS_WEIGHTS = False
    USE_FOCAL_LOSS = False

    # Reproducibility
    SEED = 42

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Utility functions
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(image_size: int, is_train: bool = True) -> transforms.Compose:
    """
    Get data transforms based on training/validation mode

    Args:
        image_size: Target image size
        is_train: Whether this is for training (with augmentation)

    Returns:
        Composed transforms
    """
    if is_train:
        if Config.USE_ADVANCED_AUG:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.RandAugment(num_ops=2, magnitude=9),  # turned on by config option
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


class PlantDiseaseDataset(Dataset):
    """
    Class for Dataset with CLASS/SPLIT/images structure

    Example structure:
        splitted_dataset/
            Pepper__bell___Bacterial_spot/
                train/
                    image1.jpg
                validation/
                    image2.jpg
                test/
                    image3.jpg
    """

    def __init__(self, root_dir: Path, split: str, transform=None):
        """
        Args:
            root_dir: Root directory containing class folders
            split: One of ['train', 'validation', 'test']
            transform: Optional transform to apply
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Collect all images and labels
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # Iterate through class folders
        class_folders = sorted([f for f in self.root_dir.iterdir() if f.is_dir()])

        for class_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            self.classes.append(class_name)
            self.class_to_idx[class_name] = class_idx

            # Look for split folder within class folder
            split_folder = class_folder / split

            if not split_folder.exists():
                print(f"Warning: {split} folder not found in {class_name}, skipping...")
                continue

            # Collect images from split folder
            for img_path in split_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split} split!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def compute_class_weights(dataset: PlantDiseaseDataset) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets

    Args:
        dataset: PlantDiseaseDataset

    Returns:
        Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = [label for _, label in dataset.samples]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance (was not used)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ModelBuilder:
    """Module to build and configure models for different transfer learning strategies"""

    @staticmethod
    def build_model(model_key: str, num_classes: int, strategy: str) -> nn.Module:
        """
        Build a model with given strategy

        Args:
            model_key: Key from Config.MODELS
            num_classes: Number of output classes
            strategy: One of ['feature_extraction', 'partial_finetuning', 'full_finetuning']

        Returns:
            Configured model
        """
        model_config = Config.MODELS[model_key]
        model_name = model_config['name']

        # Load pre-trained model
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes,
                                  img_size=model_config['image_size'])

        print(f"\n{'=' * 60}")
        print(f"Building {model_key} with {strategy} strategy")
        print(f"{'=' * 60}")

        if strategy == 'feature_extraction':
            # Freeze all layers except classifier head
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze classifier head
            ModelBuilder._unfreeze_head(model, model_key)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Strategy: Feature Extraction")
            print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        elif strategy == 'partial_finetuning':
            # Freeze all layers first
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze last N blocks + head
            ModelBuilder._unfreeze_partial(model, model_key, model_config['partial_unfreeze'])

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Strategy: Partial Fine-tuning (last {model_config['partial_unfreeze']} blocks)")
            print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        elif strategy == 'full_finetuning':
            # All layers are trainable
            for param in model.parameters():
                param.requires_grad = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Strategy: Full Fine-tuning")
            print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        print(f"{'=' * 60}\n")
        return model

    @staticmethod
    def _unfreeze_head(model: nn.Module, model_key: str):
        """Unfreeze classifier head based on model architecture"""
        if 'dinov2' in model_key or 'vit' in model_key:
            # ViT-based models: head is typically 'head' or 'fc'
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
        elif 'swin' in model_key:
            # Swin: head is typically 'head'
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True

    @staticmethod
    def _unfreeze_partial(model: nn.Module, model_key: str, num_blocks: int):
        """Unfreeze last N blocks + head"""
        if 'dinov2' in model_key or 'vit' in model_key:
            # ViT architecture: blocks are in model.blocks
            total_blocks = Config.MODELS[model_key]['num_blocks']
            start_block = total_blocks - num_blocks

            # Unfreeze last N blocks
            for i in range(start_block, total_blocks):
                block_name = f'blocks.{i}'
                for name, param in model.named_parameters():
                    if block_name in name:
                        param.requires_grad = True

            # Unfreeze head
            ModelBuilder._unfreeze_head(model, model_key)

        else:
            for name, param in model.named_parameters():
                if 'layers.3' in name or 'head' in name:
                    param.requires_grad = True


class DataManager:
    """Manage data loading and preprocessing for CLASS/SPLIT/images structure"""

    def __init__(self, data_root: Path, image_size: int, batch_size: int, num_workers: int):
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = None
        self.class_names = None

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare train, validation, and test dataloaders

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Transforms
        train_transform = get_transforms(self.image_size, is_train=True)
        val_transform = get_transforms(self.image_size, is_train=False)

        # Create datasets with CLASS/SPLIT/images structure
        self.train_dataset = PlantDiseaseDataset(
            self.data_root,
            split='train',
            transform=train_transform
        )

        self.val_dataset = PlantDiseaseDataset(
            self.data_root,
            split='validation',
            transform=val_transform
        )

        self.test_dataset = PlantDiseaseDataset(
            self.data_root,
            split='test',
            transform=val_transform
        )

        self.num_classes = len(self.train_dataset.classes)
        self.class_names = self.train_dataset.classes

        print(f"\nDataset Statistics:")
        print(f"{'=' * 10}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

        # Class distribution
        train_labels = [label for _, label in self.train_dataset.samples]
        class_counts = np.bincount(train_labels)
        print(f"\nClass distribution (train):")
        for i, (class_name, count) in enumerate(zip(self.class_names, class_counts)):
            print(f"  {class_name}: {count}")
        print(f"{'=' * 60}\n")

        # Dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset"""
        return compute_class_weights(self.train_dataset)


class Trainer:
    """Training module for transfer learning experiments"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            device: torch.device,
            experiment_dir: Path,
            model_name: str,
            strategy: str
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.experiment_dir = experiment_dir
        self.model_name = model_name
        self.strategy = strategy

        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0

        # time tracking
        self.training_time = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'lr': []
        }

        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)

            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(all_labels)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        return val_loss, val_acc, val_f1

    def train(self, epochs: int, early_stopping_patience: int, warmup_epochs: int):
        """
        Main training loop

        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
            warmup_epochs: Number of warmup epochs for learning rate
        """
        print(f"\nStarting training: {self.model_name} - {self.strategy}")
        print(f"{'=' * 60}\n")

        # Start training timer
        training_start_time = time.time()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 60)

            # Learning rate warmup
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * warmup_factor

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_f1 = self.validate()

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)

            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {100 * train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {100 * val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth')
                print(f"New best validation accuracy: {100 * val_acc:.2f}%")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            print()

        # Calculate total training time
        self.training_time = time.time() - training_start_time

        print(f"\nTraining completed!")
        print(f"Best Val Acc: {100 * self.best_val_acc:.2f}%")
        print(f"Best Val F1: {self.best_val_f1:.4f}")
        print(f"Total Training Time: {self.training_time:.2f} seconds ({self.training_time / 60:.2f} minutes)")

        # Save training history
        self.save_history()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'history': self.history,
            'training_time': self.training_time
        }
        torch.save(checkpoint, self.experiment_dir / filename)

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.experiment_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)


class Evaluator:
    """Model evaluation module"""

    def __init__(
            self,
            model: nn.Module,
            test_loader: DataLoader,
            class_names: List[str],
            device: torch.device,
            experiment_dir: Path
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.experiment_dir = experiment_dir

    def evaluate(self) -> Dict:
        """
        Evaluation on test set

        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating on test set...")
        print(f"{'=' * 60}")

        self.model.eval()
        all_preds = []
        all_labels = []

        # Track inference time
        inference_start_time = time.time()

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Calculate inference time
        inference_time = time.time() - inference_start_time
        samples_per_second = len(all_labels) / inference_time

        # Compute metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        # Precision and Recall (macro averaged)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # Precision and Recall (weighted averaged)
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        # Confusion matrix formulation
        cm = confusion_matrix(all_labels, all_preds)

        # Per-class metrics
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

        # Find worst performing classes
        worst_indices = np.argsort(per_class_f1)[:3]
        worst_classes = [(self.class_names[i], per_class_f1[i]) for i in worst_indices]


        print(f"\nTest Accuracy: {100 * accuracy:.2f}%")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        print(f"\nInference Time: {inference_time:.2f} seconds")
        print(f"Inference Speed: {samples_per_second:.2f} samples/second")

        print(f"\n3 worst performing classes:")
        for class_name, f1 in worst_classes:
            print(f"  {class_name}: F1 = {f1:.4f}")

        # Save confusion matrix
        self.plot_confusion_matrix(cm)

        # Save detailed report
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'inference_time': inference_time,
            'samples_per_second': samples_per_second,
            'per_class_metrics': {
                class_name: {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
                for class_name in self.class_names
            },
            'confusion_matrix': cm.tolist(),
            'worst_classes': worst_classes
        }

        with open(self.experiment_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to {self.experiment_dir}")
        print(f"{'=' * 60}\n")

        return results

    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()


class ExperimentRunner:
    """Run complete transfer learning experiments"""

    def __init__(self, config: Config):
        self.config = config
        set_seed(config.SEED)

        # Prepare data
        self.data_manager = DataManager(
            data_root=config.DATA_ROOT,
            image_size=224,  # Will be updated per model
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )

    def run_experiment(self, model_key: str, strategy: str):
        """
        Run a single experiment. Use for testing the work of existing code.

        Args:
            model_key: Key from Config.MODELS
            strategy: One of Config.STRATEGIES
        """
        model_config = Config.MODELS[model_key]

        # Update image size for this model
        self.data_manager.image_size = model_config['image_size']

        # Prepare data
        train_loader, val_loader, test_loader = self.data_manager.prepare_data()
        num_classes = self.data_manager.num_classes

        # Build model
        model = ModelBuilder.build_model(model_key, num_classes, strategy)

        # Loss function
        if Config.USE_FOCAL_LOSS:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif Config.USE_CLASS_WEIGHTS:
            class_weights = self.data_manager.get_class_weights().to(Config.DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer with differentiated learning rates
        if strategy == 'feature_extraction':
            # Only head parameters
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=Config.LR_HEAD,
                weight_decay=Config.WEIGHT_DECAY
            )
        else:
            # Differentiated learning rates for backbone and head
            backbone_params = []
            head_params = []

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'head' in name or 'fc' in name:
                        head_params.append(param)
                    else:
                        backbone_params.append(param)

            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': Config.LR_BACKBONE},
                {'params': head_params, 'lr': Config.LR_HEAD}
            ], weight_decay=Config.WEIGHT_DECAY)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.EPOCHS,
            eta_min=1e-6
        )

        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{model_key}_{strategy}_{timestamp}"
        experiment_dir = Config.OUTPUT_DIR / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_dict = {
            'model': model_key,
            'strategy': strategy,
            'model_config': model_config,
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'lr_backbone': Config.LR_BACKBONE,
            'lr_head': Config.LR_HEAD,
            'weight_decay': Config.WEIGHT_DECAY,
            'use_class_weights': Config.USE_CLASS_WEIGHTS,
            'use_focal_loss': Config.USE_FOCAL_LOSS,
            'seed': Config.SEED,
            'num_classes': num_classes,
            'class_names': self.data_manager.class_names
        }

        with open(experiment_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=4)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=Config.DEVICE,
            experiment_dir=experiment_dir,
            model_name=model_key,
            strategy=strategy
        )

        # Train
        trainer.train(
            epochs=Config.EPOCHS,
            early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
            warmup_epochs=Config.WARMUP_EPOCHS
        )

        # Load best model for evaluation
        checkpoint = torch.load(experiment_dir / 'best_model.pth', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get training time from checkpoint
        training_time = checkpoint.get('training_time', 0.0)

        # Evaluate
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            class_names=self.data_manager.class_names,
            device=Config.DEVICE,
            experiment_dir=experiment_dir
        )

        results = evaluator.evaluate()

        # Add training time to results
        results['training_time'] = training_time

        return results, experiment_dir

    def run_all_experiments(self, models: List[str] = None, strategies: List[str] = None):
        """
        Run all outlined experiments

        Args:
            models: List of model keys to test (None = all)
            strategies: List of strategies to test (None = all)
        """
        if models is None:
            models = list(Config.MODELS.keys())

        if strategies is None:
            strategies = Config.STRATEGIES

        results_summary = []

        for model_key in models:
            for strategy in strategies:
                print(f"\n{'=' * 80}")
                print(f"EXPERIMENT: {model_key} - {strategy}")
                print(f"{'=' * 80}\n")

                try:
                    results, exp_dir = self.run_experiment(model_key, strategy)

                    results_summary.append({
                        'model': model_key,
                        'strategy': strategy,
                        'test_accuracy': results['accuracy'],
                        'precision_macro': results['precision_macro'],
                        'recall_macro': results['recall_macro'],
                        'macro_f1': results['macro_f1'],
                        'weighted_f1': results['weighted_f1'],
                        'training_time': results['training_time'],
                        'inference_time': results['inference_time'],
                        'samples_per_second': results['samples_per_second'],
                        'experiment_dir': str(exp_dir)
                    })

                except Exception as e:
                    print(f"Error in experiment {model_key} - {strategy}: {str(e)}")
                    import traceback
                    traceback.print_exc()

        # Save summary
        summary_df = self._create_summary_table(results_summary)
        summary_path = Config.OUTPUT_DIR / 'experiments_summary.json'

        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=4)

        print(f"\n{'=' * 80}")
        print("ALL EXPERIMENTS COMPLETED")
        print(f"{'=' * 80}\n")
        print(summary_df)
        print(f"\nSummary saved to {summary_path}")

        return results_summary

    def _create_summary_table(self, results: List[Dict]) -> str:
        """Create a formatted summary table"""
        if not results:
            return "No results to display"

        # Create header - Performance Metrics
        table = f"\n{'=' * 140}\n"
        table += f"{'EXPERIMENT RESULTS SUMMARY':^140}\n"
        table += f"{'=' * 140}\n\n"

        table += "PERFORMANCE METRICS:\n"
        table += f"{'Model':<20} {'Strategy':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Macro F1':>10} {'Weighted F1':>12}\n"
        table += "-" * 140 + "\n"

        # Add performance rows
        for result in results:
            table += f"{result['model']:<20} {result['strategy']:<20} "
            table += f"{result['test_accuracy'] * 100:>9.2f}% "
            table += f"{result['precision_macro']:>9.4f} "
            table += f"{result['recall_macro']:>9.4f} "
            table += f"{result['macro_f1']:>9.4f} "
            table += f"{result['weighted_f1']:>11.4f}\n"

        # Add timing section
        table += "\n" + "=" * 140 + "\n"
        table += "TIMING METRICS:\n"
        table += f"{'Model':<20} {'Strategy':<20} {'Training Time':>20} {'Inference Time':>20} {'Samples/Second':>20}\n"
        table += "-" * 140 + "\n"

        for result in results:
            train_min = result['training_time'] / 60
            table += f"{result['model']:<20} {result['strategy']:<20} "
            table += f"{train_min:>17.2f} min "
            table += f"{result['inference_time']:>17.2f} sec "
            table += f"{result['samples_per_second']:>17.2f}\n"

        table += "=" * 140 + "\n"

        return table


def main():
    """Main execution function"""

    # Print configuration
    print("\n" + "=" * 80)
    print("TRANSFER LEARNING PIPELINE FOR PLANT DISEASE CLASSIFICATION")
    print("=" * 80)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Data Structure: CLASS/SPLIT/images")
    print(f"Data Root: {Config.DATA_ROOT}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")
    print(f"Learning Rate (Backbone): {Config.LR_BACKBONE}")
    print(f"Learning Rate (Head): {Config.LR_HEAD}")
    print(f"Using Class Weights: {Config.USE_CLASS_WEIGHTS}")
    print(f"Using Focal Loss: {Config.USE_FOCAL_LOSS}")
    print("=" * 80 + "\n")

    # Initialize experiment runner
    runner = ExperimentRunner(Config)

    print("Running all DINOv2 experiments...")
    results_summary = runner.run_all_experiments(
        models=['dinov2_vits14', 'vit_base'],
        strategies=['feature_extraction', 'partial_finetuning', 'full_finetuning']
    )

    return results_summary

# Utility functions

def quick_experiment(model: str, strategy: str):
    """
    Quick launcher for single experiments

    Usage:
        python pipeline.py
        >>> quick_experiment('dinov2_vitb14', 'partial_finetuning')

    Args:
        model: One of ['dinov2_vitb14', 'dinov2_vitl14', 'vit_base', 'swin_base', 'swin_tiny']
        strategy: One of ['feature_extraction', 'partial_finetuning', 'full_finetuning']
    """
    runner = ExperimentRunner(Config)
    results, exp_dir = runner.run_experiment(model, strategy)
    print(f"\nExperiment completed! Results saved to: {exp_dir}")
    return results


def load_trained_model(experiment_dir: Path, device: torch.device = None):
    """
    Load a trained model from an experiment directory

    Args:
        experiment_dir: Path to experiment directory (e.g., './experiments/dinov2_vitb14_partial_finetuning_20241115_143022/')
        device: Device to load model on (default: Config.DEVICE)

    Returns:
        model: Loaded model ready for inference
        config: Experiment configuration
        history: Training history

    Example:
        >>> model, config, history = load_trained_model('./experiments/dinov2_vitb14_partial_finetuning_20241115_143022/')
        >>> # Use for inference
        >>> predictions = model(images)
    """
    if device is None:
        device = Config.DEVICE

    experiment_dir = Path(experiment_dir)

    # Load configuration
    with open(experiment_dir / 'config.json', 'r') as f:
        config = json.load(f)

    print(f"\nLoading model from: {experiment_dir}")
    print(f"Model: {config['model']}")
    print(f"Strategy: {config['strategy']}")
    print(f"Number of classes: {config['num_classes']}")

    # Build model architecture
    model = ModelBuilder.build_model(
        model_key=config['model'],
        num_classes=config['num_classes'],
        strategy=config['strategy']
    )

    # Load trained weights
    checkpoint = torch.load(experiment_dir / 'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Best validation accuracy: {checkpoint['best_val_acc'] * 100:.2f}%")
    print(f"Best validation F1: {checkpoint['best_val_f1']:.4f}\n")

    return model, config, checkpoint['history']


def predict_image(model, image_path: Path, class_names: List[str], transform=None, device=None):
    """
    Predict disease class for a single image

    Args:
        model: Trained model
        image_path: Path to image file
        class_names: List of class names
        transform: Transform to apply (if None, uses default validation transform)
        device: Device to run inference on

    Returns:
        predicted_class: Predicted class name
        confidence: Confidence score (0-1)
        top_k_predictions: List of (class_name, confidence) tuples

    Example:
        >>> model, config, _ = load_trained_model('./experiments/...')
        >>> pred_class, conf, top_k = predict_image(
        >>>     model,
        >>>     './test_image.jpg',
        >>>     config['class_names']
        >>> )
        >>> print(f"Prediction: {pred_class} ({conf*100:.2f}%)")
    """
    if device is None:
        device = Config.DEVICE

    if transform is None:
        transform = get_transforms(224, is_train=False)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()

    # Get top-5 predictions
    top_k_probs, top_k_indices = torch.topk(probabilities[0], min(5, len(class_names)))
    top_k_predictions = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_k_indices, top_k_probs)
    ]

    return predicted_class, confidence_score, top_k_predictions


def continue_training(experiment_dir: Path, additional_epochs: int = 10):
    """
    Continue training a model from a checkpoint

    Args:
        experiment_dir: Path to experiment directory
        additional_epochs: Number of additional epochs to train

    Returns:
        Updated results and experiment directory

    Example:
        >>> # Train for 10 more epochs
        >>> results, exp_dir = continue_training(
        >>>     './experiments/dinov2_vitb14_partial_finetuning_20241115_143022/',
        >>>     additional_epochs=10
        >>> )
    """
    experiment_dir = Path(experiment_dir)

    # Load configuration
    with open(experiment_dir / 'config.json', 'r') as f:
        config = json.load(f)

    print(f"\nContinuing training from: {experiment_dir}")
    print(f"Additional epochs: {additional_epochs}\n")

    # Prepare data
    data_manager = DataManager(
        data_root=Config.DATA_ROOT,
        image_size=Config.MODELS[config['model']]['image_size'],
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    train_loader, val_loader, test_loader = data_manager.prepare_data()

    # Build model
    model = ModelBuilder.build_model(
        model_key=config['model'],
        num_classes=config['num_classes'],
        strategy=config['strategy']
    )

    # Load checkpoint
    checkpoint = torch.load(experiment_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Setup training
    if config['use_focal_loss']:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    elif config['use_class_weights']:
        class_weights = data_manager.get_class_weights().to(Config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if config['strategy'] == 'feature_extraction':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['lr_head'],
            weight_decay=config['weight_decay']
        )
    else:
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'head' in name or 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': config['lr_backbone']},
            {'params': head_params, 'lr': config['lr_head']}
        ], weight_decay=config['weight_decay'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=additional_epochs,
        eta_min=1e-6
    )

    # Create new experiment directory for continued training
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_exp_dir = experiment_dir.parent / f"{experiment_dir.name}_continued_{timestamp}"
    new_exp_dir.mkdir(parents=True, exist_ok=True)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        experiment_dir=new_exp_dir,
        model_name=config['model'],
        strategy=config['strategy']
    )

    # Restore history
    trainer.history = checkpoint['history']
    trainer.best_val_acc = checkpoint['best_val_acc']
    trainer.best_val_f1 = checkpoint['best_val_f1']

    # Continue training
    trainer.train(
        epochs=additional_epochs,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
        warmup_epochs=0  # No warmup for continued training
    )

    # Evaluate
    checkpoint_new = torch.load(new_exp_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint_new['model_state_dict'])

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        class_names=data_manager.class_names,
        device=Config.DEVICE,
        experiment_dir=new_exp_dir
    )

    results = evaluator.evaluate()

    print(f"\nContinued training completed!")
    print(f"Results saved to: {new_exp_dir}\n")

    return results, new_exp_dir


if __name__ == "__main__":
    main()
