"""
Module for Transfer Learning Pipeline for Plant Disease Classification with Domain Generalization
Supports: DINOv2, ViT
Dataset Structure: CLASS/SPLIT/images (e.g., Pepper__bell___Bacterial_spot/train/image.jpg)

Augmentation techniques tested in this experiment:
1. Aggressive Data Augmentation
2. CutMix
3. Test-Time Augmentation (TTA)
4. AugMix
5. Label Smoothing
6. Strong Regularization
7. Multi-Scale Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import timm  # PyTorch Image Models - best library for pre-trained models
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
import random

warnings.filterwarnings('ignore')


class Config:
    """Centralized configuration for all experiments"""

    # Model configurations with reasoning
    MODELS = {
        'dinov2_vits14': {
            'name': 'vit_small_patch14_dinov2.lvd142m',
            'reason': 'Smallest DINOv2 model, less overfitting',
            'image_size': 224,  # Using 224 for consistency (DINOv2 can handle this)
            'num_blocks': 12,
            'partial_unfreeze': 3,  # Last 3 blocks (proportionally similar)
        },
        'vit_base': {
            'name': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            'reason': 'Supervised ImageNet-21k pre-training, strong baseline',
            'image_size': 224,
            'num_blocks': 12,
            'partial_unfreeze': 3,
        }
    }

    STRATEGIES = ['feature_extraction', 'partial_finetuning', 'full_finetuning']

    # Data paths - Dataset structure: CLASS_NAME/SPLIT/images
    DATA_ROOT = Path('./splitted_dataset')  # Root folder with class folders
    OUTPUT_DIR = Path('./experiments')

    # Training HPs
    BATCH_SIZE = 16  # Reduced because of larger DINOv2 image size (518x518)
    NUM_WORKERS = 4
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5
    WARMUP_EPOCHS = 3
    GRADIENT_CLIP = 1.0

    # Learning rates
    LR_BACKBONE = 1e-5
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 0.01


    AUGMENTATION_LEVEL = 'baseline'  # can be: 'baseline', 'aggressive', 'augmix'
    USE_CUTMIX = False
    CUTMIX_ALPHA = 1.0
    CUTMIX_PROB = 0.5
    LABEL_SMOOTHING = 0.0
    USE_TTA = False
    TTA_NUM_AUGMENTATIONS = 5

    USE_ADVANCED_AUG = False

    # Loss
    USE_CLASS_WEIGHTS = False
    USE_FOCAL_LOSS = False

    # Reproducibility
    SEED = 42

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class AugMixTransform:
    """
    Specifically designed for domain shift robustness
    AugMix: A Simple Data Augmentation Method
    Source: https://arxiv.org/abs/1912.02781
    """

    def __init__(self, severity=3, width=3, depth=2, alpha=1.0):
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

        # Augmentation operations
        self.augmentations = [
            self.autocontrast,
            self.equalize,
            self.posterize,
            self.rotate,
            self.solarize,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,
            self.color,
            self.contrast,
            self.brightness,
            self.sharpness
        ]

    def __call__(self, img):
        """Apply AugMix augmentation"""
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = np.zeros_like(np.array(img), dtype=np.float32)

        for i in range(self.width):
            image_aug = img.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)

            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.severity)

            mix += ws[i] * np.array(image_aug, dtype=np.float32)

        mixed = (1 - m) * np.array(img, dtype=np.float32) + m * mix
        return Image.fromarray(np.uint8(mixed))

    def int_parameter(self, level, maxval):
        return int(level * maxval / 10)

    def float_parameter(self, level, maxval):
        return float(level) * maxval / 10.

    def autocontrast(self, pil_img, level):
        return ImageOps.autocontrast(pil_img)

    def equalize(self, pil_img, level):
        return ImageOps.equalize(pil_img)

    def posterize(self, pil_img, level):
        level = self.int_parameter(level, 4)
        return ImageOps.posterize(pil_img, 4 - level)

    def rotate(self, pil_img, level):
        degrees = self.int_parameter(level, 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)

    def solarize(self, pil_img, level):
        level = self.int_parameter(level, 256)
        return ImageOps.solarize(pil_img, 256 - level)

    def shear_x(self, pil_img, level):
        level = self.float_parameter(level, 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)

    def shear_y(self, pil_img, level):
        level = self.float_parameter(level, 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)

    def translate_x(self, pil_img, level):
        level = self.int_parameter(level, pil_img.size[0] / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)

    def translate_y(self, pil_img, level):
        level = self.int_parameter(level, pil_img.size[1] / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)

    def color(self, pil_img, level):
        level = self.float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Color(pil_img).enhance(level)

    def contrast(self, pil_img, level):
        level = self.float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Contrast(pil_img).enhance(level)

    def brightness(self, pil_img, level):
        level = self.float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Brightness(pil_img).enhance(level)

    def sharpness(self, pil_img, level):
        level = self.float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Sharpness(pil_img).enhance(level)


def get_enhanced_transforms(image_size: int, is_train: bool = True, 
                           augmentation_level: str = 'baseline') -> transforms.Compose:
    """
    Get enhanced data transforms based on training/validation mode and augmentation level

    Args:
        image_size: Target image size
        is_train: Whether this is for training (with augmentation)
        augmentation_level: 'baseline', 'aggressive', or 'augmix'

    Returns:
        Composed transforms
    """
    # resize dimension should be slightly larger than target for cropping
    resize_dim = int(image_size * 1.14)
    
    if not is_train:
        # Validation/Test transforms - always use baseline
        return transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Training transforms based on augmentation level
    if augmentation_level == 'baseline':
        transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augmentation_level == 'aggressive':
        transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(image_size),

            # Imitate varied lighting conditions
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            ),

            # Imitate varying camera angles
            transforms.RandomRotation(45),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.7, 1.3),
                shear=15
            ),

            # Imitate varying image quality
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # MUST add after ToTensor
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
        ])
    
    elif augmentation_level == 'augmix':
        transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(image_size),
            AugMixTransform(severity=3, width=3, depth=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")

    return transform


def cutmix_data(images, labels, alpha=1.0):
    """
    CutMix: Regularization Strategy to Train Strong Classifiers
    Source: https://arxiv.org/abs/1905.04899

    Args:
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        alpha: Beta distribution parameter

    Returns:
        Mixed images, labels_a, labels_b, lambda
    """
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)

    # Random permutation
    index = torch.randperm(batch_size).to(images.device)

    # Get random bounding box
    W = images.size(2)
    H = images.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    labels_a = labels
    labels_b = labels[index]

    return images, labels_a, labels_b, lam


def cutmix_criterion(criterion, pred, labels_a, labels_b, lam):
    """
    CutMix loss calculation

    Args:
        criterion: Loss function
        pred: Model predictions
        labels_a: Original labels
        labels_b: Mixed labels
        lam: Mixing ratio

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) for robust predictions
    Applies multiple augmentations and averages predictions
    """

    def __init__(self, num_augmentations: int = 5, image_size: int = 224):
        self.num_augmentations = num_augmentations
        self.image_size = image_size

        # TTA transforms (lighter than training augmentation)
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.FiveCrop(image_size),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda tensors: torch.stack([
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t) 
                    for t in tensors
                ]))
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        ]

    def predict(self, model, image_path: str, device: torch.device) -> torch.Tensor:
        """
        Predict with TTA

        Args:
            model: Trained model
            image_path: Path to image
            device: Device to run on

        Returns:
            Averaged predictions
        """
        model.eval()
        image = Image.open(image_path).convert('RGB')

        predictions = []

        with torch.no_grad():
            for transform in self.tta_transforms[:self.num_augmentations]:
                augmented = transform(image)

                # Handle FiveCrop output
                if augmented.dim() == 4:  # FiveCrop returns (5, C, H, W)
                    augmented = augmented.to(device)
                    batch_pred = model(augmented)
                    predictions.append(batch_pred.mean(dim=0, keepdim=True))
                else:
                    augmented = augmented.unsqueeze(0).to(device)
                    pred = model(augmented)
                    predictions.append(pred)

        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        return F.softmax(avg_pred, dim=1)


class PlantDiseaseDataset(Dataset):
    """
    Class for dataset with CLASS/SPLIT/images structure

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
    # Count samples per class
    class_counts = torch.zeros(len(dataset.classes))

    for _, label in dataset.samples:
        class_counts[label] += 1

    # Compute weights (inverse frequency)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(dataset.classes)

    return class_weights


class DataManager:
    """Handles data loading and preprocessing"""

    def __init__(self, data_root: Path, image_size: int, batch_size: int, num_workers: int):
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """
        Prepare train, validation, and test dataloaders

        Returns:
            train_loader, val_loader, test_loader, class_names
        """
        # Get transforms based on config
        train_transform = get_enhanced_transforms(
            self.image_size, 
            is_train=True,
            augmentation_level=Config.AUGMENTATION_LEVEL
        )
        
        val_transform = get_enhanced_transforms(
            self.image_size, 
            is_train=False
        )

        # Create datasets
        train_dataset = PlantDiseaseDataset(self.data_root, 'train', transform=train_transform)
        val_dataset = PlantDiseaseDataset(self.data_root, 'validation', transform=val_transform)
        test_dataset = PlantDiseaseDataset(self.data_root, 'test', transform=val_transform)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # Get class names
        class_names = train_dataset.classes

        print(f"\n{'='*80}")
        print(f"DATA SUMMARY")
        print(f"{'='*80}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Augmentation level: {Config.AUGMENTATION_LEVEL}")
        print(f"Use CutMix: {Config.USE_CUTMIX}")
        print(f"Label Smoothing: {Config.LABEL_SMOOTHING}")
        print(f"{'='*80}\n")

        return train_loader, val_loader, test_loader, class_names


class ModelBuilder:
    """Builds pre-trained models with custom classification heads"""

    @staticmethod
    def build_model(model_key: str, num_classes: int, strategy: str = 'feature_extraction') -> nn.Module:
        """
        Build model with appropriate transfer learning strategy

        Args:
            model_key: Key from Config.MODELS
            num_classes: Number of output classes
            strategy: 'feature_extraction', 'partial_finetuning', or 'full_finetuning'

        Returns:
            Configured model
        """
        model_config = Config.MODELS[model_key]
        model_name = model_config['name']

        print(f"\nBuilding model: {model_key}")
        print(f"  - Strategy: {strategy}")
        print(f"  - Base model: {model_name}")
        print(f"  - Reason: {model_config['reason']}")

        # Load pre-trained model
        # For DINOv2, explicitly set img_size to handle 224x224 input
        if 'dinov2' in model_key:
            model = timm.create_model(
                model_name, 
                pretrained=True, 
                num_classes=num_classes,
                img_size=model_config['image_size']  # Explicitly set image size
            )
        else:
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

        # Apply transfer learning strategy
        if strategy == 'feature_extraction':
            # Freeze all layers except classification head
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze classification head
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True

            print("  - Frozen: All backbone layers")
            print("  - Trainable: Classification head only")

        elif strategy == 'partial_finetuning':
            # Freeze early layers, unfreeze later layers + head
            num_blocks = model_config['num_blocks']
            partial_unfreeze = model_config['partial_unfreeze']

            # Freeze all first
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze last N blocks (architecture-specific)
            if 'dinov2' in model_key or 'vit' in model_key:
                # Vision Transformer blocks
                blocks = model.blocks
                for block in blocks[-(partial_unfreeze):]:
                    for param in block.parameters():
                        param.requires_grad = True
            elif 'swin' in model_key:
                # Swin Transformer layers
                layers = model.layers
                for layer in layers[-(partial_unfreeze):]:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Unfreeze head
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True

            print(f"  - Frozen: First {num_blocks - partial_unfreeze} blocks")
            print(f"  - Trainable: Last {partial_unfreeze} blocks + head")

        elif strategy == 'full_finetuning':
            # Train all parameters
            for param in model.parameters():
                param.requires_grad = True

            print("  - Trainable: All layers (full fine-tuning)")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)\n")

        return model


class EnhancedTrainer:
    """
    Enhanced trainer with CutMix support
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 criterion: nn.Module, device: torch.device,
                 use_cutmix: bool = False, cutmix_alpha: float = 1.0, 
                 cutmix_prob: float = 0.5):
        """
        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            use_cutmix: Whether to use CutMix
            cutmix_alpha: CutMix alpha parameter
            cutmix_prob: Probability of applying CutMix
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob

    def train_epoch(self, train_loader: DataLoader, epoch: int, 
                   grad_clip: float = None) -> Tuple[float, float]:
        """
        Train for one epoch with optional CutMix

        Args:
            train_loader: Training dataloader
            epoch: Current epoch number
            grad_clip: Gradient clipping value

        Returns:
            Average loss and accuracy
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Apply CutMix with probability
            if self.use_cutmix and np.random.rand() < self.cutmix_prob:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, self.cutmix_alpha)
                
                # Forward pass
                outputs = self.model(images)
                loss = cutmix_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                
                # Accuracy (use original labels for simplicity)
                _, predicted = outputs.max(1)
                correct += (lam * predicted.eq(labels_a).sum().item() + 
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                # Normal training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Accuracy
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc


class Evaluator:
    """Evaluates model performance with optional TTA"""

    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str],
                 use_tta: bool = False, tta_num_augmentations: int = 5):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.use_tta = use_tta
        self.tta = TestTimeAugmentation(num_augmentations=tta_num_augmentations) if use_tta else None

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluate model on data

        Args:
            data_loader: DataLoader to evaluate

        Returns:
            accuracy, macro_f1, weighted_f1, all_preds, all_labels
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)

                if self.use_tta and self.tta is not None:
                    # TTA not supported in batch mode for simplicity
                    # Fall back to normal evaluation
                    outputs = self.model(images)
                else:
                    outputs = self.model(images)

                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        accuracy = (all_preds == all_labels).mean()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

        return accuracy, macro_f1, weighted_f1, all_preds, all_labels

    def detailed_evaluation(self, data_loader: DataLoader) -> Dict:
        """
        Detailed evaluation with per-class metrics

        Args:
            data_loader: DataLoader to evaluate

        Returns:
            Dictionary with detailed metrics
        """
        accuracy, macro_f1, weighted_f1, all_preds, all_labels = self.evaluate(data_loader)

        # Classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Find worst performing classes
        per_class_f1 = {
            class_name: report[class_name]['f1-score']
            for class_name in self.class_names
        }
        worst_classes = sorted(per_class_f1.items(), key=lambda x: x[1])[:5]

        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_metrics': {k: v for k, v in report.items() if k in self.class_names},
            'confusion_matrix': cm.tolist(),
            'worst_classes': worst_classes
        }

        return results


class ExperimentRunner:
    """Orchestrates complete training pipeline"""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE

        # Set seed for reproducibility
        set_seed(config.SEED)

    def run_experiment(self, model_key: str, strategy: str) -> Tuple[Dict, Path]:
        """
        Run single experiment

        Args:
            model_key: Model configuration key
            strategy: Transfer learning strategy

        Returns:
            Results dictionary and experiment directory
        """
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{model_key}_{strategy}_{Config.AUGMENTATION_LEVEL}"
        if Config.USE_CUTMIX:
            exp_name += "_cutmix"
        if Config.LABEL_SMOOTHING > 0:
            exp_name += f"_ls{Config.LABEL_SMOOTHING}"
        exp_name += f"_{timestamp}"
        
        exp_dir = Config.OUTPUT_DIR / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*100}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"{'='*100}\n")

        # Prepare data
        image_size = Config.MODELS[model_key]['image_size']
        data_manager = DataManager(
            Config.DATA_ROOT,
            image_size,
            Config.BATCH_SIZE,
            Config.NUM_WORKERS
        )
        train_loader, val_loader, test_loader, class_names = data_manager.prepare_data()

        # Build model
        model = ModelBuilder.build_model(model_key, len(class_names), strategy)
        model = model.to(self.device)

        # Setup optimizer with different learning rates
        if strategy == 'feature_extraction':
            # Only head parameters
            if hasattr(model, 'head'):
                optimizer = torch.optim.AdamW(model.head.parameters(), 
                                              lr=Config.LR_HEAD, 
                                              weight_decay=Config.WEIGHT_DECAY)
            else:
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                              lr=Config.LR_HEAD, 
                                              weight_decay=Config.WEIGHT_DECAY)
        else:
            # Separate learning rates for backbone and head
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

        # Loss function with label smoothing
        if Config.LABEL_SMOOTHING > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
            print(f"Using Label Smoothing: {Config.LABEL_SMOOTHING}")
        else:
            criterion = nn.CrossEntropyLoss()

        # Enhanced trainer with CutMix
        trainer = EnhancedTrainer(
            model, optimizer, criterion, self.device,
            use_cutmix=Config.USE_CUTMIX,
            cutmix_alpha=Config.CUTMIX_ALPHA,
            cutmix_prob=Config.CUTMIX_PROB
        )

        # Evaluator
        evaluator = Evaluator(model, self.device, class_names, 
                             use_tta=Config.USE_TTA, 
                             tta_num_augmentations=Config.TTA_NUM_AUGMENTATIONS)

        # Training loop
        best_val_f1 = 0.0
        best_val_acc = 0.0
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'lr': []
        }

        print(f"\nStarting training for {Config.EPOCHS} epochs...")
        print(f"Early stopping patience: {Config.EARLY_STOPPING_PATIENCE}")
        print(f"Gradient clipping: {Config.GRADIENT_CLIP}\n")

        start_time = time.time()

        for epoch in range(1, Config.EPOCHS + 1):
            # Train
            train_loss, train_acc = trainer.train_epoch(
                train_loader, epoch, Config.GRADIENT_CLIP
            )

            # Validate
            val_acc, val_f1, _, _, _ = evaluator.evaluate(val_loader)
            val_loss = 0.0  # Placeholder

            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['lr'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{Config.EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                patience_counter = 0

                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_val_f1': best_val_f1,
                    'epoch': epoch,
                    'history': history
                }
                torch.save(checkpoint, exp_dir / 'best_model.pth')
                print(f"  ✓ Saved best model (Val F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  Early stopping counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

            # Early stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

            # Step scheduler
            scheduler.step()

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")

        # Load best model for testing
        checkpoint = torch.load(exp_dir / 'best_model.pth', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_results = evaluator.detailed_evaluation(test_loader)

        # Save results
        config_dict = {
            'model': model_key,
            'strategy': strategy,
            'augmentation_level': Config.AUGMENTATION_LEVEL,
            'use_cutmix': Config.USE_CUTMIX,
            'label_smoothing': Config.LABEL_SMOOTHING,
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'lr_backbone': Config.LR_BACKBONE,
            'lr_head': Config.LR_HEAD,
            'weight_decay': Config.WEIGHT_DECAY,
            'num_classes': len(class_names),
            'class_names': class_names,
            'training_time': training_time
        }

        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=4)

        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=4)

        with open(exp_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4)

        # Print final results
        print(f"\n{'='*100}")
        print(f"FINAL RESULTS - {exp_name}")
        print(f"{'='*100}")
        print(f"Test Accuracy: {test_results['accuracy']*100:.2f}%")
        print(f"Macro F1: {test_results['macro_f1']:.4f}")
        print(f"Weighted F1: {test_results['weighted_f1']:.4f}")
        print(f"Training Time: {training_time/60:.2f} minutes")
        print(f"\nWorst performing classes:")
        for class_name, f1 in test_results['worst_classes']:
            print(f"  {class_name}: {f1:.4f}")
        print(f"{'='*100}\n")

        return test_results, exp_dir


def run_domain_generalization_experiments(
    models: List[str] = ['dinov2_vitb14'],
    strategies: List[str] = ['partial_finetuning'],
    augmentation_levels: List[str] = ['baseline', 'aggressive', 'aggressive', 'aggressive', 'augmix'],
    cutmix_settings: List[bool] = [False, False, True, True, True],
    label_smoothing_settings: List[float] = [0.0, 0.0, 0.0, 0.1, 0.1],
    weight_decay_settings: List[float] = [0.01, 0.01, 0.01, 0.01, 0.05]
) -> List[Dict]:
    """
    Run comprehensive domain generalization experiments

    Args:
        models: List of model keys to test
        strategies: List of training strategies
        augmentation_levels: List of augmentation levels
        cutmix_settings: List of CutMix on/off settings
        label_smoothing_settings: List of label smoothing values
        weight_decay_settings: List of weight decay values

    Returns:
        List of all experiment results
    """
    print("\n" + "="*100)
    print("DOMAIN GENERALIZATION EXPERIMENTS")
    print("Lab Data (PlantVillage) → Real-World Data (PlantDoc)")
    print("="*100 + "\n")

    all_results = []
    runner = ExperimentRunner(Config)

    # Define augmentation experiment configurations
    experiment_configs = [
        {
            'name': 'baseline',
            'description': 'No augmentation (original)',
            'augmentation_level': 'baseline',
            'use_cutmix': False,
            'label_smoothing': 0.0,
            'weight_decay': 0.01
        },
        {
            'name': 'aggressive',
            'description': 'Aggressive augmentation',
            'augmentation_level': 'aggressive',
            'use_cutmix': False,
            'label_smoothing': 0.0,
            'weight_decay': 0.01
        },
        {
            'name': 'aggressive_cutmix',
            'description': 'Aggressive + CutMix',
            'augmentation_level': 'aggressive',
            'use_cutmix': True,
            'label_smoothing': 0.0,
            'weight_decay': 0.01
        },
        {
            'name': 'aggressive_cutmix_labelsmooth',
            'description': 'Aggressive + CutMix + Label Smoothing',
            'augmentation_level': 'aggressive',
            'use_cutmix': True,
            'label_smoothing': 0.1,
            'weight_decay': 0.01
        },
        {
            'name': 'aggressive_cutmix_labelsmooth_strongreg',
            'description': 'All techniques (Aggressive + CutMix + Label Smoothing + Strong Regularization)',
            'augmentation_level': 'aggressive',
            'use_cutmix': True,
            'label_smoothing': 0.1,
            'weight_decay': 0.05
        },
        {
            'name': 'augmix',
            'description': 'AugMix augmentation',
            'augmentation_level': 'augmix',
            'use_cutmix': False,
            'label_smoothing': 0.0,
            'weight_decay': 0.01
        },
        {
            'name': 'augmix_cutmix_labelsmooth_strongreg',
            'description': 'AugMix + CutMix + Label Smoothing + Strong Regularization',
            'augmentation_level': 'augmix',
            'use_cutmix': True,
            'label_smoothing': 0.1,
            'weight_decay': 0.05
        }
    ]

    # Run experiments for each combination
    for model_key in models:
        for strategy in strategies:
            for exp_config in experiment_configs:
                # Update config
                Config.AUGMENTATION_LEVEL = exp_config['augmentation_level']
                Config.USE_CUTMIX = exp_config['use_cutmix']
                Config.LABEL_SMOOTHING = exp_config['label_smoothing']
                Config.WEIGHT_DECAY = exp_config['weight_decay']

                print(f"\n{'='*100}")
                print(f"EXPERIMENT: {model_key}_{strategy}_{exp_config['name']}")
                print(f"Description: {exp_config['description']}")
                print(f"{'='*100}\n")

                # Run experiment
                results, exp_dir = runner.run_experiment(model_key, strategy)

                # Store results
                result_summary = {
                    'model': model_key,
                    'strategy': strategy,
                    'augmentation': exp_config['name'],
                    'description': exp_config['description'],
                    'test_accuracy': results['accuracy'],
                    'macro_f1': results['macro_f1'],
                    'weighted_f1': results['weighted_f1'],
                    'experiment_dir': str(exp_dir),
                    'config': exp_config
                }
                all_results.append(result_summary)

    # Save comprehensive comparison
    with open(Config.OUTPUT_DIR / 'domain_generalization_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    # Print comparison
    print_domain_gen_comparison(all_results)

    return all_results


def print_domain_gen_comparison(results: List[Dict]):
    """
    Print formatted comparison table of domain generalization results
    Generated by Gemini + edited to fit
    """

    print("\n" + "="*150)
    print("DOMAIN GENERALIZATION RESULTS SUMMARY")
    print("="*150 + "\n")

    # Group by model and strategy
    for model in set(r['model'] for r in results):
        for strategy in set(r['strategy'] for r in results):

            filtered = [r for r in results if r['model'] == model and r['strategy'] == strategy]

            if not filtered:
                continue

            print(f"\n{model.upper()} - {strategy.upper()}")
            print("-"*150)
            print(f"{'Augmentation':<60} {'Test Acc':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Improvement':<15}")
            print("-"*150)

            baseline_acc = None

            for r in filtered:
                acc = r.get('test_accuracy', 0) * 100
                f1 = r.get('macro_f1', 0)
                wf1 = r.get('weighted_f1', 0)

                if 'baseline' in r['augmentation']:
                    baseline_acc = acc
                    improvement = "-"
                else:
                    improvement = f"+{acc - baseline_acc:.2f}%" if baseline_acc else "-"

                print(f"{r['description']:<60} {acc:>10.2f}% {f1:>11.4f} {wf1:>11.4f} {improvement:>14}")

            print()

    print("="*150 + "\n")


def run_comprehensive_comparison():
    """
    Run comprehensive comparison of models, strategies, and augmentation configurations
    Compares: dinov2_vits14 and vit_base
    Strategies: feature_extraction and partial_finetuning
    Augmentations: All 7 configurations
    
    Total experiments: 2 models × 2 strategies × 7 augmentations = 28 experiments
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE DOMAIN GENERALIZATION COMPARISON")
    print("Models: dinov2_vits14, vit_base")
    print("Strategies: feature_extraction, partial_finetuning")
    print("Augmentations: 7 configurations (baseline to full stack)")
    print("Total experiments: 28")
    print("="*100 + "\n")

    models = ['dinov2_vits14', 'vit_base']
    strategies = ['feature_extraction', 'partial_finetuning']
    
    # Define experiment configurations
    experiment_configs = [
        {
            'name': 'baseline',
            'description': 'Baseline (no augmentation)',
            'augmentation_level': 'baseline',
            'use_cutmix': False,
            'label_smoothing': 0.0,
            'weight_decay': 0.01
        },
        {
            'name': 'aggressive',
            'description': 'Aggressive augmentation',
            'augmentation_level': 'aggressive',
            'use_cutmix': False,
            'label_smoothing': 0.0,
            'weight_decay': 0.01
        },
        {
            'name': 'aggressive_cutmix',
            'description': 'Aggressive + CutMix',
            'augmentation_level': 'aggressive',
            'use_cutmix': True,
            'label_smoothing': 0.0,
            'weight_decay': 0.01
        },
        {
            'name': 'aggressive_cutmix_ls',
            'description': 'Aggressive + CutMix + Label Smoothing',
            'augmentation_level': 'aggressive',
            'use_cutmix': True,
            'label_smoothing': 0.1,
            'weight_decay': 0.01
        },
        {
            'name': 'aggressive_full',
            'description': 'Aggressive + CutMix + LS + Strong Reg',
            'augmentation_level': 'aggressive',
            'use_cutmix': True,
            'label_smoothing': 0.1,
            'weight_decay': 0.05
        },
        {
            'name': 'augmix',
            'description': 'AugMix augmentation',
            'augmentation_level': 'augmix',
            'use_cutmix': False,
            'label_smoothing': 0.0,
            'weight_decay': 0.01
        },
        {
            'name': 'augmix_full',
            'description': 'AugMix + CutMix + LS + Strong Reg',
            'augmentation_level': 'augmix',
            'use_cutmix': True,
            'label_smoothing': 0.1,
            'weight_decay': 0.05
        }
    ]

    all_results = []
    runner = ExperimentRunner(Config)
    
    total_experiments = len(models) * len(strategies) * len(experiment_configs)
    current_experiment = 0

    # Run experiments for each combination
    for model_key in models:
        for strategy in strategies:
            for exp_config in experiment_configs:
                current_experiment += 1
                
                # Update config
                Config.AUGMENTATION_LEVEL = exp_config['augmentation_level']
                Config.USE_CUTMIX = exp_config['use_cutmix']
                Config.LABEL_SMOOTHING = exp_config['label_smoothing']
                Config.WEIGHT_DECAY = exp_config['weight_decay']

                print(f"\n{'='*100}")
                print(f"EXPERIMENT {current_experiment}/{total_experiments}")
                print(f"Model: {model_key} | Strategy: {strategy} | Aug: {exp_config['name']}")
                print(f"Description: {exp_config['description']}")
                print(f"{'='*100}\n")

                # Run experiment
                try:
                    results, exp_dir = runner.run_experiment(model_key, strategy)

                    # Store results
                    result_summary = {
                        'model': model_key,
                        'strategy': strategy,
                        'augmentation': exp_config['name'],
                        'description': exp_config['description'],
                        'test_accuracy': results['accuracy'],
                        'macro_f1': results['macro_f1'],
                        'weighted_f1': results['weighted_f1'],
                        'experiment_dir': str(exp_dir),
                        'config': exp_config
                    }
                    all_results.append(result_summary)
                    
                    print(f"\n✓ Experiment {current_experiment}/{total_experiments} completed successfully")
                    
                except Exception as e:
                    print(f"\n✗ Experiment {current_experiment}/{total_experiments} failed: {str(e)}")
                    continue

    # Save comprehensive comparison
    output_file = Config.OUTPUT_DIR / 'comprehensive_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n\nResults saved to: {output_file}")

    # Print comprehensive comparison table
    print_comprehensive_comparison_table(all_results)

    return all_results


def print_comprehensive_comparison_table(results: List[Dict]):
    """
    Print a comprehensive comparison table showing all models, strategies, and augmentations
    """
    
    print("\n" + "="*180)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*180 + "\n")

    # Get unique values
    models = sorted(set(r['model'] for r in results))
    strategies = sorted(set(r['strategy'] for r in results))
    augmentations = []
    aug_seen = set()
    for r in results:
        if r['augmentation'] not in aug_seen:
            augmentations.append((r['augmentation'], r['description']))
            aug_seen.add(r['augmentation'])

    # Print table for each model
    for model in models:
        print(f"\n{'='*180}")
        print(f"MODEL: {model.upper()}")
        print(f"{'='*180}\n")
        
        for strategy in strategies:
            print(f"\nStrategy: {strategy.upper()}")
            print("-"*180)
            print(f"{'Augmentation Configuration':<50} {'Description':<45} {'Test Acc':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Improvement':<15}")
            print("-"*180)
            
            # Filter results for this model and strategy
            filtered = [r for r in results if r['model'] == model and r['strategy'] == strategy]
            
            # Get baseline accuracy for comparison
            baseline_acc = None
            baseline_result = next((r for r in filtered if 'baseline' in r['augmentation']), None)
            if baseline_result:
                baseline_acc = baseline_result['test_accuracy'] * 100
            
            # Print results for each augmentation
            for aug_name, aug_desc in augmentations:
                result = next((r for r in filtered if r['augmentation'] == aug_name), None)
                
                if result:
                    acc = result['test_accuracy'] * 100
                    macro_f1 = result['macro_f1']
                    weighted_f1 = result['weighted_f1']
                    
                    if 'baseline' in aug_name:
                        improvement = "BASELINE"
                    elif baseline_acc is not None:
                        diff = acc - baseline_acc
                        improvement = f"{diff:+.2f}%"
                    else:
                        improvement = "N/A"
                    
                    print(f"{aug_name:<50} {aug_desc:<45} {acc:>10.2f}% {macro_f1:>11.4f} {weighted_f1:>11.4f} {improvement:>14}")
                else:
                    print(f"{aug_name:<50} {aug_desc:<45} {'N/A':>10} {'N/A':>11} {'N/A':>11} {'N/A':>14}")
            
            print()

    # Print summary: Best configuration for each model-strategy combination
    print("\n" + "="*180)
    print("BEST CONFIGURATIONS SUMMARY")
    print("="*180 + "\n")
    print(f"{'Model':<20} {'Strategy':<25} {'Best Augmentation':<50} {'Test Acc':<12} {'Macro F1':<12} {'Improvement':<15}")
    print("-"*180)
    
    for model in models:
        for strategy in strategies:
            # Filter results for this model and strategy
            filtered = [r for r in results if r['model'] == model and r['strategy'] == strategy]
            
            if not filtered:
                continue
            
            # Find best by macro F1
            best = max(filtered, key=lambda x: x['macro_f1'])
            
            # Get baseline for improvement calculation
            baseline = next((r for r in filtered if 'baseline' in r['augmentation']), None)
            if baseline:
                improvement = f"+{(best['test_accuracy'] - baseline['test_accuracy'])*100:.2f}%"
            else:
                improvement = "N/A"
            
            print(f"{model:<20} {strategy:<25} {best['augmentation']:<50} {best['test_accuracy']*100:>10.2f}% {best['macro_f1']:>11.4f} {improvement:>14}")
    
    print()

    # Print model comparison: Which model is better overall?
    print("\n" + "="*180)
    print("MODEL COMPARISON (Average across all augmentations)")
    print("="*180 + "\n")
    print(f"{'Model':<20} {'Strategy':<25} {'Avg Test Acc':<15} {'Avg Macro F1':<15} {'Best Test Acc':<15} {'Best Macro F1':<15}")
    print("-"*180)
    
    for model in models:
        for strategy in strategies:
            filtered = [r for r in results if r['model'] == model and r['strategy'] == strategy]
            
            if not filtered:
                continue
            
            avg_acc = np.mean([r['test_accuracy'] for r in filtered]) * 100
            avg_f1 = np.mean([r['macro_f1'] for r in filtered])
            best_acc = max([r['test_accuracy'] for r in filtered]) * 100
            best_f1 = max([r['macro_f1'] for r in filtered])
            
            print(f"{model:<20} {strategy:<25} {avg_acc:>13.2f}% {avg_f1:>14.4f} {best_acc:>13.2f}% {best_f1:>14.4f}")
    
    print("\n" + "="*180 + "\n")

    # Print augmentation effectiveness ranking
    print("\n" + "="*180)
    print("AUGMENTATION EFFECTIVENESS RANKING (Average across all models and strategies)")
    print("="*180 + "\n")
    
    aug_performance = {}
    for aug_name, aug_desc in augmentations:
        aug_results = [r for r in results if r['augmentation'] == aug_name]
        if aug_results:
            avg_acc = np.mean([r['test_accuracy'] for r in aug_results]) * 100
            avg_f1 = np.mean([r['macro_f1'] for r in aug_results])
            aug_performance[aug_name] = {
                'description': aug_desc,
                'avg_acc': avg_acc,
                'avg_f1': avg_f1
            }
    
    # Sort by average F1
    sorted_augs = sorted(aug_performance.items(), key=lambda x: x[1]['avg_f1'], reverse=True)
    
    print(f"{'Rank':<6} {'Augmentation':<50} {'Description':<45} {'Avg Test Acc':<15} {'Avg Macro F1':<15}")
    print("-"*180)
    
    for rank, (aug_name, perf) in enumerate(sorted_augs, 1):
        print(f"{rank:<6} {aug_name:<50} {perf['description']:<45} {perf['avg_acc']:>13.2f}% {perf['avg_f1']:>14.4f}")
    
    print("\n" + "="*180 + "\n")


def main():
    """Main execution function"""
    
    print("\n" + "="*100)
    print("PLANT DISEASE CLASSIFICATION - DOMAIN GENERALIZATION PIPELINE")
    print("="*100 + "\n")

    # runs all augmentations for 2 transfer learning strategies proven to be better
    # Compares: dinov2_vits14 vs vit_base
    # Strategies: feature_extraction vs partial_finetuning  
    # Augmentations: All 7 configurations
    # Total is 28 experiments with detailed comparison tables
    
    results = run_comprehensive_comparison()


if __name__ == "__main__":
    main()
