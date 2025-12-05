
!pip install torch torchvision tqdm pillow matplotlib seaborn scikit-learn xgboost -q

print("✓ All packages installed!")

# Check GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# Set your dataset path
DATASET_DIR = './scenario1_dataset'  
SAVE_DIR = './scenario1_results'     
print(f"Dataset directory: {DATASET_DIR}")
print(f"Results directory: {SAVE_DIR}")



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import numpy as np
import os
import time
from tqdm import tqdm # Using standard tqdm for .py script
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported!")

# Create the local results directory
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"✓ Results directory '{SAVE_DIR}' created or already exists.")



class PlantDiseaseDataset(Dataset):
    """
    Custom Dataset for Scenario 1 structure:
    scenario1_dataset/PLANT_CLASS_DISEASE/(train|validate|test)/images
    """
    def __init__(self, root_dir, split, transform=None, class_to_idx=None):
        """
        Args:
            root_dir: Root directory (e.g., /content/drive/MyDrive/scenario1_dataset)
            split: 'train', 'validate', or 'test'
            transform: Transforms to apply
            class_to_idx: Optional class mapping
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Get all class folders
        self.classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])

        # Create class to index mapping
        if class_to_idx is None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx

        # Collect all image paths
        self.samples = []

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)
            split_folder = os.path.join(class_folder, split)

            if not os.path.exists(split_folder):
                print(f"Warning: {split} folder not found in {class_name}")
                continue

            class_idx = self.class_to_idx[class_name]

            # Get all images
            for img_name in os.listdir(split_folder):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(split_folder, img_name)
                    self.samples.append((img_path, class_idx))

        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes ({split} split)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

print("✓ Dataset class defined!")



# Training transforms (moderate augmentation)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms (no augmentation)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✓ Transforms defined!")




# Load training data (Plant Village)
train_dataset = PlantDiseaseDataset(
    root_dir=DATASET_DIR,
    split='train',
    transform=train_transform
)

# Load validation data (Plant Village)
val_dataset = PlantDiseaseDataset(
    root_dir=DATASET_DIR,
    split='validation',
    transform=val_test_transform,
    class_to_idx=train_dataset.class_to_idx
)

# Load test data (Plant Doc - cross-dataset)
test_dataset = PlantDiseaseDataset(
    root_dir=DATASET_DIR,
    split='test',
    transform=val_test_transform,
    class_to_idx=train_dataset.class_to_idx
)

num_classes = len(train_dataset.classes)
class_names = train_dataset.classes

print(f"\n{'='*60}")
print(f"Dataset Summary:")
print(f"{'='*60}")
print(f"Number of classes: {num_classes}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples (Plant Doc): {len(test_dataset)}")
print(f"{'='*60}")

# Create DataLoaders
BATCH_SIZE = 64  

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"✓ DataLoaders created!")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")



# VGG13 Base Architecture
class VGG13Base(nn.Module):
    def __init__(self):
        super(VGG13Base, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)

        return x

print("✓ VGG13 Base defined!")

# CBAM Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# Model 1: VGG13 + Attention
class VGG13_Attention(nn.Module):
    def __init__(self, num_classes=12, dropout=0.5):
        super(VGG13_Attention, self).__init__()
        self.vgg_base = VGG13Base()
        self.attention = CBAM(in_channels=512, reduction=16, kernel_size=7)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.vgg_base(x)
        x = self.attention(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

print("✓ VGG13 + Attention defined!")

# Model 2: VGG13 + XGBoost (Feature Extractor)
class VGG13_FeatureExtractor(nn.Module):
    def __init__(self, num_classes=12, dropout=0.5):
        super(VGG13_FeatureExtractor, self).__init__()
        self.vgg_base = VGG13Base()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.vgg_base(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        with torch.no_grad():
            x = self.vgg_base(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        return x

print("✓ VGG13 + XGBoost (Feature Extractor) defined!")

# Model 3: VGG13 + Lightweight ViT
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=512, embed_dim=256):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.projection(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4, dropout=0.3):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class VGG13_ViT(nn.Module):
    def __init__(self, num_classes=12, embed_dim=256, num_heads=4, num_layers=2, mlp_ratio=4, dropout=0.3):
        super(VGG13_ViT, self).__init__()
        self.vgg_base = VGG13Base()
        self.patch_embed = PatchEmbedding(in_channels=512, embed_dim=embed_dim)
        # Assuming spatial size is 7x7 after VGG -> 49 patches + 1 cls token = 50 total
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head_dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.vgg_base(x)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.norm(x)
        cls_output = x[:, 0]
        x = self.head_dropout(cls_output)
        x = self.head(x)
        return x

print("✓ VGG13 + ViT defined!")



def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(val_loader), 100. * correct / total, all_preds, all_labels

print("✓ Training functions defined!")




MODEL_NAME = 'attention'  # 'attention', 'xgboost', or 'vit'
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
PATIENCE = 20  # Early stopping patience

print(f"Training {MODEL_NAME.upper()} model...")
print(f"Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}, Patience: {PATIENCE}")

# Create model
if MODEL_NAME == 'attention':
    model = VGG13_Attention(num_classes=num_classes, dropout=0.5).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
elif MODEL_NAME == 'xgboost':
    model = VGG13_FeatureExtractor(num_classes=num_classes, dropout=0.5).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
elif MODEL_NAME == 'vit':
    model = VGG13_ViT(num_classes=num_classes, dropout=0.3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

criterion = nn.CrossEntropyLoss()

print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Training loop
best_val_acc = 0.0
patience_counter = 0
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

print(f"\n{'='*70}")
print(f"Starting training...")
print(f"{'='*70}\n")

start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

    # Validate
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

    # Update scheduler
    if MODEL_NAME in ['attention', 'xgboost']:
        scheduler.step(val_acc)
    else:
        scheduler.step()

    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Print summary
    print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Early stopping check and model saving
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        save_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_scenario1_best.pth')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
        }, save_path)
        print(f" ✓ Best model saved! Path: {save_path}")
    else:
        patience_counter += 1
        print(f" Patience: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

training_time_seconds = time.time() - start_time
training_time_minutes = training_time_seconds / 60
print(f"\n{'='*70}")
print(f"Training completed in {training_time_minutes:.2f} minutes")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"{'='*70}")



# ============================================================================
# COMPREHENSIVE EVALUATION WITH ALL METRICS
# ============================================================================

load_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_scenario1_best.pth')

if os.path.exists(load_path):
    # Load best model
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    history = checkpoint.get('history', history) # Use saved history if available

    print(f"✓ Loaded best model (Val Acc: {checkpoint['best_val_acc']:.2f}%)")

    # Test on Plant Doc with all metrics
    print("\nTesting on Plant Doc (cross-dataset evaluation)...")

    # Measure inference time
    inference_times = []
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)

            # Measure inference time (per batch)
            start_batch_time = time.time()
            outputs = model(images)
            inference_time = (time.time() - start_batch_time) * 1000 # Convert to ms
            inference_times.append(inference_time)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate all metrics
    test_accuracy = 100. * correct / total
    # Added zero_division=0 for robust metric calculation
    test_precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    test_recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    test_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    test_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

    # Inference time metrics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    avg_inference_per_image = avg_inference_time / BATCH_SIZE

    print(f"\n{'='*70}")
    print(f"Cross-Dataset Test Results ({MODEL_NAME.upper()})")
    print(f"{'='*70}")
    print(f" Test Accuracy: {test_accuracy:.2f}%")
    print(f" Macro Precision: {test_precision_macro:.2f}%")
    print(f" Macro Recall: {test_recall_macro:.2f}%")
    print(f" Macro F1 Score: {test_f1_macro:.2f}%")
    print(f" Weighted F1 Score: {test_f1_weighted:.2f}%")
    print(f" Avg Inference Time: {avg_inference_time:.2f} ms/batch ({avg_inference_per_image:.2f} ms/image)")
    print(f" Std Inference Time: {std_inference_time:.2f} ms/batch")
    print(f"{'='*70}")

    # Save metrics to dictionary
    import json
    metrics = {
        'model_name': MODEL_NAME,
        'scenario': 1,
        # Accuracy metrics
        'val_accuracy': checkpoint['best_val_acc'],
        'test_accuracy': test_accuracy,
        'accuracy_drop': checkpoint['best_val_acc'] - test_accuracy,
        # Precision & Recall
        'precision_macro': test_precision_macro,
        'recall_macro': test_recall_macro,
        # F1 Scores
        'f1_macro': test_f1_macro,
        'f1_weighted': test_f1_weighted,
        # Time metrics
        'training_time_minutes': training_time_minutes,
        'inference_time_ms_per_batch': avg_inference_time,
        'inference_time_ms_per_image': avg_inference_per_image,
        'inference_time_std': std_inference_time,
        # Additional info
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'batch_size': BATCH_SIZE,
        'num_epochs_trained': len(history['train_loss'])
    }

    json_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_scenario1_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n✓ Metrics saved to {json_path}")


 

    # Confusion Matrix
    test_preds = np.array(all_preds)
    test_labels = np.array(all_labels)
    cm = confusion_matrix(test_labels, test_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {MODEL_NAME.upper()} (Scenario 1)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{MODEL_NAME}_scenario1_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Confusion matrix saved!")

    # Classification Report
    print("\nClassification Report:")
    # Added zero_division=0 for robust metric calculation
    print(classification_report(test_labels, test_preds, target_names=class_names, digits=4, zero_division=0))



    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    plt.suptitle(f'Scenario 1: {MODEL_NAME.upper()} Model Evaluation', fontsize=18, fontweight='bold')

    # 1. Training Curves (Combined Loss and Accuracy)
    ax1 = axes[0, 0]
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training and Validation Loss - {MODEL_NAME.upper()}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Secondary Y-axis for Accuracy curves
    ax2_acc = ax1.twinx()
    ax2_acc.plot(history['train_acc'], label='Train Acc', linewidth=2, color='green', linestyle='--')
    ax2_acc.plot(history['val_acc'], label='Val Acc', linewidth=2, color='red', linestyle='--')
    ax2_acc.set_ylabel('Accuracy (%)', fontsize=12, color='green')
    ax2_acc.tick_params(axis='y', labelcolor='green')
    ax2_acc.legend(loc='upper right', fontsize=10)


    # 2. All Test Metrics Bar Chart
    ax2 = axes[0, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Macro F1', 'Weighted F1']
    metric_values = [test_accuracy, test_precision_macro, test_recall_macro, test_f1_macro, test_f1_weighted]
    bars2 = ax2.barh(metric_names, metric_values, color='#3498db')
    ax2.set_xlabel('Score (%)', fontsize=12)
    ax2.set_title(f'All Metrics (Test Set) - {MODEL_NAME.upper()}', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                 f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=10)

    # 3. Time comparison
    ax3 = axes[1, 0]
    times = [training_time_minutes * 60, avg_inference_per_image / 1000] # Convert to seconds
    labels = [f'Training\n({training_time_minutes:.1f} min)', f'Inference\n({avg_inference_per_image:.2f} ms)']
    colors = ['#f39c12', '#9b59b6']
    bars3 = ax3.bar(labels, times, color=colors)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_title(f'Time Metrics - {MODEL_NAME.upper()}', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    for bar in bars3:
        height = bar.get_height()
        if height > 60:
            text = f'{height/60:.1f} min'
        else:
            text = f'{height:.2f} s'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 text, ha='center', va='bottom', fontweight='bold')

    # 4. F1 Score comparison
    ax4 = axes[1, 1]
    f1_types = ['Macro F1', 'Weighted F1']
    f1_values = [test_f1_macro, test_f1_weighted]
    bars4 = ax4.bar(f1_types, f1_values, color=['#1abc9c', '#16a085'])
    ax4.set_ylabel('F1 Score (%)', fontsize=12)
    ax4.set_title(f'F1 Score Comparison (Test Set) - {MODEL_NAME.upper()}', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 100])
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig(os.path.join(SAVE_DIR, f'{MODEL_NAME}_scenario1_summary_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Summary plots saved!")
else:
    print(f"\nWarning: Skipping cross-dataset evaluation and plotting because the best model checkpoint was not found at {load_path}. Please run the script in an environment where the training step can complete successfully.")

print("\n\nAll steps completed.")
