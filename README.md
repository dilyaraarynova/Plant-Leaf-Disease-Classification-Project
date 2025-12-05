#  Plant Leaf Disease Classification

This repository contains experiments for **crop leaf disease classification** using CNNs, hybrid models, and pretrained Vision Transformers.  

## Project Summary

We evaluate:

- **CNNs:** VGG13, ResNet18  
- **Hybrid Models:**  
  - VGG13 + CBAM  
  - VGG13 + XGBoost  
  - VGG13 + Lightweight ViT  
- **Transformers (Transfer Learning):**  
  - ViT-Base  
  - DINOv2 ViT-S/14  
- **Transfer Learning Strategies:**  
  - Feature extraction  
  - Partial fine-tuning  
  - Full fine-tuning  
- **Augmentations tested:** AugMix, CutMix, aggressive transforms, label smoothing, etc.

Two evaluation setups:
1. Train on PlantVillage → Test on PlantDoc (domain shift)  
2. 5-fold CV on PlantDoc (small real-world dataset)

---

## Key Results

### CNNs (from scratch)
| Model | Accuracy (PlantDoc) |
|-------|----------------------|
| **ResNet18** | **26.67%** |
| VGG13 | 18.45% |

### Transfer Learning
| Model | Strategy | Accuracy |
|--------|----------|----------|
| **DINOv2 ViT-S/14** | Feature Extraction | **39.69%** |
| ViT-Base | Partial FT | 36.77% |

### Best Augmentation
- **AugMix** → up to **+3.3%** accuracy boost  
- Strong regularization → harms performance  

**Conclusion:** Transfer learning + targeted augmentations significantly improve real-world performance.

---

## Project Structure
```
project-root/
├── src/
│   ├── preprocessing/      
│   └── training/           
└── requirements.txt       
```
---

## 🧪 Usage

### Install dependencies
```bash
pip install -r requirements.txt
```
### Preprocess data
```bash
jupyter execute src/preprocessing/EDA.ipynb 
```
### Train and evaluate a model
```bash
python src/training/hybridmodels_experiment1_training_colab.py
```
```bash
jupyter execute src/training/vgg_and_resnet.ipynb 
```

## Contributors
**CNN Models:** Amina, Madina, Zhazelya

**Transfer Learning:** Ablay, Dilyara

**Data & Admin:** Dana

**Writing & Review:** All authors

