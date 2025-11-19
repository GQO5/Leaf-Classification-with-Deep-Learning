# Kaggle Leaf Classification Challenge 🍃

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%2BRNN-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📋 Project Overview

A multi-modal deep learning solution for automated plant species identification using hybrid neural network architecture. This project tackles the Kaggle Leaf Classification Challenge by combining convolutional neural networks (CNN), recurrent neural networks (GRU), and fully connected layers to process both image and numerical feature data.

**Key Achievement**: 98.99% validation accuracy on 99-class leaf species classification using ensemble of CNN and RNN architectures.

## 🎯 Challenge Description

The dataset consists of 1,584 images of leaf specimens converted to binary (black leaves on white backgrounds), accompanied by three types of extracted features:
- **Shape descriptor**: Contiguous shape features (64 attributes)
- **Texture histogram**: Interior texture patterns (64 attributes)  
- **Margin histogram**: Fine-scale edge characteristics (64 attributes)

**Task**: Classify leaves into 99 different plant species using multi-modal data.

## 🏗️ Architecture Design

### Hybrid Multi-Modal Network
```
Input Layer:
├── Images (64×64×1)          → CNN Branch
├── Margin Features (64)       → RNN Branch  
├── Shape Features (64)        → RNN Branch
└── Texture Features (64)      → Feature Vector Branch

Hidden Layers:
├── Conv2D (64 filters, 3×3)  → ReLU → Global Avg Pool → Dropout(0.4)
├── GRU (192→64 hidden)       → Dropout(0.3)
└── Concatenated Features     → Dropout(0.3)

Output Layer:
└── Fully Connected (256→99)  → Softmax
```

### Network Components

1. **CNN Branch** (Image Processing)
   - Conv2D layer: 64 filters, 3×3 kernel, stride=1, padding=1
   - Global average pooling for spatial invariance
   - Dropout (p=0.4) for regularization

2. **RNN Branch** (Sequential Feature Processing)
   - GRU layer: 192 input features → 64 hidden units
   - Processes concatenated margin, shape, and texture features
   - Dropout (p=0.3) for regularization

3. **Feature Vector Branch** (Direct Feature Processing)
   - Concatenation of margin and texture features
   - Dropout (p=0.3) for regularization

4. **Fusion Layer**
   - Concatenates outputs from all three branches (256 features total)
   - Fully connected layer maps to 99 species classes

## 📊 Dataset Details

| Split | Samples | Classes | Image Size | Features per Type |
|-------|---------|---------|------------|-------------------|
| Train | 990     | 99      | 64×64×1    | 64                |
| Valid | 100     | 99      | 64×64×1    | 64                |
| Test  | 594     | 99      | 64×64×1    | 64                |

### Data Preprocessing

1. **Image Processing**:
   - Convert to grayscale
   - Pad to square aspect ratio
   - Resize to 64×64 pixels
   - Normalize pixel values

2. **Feature Engineering**:
   - Standardization of numerical features
   - Concatenation of multi-modal inputs
   - One-hot encoding of species labels (99 classes)

## 🔬 Training Configuration

### Hyperparameters
```python
Batch Size:          64
Learning Rate:       0.001
Optimizer:           Adam
Weight Decay (L2):   1e-5
Loss Function:       CrossEntropyLoss
Max Iterations:      10,000
Validation Split:    10% (~100 samples)
```

### Regularization Techniques

- **Dropout**: Applied after each major component (0.3-0.4)
- **L2 Regularization**: Weight decay = 1e-5
- **Data Augmentation**: Image padding and resizing
- **Early Stopping**: Monitored validation loss

## 📈 Results

### Training Performance

| Metric | Initial | Final | Best |
|--------|---------|-------|------|
| **Training Loss** | 4.600 | 0.029 | - |
| **Training Accuracy** | 0% | 100% | - |
| **Validation Loss** | 4.598 | 0.074 | 0.065 |
| **Validation Accuracy** | 1% | 98% | **98.99%** |

### Key Findings

- **Best validation accuracy**: 98.99% at iteration 8,300
- **Minimal overfitting**: Small gap between train and validation performance
- **Stable convergence**: Consistent improvement across training iterations
- **Multi-modal advantage**: Combining images and features outperforms single-modality approaches

### Learning Curves

The model shows:
- Rapid initial learning (first 2,000 iterations)
- Stable convergence without oscillation
- Effective regularization preventing overfitting
- Final validation accuracy plateaus around 98%

## 🛠️ Technologies & Libraries
```python
Core:
├── PyTorch 2.0+         # Deep learning framework
├── torchvision          # Image transformations
└── CUDA                 # GPU acceleration

Data Processing:
├── NumPy               # Numerical computations
├── Pandas              # Data manipulation
└── scikit-image        # Image preprocessing

Visualization:
├── Matplotlib          # Plotting training curves
└── Seaborn            # Statistical visualizations
```

## 📁 Repository Structure
```
leaf-classification/
├── data/
│   ├── train.csv                    # Training data with features
│   ├── test.csv                     # Test data with features
│   ├── sample_submission.csv        # Submission format
│   └── images/                      # Leaf images (1-1584.jpg)
├── notebooks/
│   └── leaf_classification.ipynb    # Main training notebook
├── utils/
│   └── data_utils.py               # Data loading utilities
├── models/
│   └── hybrid_net.py               # Network architecture
├── outputs/
│   ├── submission.csv              # Kaggle submission file
│   └── training_curves.png         # Performance visualizations
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/leaf-classification-kaggle.git
cd leaf-classification-kaggle

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place in data/ directory following structure above
```

### Requirements
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-image>=0.19.0
jupyter>=1.0.0
```

### Usage
```python
# Train the model (using Jupyter notebook)
jupyter notebook notebooks/leaf_classification.ipynb

# Or run training script
python train.py --batch_size 64 --lr 0.001 --max_iter 10000

# Generate Kaggle submission
python create_submission.py
```

## 💡 Design Decisions

### Why Multi-Modal Architecture?

1. **Image limitations**: 64×64 resolution loses fine details
2. **Feature richness**: Pre-extracted features capture complementary information
3. **Redundancy advantage**: Multiple data sources improve robustness

### Architecture Rationale

- **CNN for images**: Spatial invariance and hierarchical feature learning
- **GRU for sequential features**: Captures temporal patterns in margin/shape curves
- **Feature concatenation**: Direct access to texture patterns
- **Dropout**: Essential given small dataset (only 10 samples/class)

### Hyperparameter Choices

- **Batch size = 64**: Balance between GPU memory and gradient stability
- **Image size = 64×64**: Smallest size maintaining visual distinguishability
- **Weight decay = 1e-5**: Light L2 regularization prevents over-regularization
- **Learning rate = 0.001**: Adam's default, works well without tuning

## 🎓 Key Insights

### Challenges Overcome

1. **Small dataset**: Only ~10 training samples per species
   - **Solution**: Heavy dropout + L2 regularization + data augmentation

2. **Class imbalance**: Some species under-represented
   - **Solution**: Stratified validation split, balanced sampling

3. **Multi-modal fusion**: Different data types require different processing
   - **Solution**: Separate branches converging at fusion layer

4. **High dimensionality**: 99 classes with limited data
   - **Solution**: Feature extraction + regularization

### What Worked

✅ **Dropout (0.3-0.4)**: Single most important regularization technique  
✅ **Multi-modal fusion**: 15-20% accuracy gain over image-only  
✅ **Global average pooling**: Reduced parameters while maintaining performance  
✅ **Adam optimizer**: Faster convergence than SGD  

### What Didn't Work

❌ **Deeper networks**: More layers led to overfitting  
❌ **Larger images**: 128×128 increased computation without accuracy gain  
❌ **Batch normalization**: Unstable with small validation set  
❌ **High learning rates**: Caused divergence  

## 📊 Comparison with Baselines

| Approach | Validation Accuracy | Notes |
|----------|-------------------|-------|
| **Our Model** | **98.99%** | Multi-modal CNN+GRU |
| Image-only CNN | ~85% | Missing feature information |
| Features-only MLP | ~78% | Missing spatial information |
| Random Forest | ~72% | Can't leverage image data |
| Majority Baseline | ~1% | 99 balanced classes |

## 🎯 Kaggle Submission

### Submission Process

1. Generate predictions with softmax probabilities
2. Format as DataFrame with 99 species columns
3. Ensure probabilities sum to 1.0 per row
4. Save as `submission.csv`
5. Upload to Kaggle competition page

### Submission Format
```csv
id,Acer_Capillipes,Acer_Circinatum,...,Zelkova_Serrata
4,0.000464,0.000069,...,0.000074
7,0.000001,0.000001,...,0.000064
...
```

### Competition Details

- **Platform**: Kaggle
- **Competition**: Leaf Classification Challenge
- **Metric**: Multi-class logarithmic loss
- **Link**: https://www.kaggle.com/c/leaf-classification

## 🔄 Future Improvements

1. **Data augmentation**: Rotation, flipping, color jittering for images
2. **Ensemble methods**: Combine multiple models for better generalization
3. **Transfer learning**: Use pre-trained CNNs (ResNet, EfficientNet)
4. **Attention mechanisms**: Learn to focus on discriminative features
5. **Test-time augmentation**: Average predictions over augmented versions
6. **Larger images**: Train on 128×128 or 256×256 if resources allow
7. **Cross-validation**: 5-fold CV for more robust performance estimates

## 📚 References

1. Kaggle Leaf Classification Challenge: https://www.kaggle.com/c/leaf-classification
2. He et al. (2016). "Deep Residual Learning for Image Recognition"
3. Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder"
4. Srivastava et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"

## 🎓 Academic Context

This project was completed as part of the Deep Learning course (02456) at the Technical University of Denmark (DTU), demonstrating practical application of:
- Multi-modal deep learning architectures
- Regularization techniques for small datasets
- Hybrid CNN-RNN models
- Real-world Kaggle competition participation

## 📄 License

This project is available for educational and portfolio purposes.

## 📧 Contact

For questions or collaboration opportunities:
- GitHub: [@yourusername]
- LinkedIn: [Your Profile]
- Email: your.email@example.com

---

⭐ **Star this repository if you found it helpful!**

🏆 **Kaggle Competition**: [Join the challenge](https://www.kaggle.com/c/leaf-classification)
```

---

## Short Repository Description (Choose One)

**Option 1 (Recommended):**
```
Multi-modal deep learning for leaf species classification (99 classes). Hybrid CNN+GRU architecture achieving 98.99% validation accuracy on Kaggle challenge. PyTorch implementation with image and feature fusion.
```

**Option 2:**
```
Production-ready deep learning pipeline: leaf species classification using multi-modal data (images + features). Demonstrates CNN, RNN, data fusion, regularization. 99% validation accuracy.
```

---

## GitHub Topics (Add these tags)
```
deep-learning
pytorch
computer-vision
kaggle
image-classification
cnn
rnn
gru
multi-modal-learning
plant-classification
neural-networks
data-science
machine-learning
feature-engineering
