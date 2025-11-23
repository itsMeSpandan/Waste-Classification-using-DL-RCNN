# Waste Detection & Classification System

An intelligent waste management system using deep learning for automated waste detection and classification. The system combines a custom CNN classifier with R-CNN object detection to identify and categorize waste as either **Organic (WET)** or **Recyclable (DRY)**.

---

## 🌟 Features

### CNN Waste Classifier
- ✅ **Binary Classification**: Organic vs Recyclable waste
- ✅ **Custom CNN Architecture**: 4 convolutional blocks with batch normalization
- ✅ **Kaggle Dataset Integration**: Automatic dataset download and organization
- ✅ **GPU Acceleration**: CUDA support for faster training
- ✅ **Data Augmentation**: Advanced augmentation for robust model training
- ✅ **Training Visualization**: Comprehensive plots and metrics
- ✅ **128×128 Input**: Optimized for speed and accuracy

### R-CNN Waste Detector
- ✅ **Multi-Object Detection**: Detect multiple waste items in single image
- ✅ **Region Proposal**: Grid-based, random, and edge-based strategies
- ✅ **Integrated CNN Classification**: Uses trained CNN for region classification
- ✅ **DRY/WET Categorization**: Practical waste sorting labels
- ✅ **Heatmap Visualization**: Confidence-based detection heatmaps
- ✅ **Batch Processing**: Process entire folders of images
- ✅ **Non-Maximum Suppression**: Removes duplicate detections
- ✅ **Comprehensive Statistics**: Detailed waste analysis reports

### Visualization Tools
- ✅ **Training History Plots**: Accuracy, loss, learning rate curves
- ✅ **Detection Overviews**: 6-panel comprehensive visualization
- ✅ **Batch Analysis Graphs**: Multi-image statistics and distributions
- ✅ **Confidence Heatmaps**: Visual representation of detection confidence

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Usage Guide](#usage-guide)
6. [Model Training](#model-training)
7. [Detection & Classification](#detection--classification)
8. [Batch Processing](#batch-processing)
9. [Visualization](#visualization)
10. [Performance](#performance)
11. [Results](#results)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)
14. [License](#license)

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8 - 3.11
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **Processor**: Intel i5 or equivalent

### Recommended Requirements
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
- **RAM**: 16 GB or more
- **Storage**: 10 GB free space (for datasets)
- **Processor**: Intel i7 or AMD Ryzen 7

### GPU Acceleration (Optional but Recommended)
- **CUDA**: Version 11.8 or 12.x
- **cuDNN**: Compatible version with CUDA
- **NVIDIA Driver**: Latest stable version

---

## 🚀 Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/waste-detection-system.git
cd waste-detection-system
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**For CPU-only:**
```bash
pip install -r requirements.txt
```

**For GPU acceleration:**
```bash
pip install tensorflow[and-cuda]
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

---

## ⚡ Quick Start

### 1. Train the CNN Model

```bash
python cnn_waste_classifier.py
```

**Follow the prompts:**
- Select option `1` to download dataset and train
- Choose a Kaggle dataset (or use default)
- Set training epochs (default: 30)
- Wait for training to complete

### 2. Detect Waste in Image

```bash
python rcnn_waste_detector.py
```

**Follow the prompts:**
- Select option `1` for single image detection
- Enter path to your image
- View detection results and visualization

### 3. Batch Process Images

```bash
python rcnn_waste_detector.py
```

**Follow the prompts:**
- Select option `2` for batch processing
- Enter folder path containing images
- Review batch statistics

### 4. Visualize Batch Results

```bash
python visualize_batch_results.py
```

---

## 📁 Project Structure

```
waste-detection-system/
├── cnn_waste_classifier.py       # CNN training and classification
├── rcnn_waste_detector.py        # R-CNN object detection
├── visualize_batch_results.py    # Batch results visualization
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── METHODOLOGY.md                 # Technical methodology document
│
├── models/                        # Trained models (created after training)
│   └── cnn_waste_model.h5        # Main trained CNN model
│
├── waste_dataset/                 # Dataset (downloaded automatically)
│   └── organized/
│       ├── train/
│       │   ├── organic/
│       │   └── recyclable/
│       └── val/
│           ├── organic/
│           └── recyclable/
│
├── input_images/                  # Input images for detection
│
├── batch_test/                    # Batch processing results
│   ├── rcnn_batch_detection_results.json
│   ├── batch_detection_visualization.png
│   └── batch_detection_detailed_analysis.png
│
└── outputs/                       # Detection outputs (created automatically)
    ├── *_rcnn_complete_overview.jpg
    ├── rcnn_detection_results_*.json
    ├── cnn_training_history.png
    └── cnn_training_history.json
```

---

## 📖 Usage Guide

### CNN Waste Classifier (`cnn_waste_classifier.py`)

#### Main Menu Options:

**1. Download dataset and train CNN model**
- Downloads waste dataset from Kaggle
- Automatically organizes into train/val splits
- Trains custom CNN architecture
- Saves best model to `cnn_waste_model.h5`

**2. Train CNN with existing dataset**
- Uses previously downloaded dataset
- Skips download step
- Useful for retraining with different parameters

**3. Classify an image**
- Loads trained model
- Classifies single image
- Shows probability breakdown
- Saves results to JSON

**4. Show CNN training graphs/plots**
- Displays training history
- Shows accuracy/loss curves
- Learning rate schedule
- Summary statistics

**5. GPU diagnostics and test**
- Checks GPU availability
- Tests GPU computation
- Shows memory usage
- Provides troubleshooting tips

**6. Exit**

---

### R-CNN Waste Detector (`rcnn_waste_detector.py`)

#### Main Menu Options:

**1. R-CNN detection with trained CNN classifier**
- Loads trained CNN model
- Generates region proposals (~100 per image)
- Classifies each region
- Applies Non-Maximum Suppression
- Creates 6-panel visualization
- Saves detection results to JSON

**2. Batch R-CNN detection on image folder**
- Processes all images in folder
- Generates individual detections
- Aggregates statistics
- Saves batch results JSON
- Provides summary report

**3. Reload CNN model**
- Lists available model files
- Reloads selected model
- Tests model connection
- Useful for switching models

**4. Exit**

---

### Visualization Tool (`visualize_batch_results.py`)

Creates comprehensive graphs from batch detection results:

**Main Visualization (9 panels):**
1. Overall statistics bar chart
2. DRY vs WET pie chart
3. Average objects per image
4. Stacked bar chart (objects per image)
5. Detection count distribution
6. Confidence score distribution
7. Top 10 images by object count
8. Summary statistics box

**Detailed Analysis (4 panels):**
1. All confidence scores histogram
2. Confidence by class box plot
3. DRY vs WET scatter plot
4. Total detections by class

**Usage:**
```bash
python visualize_batch_results.py

# Press Enter to use default path
# Or enter custom JSON path
# Option to create detailed analysis
```

---

## 🎓 Model Training

### Kaggle Setup (First Time Only)

1. **Create Kaggle Account**: https://www.kaggle.com
2. **Get API Token**:
   - Go to Kaggle → Settings → API
   - Click "Create New API Token"
   - Download `kaggle.json`
3. **Place kaggle.json**:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

### Training Process

```bash
python cnn_waste_classifier.py
```

**Training Configuration:**
- **Architecture**: Custom CNN (4 conv blocks)
- **Input Size**: 128×128×3 RGB
- **Classes**: 2 (organic, recyclable)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 30 (default)
- **Callbacks**:
  - ModelCheckpoint (save best model)
  - EarlyStopping (patience=8)
  - ReduceLROnPlateau (patience=4)

**Data Augmentation:**
- Rotation: ±40°
- Width/Height Shift: ±30%
- Horizontal/Vertical Flip
- Zoom: ±30%
- Shear: ±20%
- Brightness: 80-120%

**Expected Training Time:**
- GPU: 10-20 minutes (30 epochs)
- CPU: 2-4 hours (30 epochs)

**Output Files:**
- `cnn_waste_model.h5` - Best model weights
- `cnn_training_history.png` - Training plots
- `cnn_training_history.json` - Training metrics

---

## 🔍 Detection & Classification

### Single Image Detection

```bash
python rcnn_waste_detector.py
# Select option 1
# Enter image path: input_images/waste_001.jpg
```

**Detection Pipeline:**
1. Load trained CNN model
2. Generate ~100 region proposals
3. Classify each region with CNN
4. Filter by confidence (>0.5)
5. Apply Non-Maximum Suppression
6. Generate visualizations
7. Save results to JSON

**Output Files:**
- `{image_name}_rcnn_complete_overview.jpg` - 6-panel visualization
- `rcnn_detection_results_{image_name}.json` - Complete detection data

**Detection Results Include:**
- Bounding boxes with coordinates
- Class predictions (organic/recyclable)
- Confidence scores
- DRY/WET classification
- Recyclability status
- Size and location data

### Single Image Classification

```bash
python cnn_waste_classifier.py
# Select option 3
# Enter image path: input_images/waste_001.jpg
```

**Output:**
- Predicted class
- Confidence percentage
- Probability breakdown
- Visual bar chart

---

## 📦 Batch Processing

### Process Entire Folder

```bash
python rcnn_waste_detector.py
# Select option 2
# Enter folder path: input_images/
```

**Batch Features:**
- Processes all jpg, jpeg, png, bmp, tiff files
- Individual detection for each image
- Aggregated statistics across all images
- Progress tracking
- Error handling

**Batch Statistics:**
- Total objects detected
- DRY vs WET breakdown
- Average objects per image
- Confidence distributions
- Per-image summaries

**Output:**
- `batch_test/rcnn_batch_detection_results.json`
  - Batch info and metadata
  - Overall statistics
  - Detailed results per image
  - Individual detections

### Visualize Batch Results

```bash
python visualize_batch_results.py
```

**Creates:**
- `batch_detection_visualization.png` (9-panel overview)
- `batch_detection_visualization.pdf` (high-res PDF)
- `batch_detection_detailed_analysis.png` (4-panel detailed)

---

## 📊 Visualization

### Training Visualizations

**CNN Training History** (`cnn_training_history.png`):
- Accuracy over epochs (train & validation)
- Loss over epochs (train & validation)
- Learning rate schedule
- Training summary statistics

### Detection Visualizations

**R-CNN Complete Overview** (`*_rcnn_complete_overview.jpg`):
1. **Original Image** - Unmodified input
2. **Detected Objects** - Bounding boxes with labels
3. **Confidence Heatmap** - Gaussian-blurred heat visualization
4. **Waste Distribution** - DRY vs WET bar chart
5. **Class Breakdown** - Pie chart of detected classes
6. **Summary Statistics** - Text box with metrics

**Labels Format:**
```
[R/W] DRY/WET(class) confidence
```
- `[R]` = Recyclable (DRY)
- `[W]` = Waste/Organic (WET)
- Color-coded: Blue (DRY), Brown (WET)

### Batch Visualizations

**Main Visualization** (9 panels):
- Statistics, distributions, top performers, summary

**Detailed Analysis** (4 panels):
- Confidence distributions, class comparisons, scatter plots

---

## ⚡ Performance

### Model Performance

**CNN Classifier:**
- Input: 128×128×3
- Parameters: ~1-5M (depending on architecture)
- Inference Time: 50-100ms per image (GPU)
- Accuracy: 85-95% (depends on dataset quality)

**R-CNN Detector:**
- Region Proposals: ~100 per image
- Detection Time: 5-15 seconds per image (GPU)
- Objects Per Image: 0-15 (with NMS)
- Confidence Threshold: 0.5 (50%)

### Hardware Performance

| Hardware | CNN Training (30 epochs) | R-CNN Detection | Batch (100 images) |
|----------|-------------------------|-----------------|-------------------|
| **RTX 3060** | 10-15 min | 3-5 sec | 5-8 min |
| **GTX 1060** | 20-30 min | 8-12 sec | 15-20 min |
| **CPU (i7)** | 2-3 hours | 30-60 sec | 50-100 min |
| **CPU (i5)** | 3-4 hours | 45-90 sec | 75-150 min |

### Optimization Tips

**For Faster Training:**
1. Use GPU acceleration
2. Reduce batch size if OOM errors
3. Use fewer epochs for quick testing
4. Enable mixed precision (advanced)

**For Faster Detection:**
1. Reduce `region_proposals_count` (default: 100)
2. Increase `confidence_threshold` (default: 0.5)
3. Lower `nms_threshold` for fewer detections
4. Use GPU acceleration

---

## 📈 Results

### Sample Detection Results

**Image: waste_001.jpg**
```
Total Objects: 12
├─ DRY (Recyclable): 8 objects (66.7%)
├─ WET (Organic): 4 objects (33.3%)
└─ Average Confidence: 78.2%

Top Detections:
1. Recyclable - 95.3% confidence (plastic bottle)
2. Recyclable - 89.7% confidence (cardboard box)
3. Organic - 85.1% confidence (food waste)
```

### Batch Processing Example

**Folder: input_images/ (62 images)**
```
Overall Statistics:
├─ Total Objects: 660
├─ DRY Waste: 353 (53.5%)
├─ WET Waste: 307 (46.5%)
├─ Avg Objects/Image: 10.6
└─ Avg Confidence: 77.8%

Distribution:
├─ Min Objects: 5
├─ Max Objects: 15
└─ Most Common: 10-12 objects
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. No GPU Detected

**Problem:** GPU not being used for training/detection

**Solution:**
```bash
# Uninstall CPU-only TensorFlow
pip uninstall tensorflow

# Install GPU version
pip install tensorflow[and-cuda]

# Verify NVIDIA driver
nvidia-smi

# Test GPU in Python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### 2. Out of Memory (OOM) Error

**Problem:** GPU runs out of memory during training

**Solution:**
- Reduce batch size: `batch_size=16` or `batch_size=8`
- Reduce image size (already optimized at 128×128)
- Close other GPU applications
- Enable memory growth (already implemented)

#### 3. Kaggle Download Fails

**Problem:** Cannot download dataset from Kaggle

**Solution:**
1. Verify `kaggle.json` exists and has correct credentials
2. Check internet connection
3. Try alternative dataset from menu
4. Manual download:
   - Download from Kaggle website
   - Extract to `waste_dataset/`
   - Organize into `organized/train/` and `organized/val/`

#### 4. Model Loading Error

**Problem:** Cannot load `cnn_waste_model.h5`

**Solution:**
```bash
# Retrain model with current TensorFlow version
python cnn_waste_classifier.py
# Select option 2 (train with existing data)

# Or reload model
python rcnn_waste_detector.py
# Select option 3 (reload CNN model)
```

#### 5. No Objects Detected

**Problem:** R-CNN finds no waste objects

**Solution:**
- Check image quality (not too blurry/dark)
- Lower confidence threshold in code
- Ensure model is trained properly
- Verify image contains waste objects
- Try different image

#### 6. Blank Training Plots

**Problem:** Training plots show no data

**Solution:**
- Retrain the model completely
- Delete old `cnn_training_history.json`
- Ensure training completes without errors
- Check for TensorFlow version compatibility

---

## 🧪 Advanced Usage

### Custom Configuration

Edit parameters in scripts:

**cnn_waste_classifier.py:**
```python
# Change input size
self.input_shape = (256, 256, 3)  # Higher resolution

# Change epochs
epochs = 50  # More training

# Change batch size
batch_size = 16  # Smaller for GPU memory
```

**rcnn_waste_detector.py:**
```python
# Change confidence threshold
self.confidence_threshold = 0.6  # Stricter filtering

# Change region proposals
self.region_proposals_count = 200  # More proposals

# Change NMS threshold
self.nms_threshold = 0.4  # More aggressive filtering
```

### Custom Dataset

1. Organize your dataset:
```
waste_dataset/organized/
├── train/
│   ├── organic/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── recyclable/
│       ├── img001.jpg
│       └── img002.jpg
└── val/
    ├── organic/
    └── recyclable/
```

2. Train with custom data:
```bash
python cnn_waste_classifier.py
# Select option 2 (train with existing data)
```

---

## 📝 Output File Formats

### Detection JSON Format

**rcnn_detection_results_{image}.json:**
```json
{
  "detection_info": {
    "method": "R-CNN",
    "image_path": "path/to/image.jpg",
    "analysis_timestamp": "2025-11-23T10:30:00"
  },
  "detection_summary": {
    "total_objects": 12,
    "average_confidence": 0.782,
    "max_confidence": 0.953
  },
  "waste_classification": {
    "dry_waste_count": 8,
    "wet_waste_count": 4,
    "dry_percentage": 66.7
  },
  "individual_detections": [
    {
      "object_id": 1,
      "class": "recyclable",
      "confidence": 0.953,
      "waste_type": "dry",
      "is_recyclable": true,
      "bounding_box": [150, 200, 300, 400]
    }
  ]
}
```

### Batch Results JSON Format

**rcnn_batch_detection_results.json:**
```json
{
  "batch_info": {
    "total_images_processed": 62,
    "processing_timestamp": "2025-11-23T10:30:00"
  },
  "overall_statistics": {
    "total_objects_detected": 660,
    "total_dry_waste": 353,
    "total_wet_waste": 307,
    "average_objects_per_image": 10.65
  },
  "detailed_results": [...]
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/waste-detection-system.git

# Create development branch
git checkout -b dev

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **TensorFlow Team** - Deep learning framework
- **Kaggle Community** - Waste classification datasets
- **OpenCV** - Image processing capabilities
- **Matplotlib** - Visualization tools

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/waste-detection-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/waste-detection-system/discussions)

---

## 🔄 Version History

### v1.0.0 (Current)
- ✅ Custom CNN waste classifier
- ✅ R-CNN object detection
- ✅ DRY/WET classification
- ✅ Batch processing
- ✅ Comprehensive visualizations
- ✅ GPU acceleration support
- ✅ Kaggle dataset integration

### Planned Features (v2.0.0)
- [ ] Real-time video detection
- [ ] Multi-class detection (plastic, metal, paper, glass)
- [ ] Mobile app deployment
- [ ] Cloud API integration
- [ ] Advanced metrics dashboard
- [ ] Transfer learning with larger models

---

**Made with ❤️ for a cleaner planet 🌍**
