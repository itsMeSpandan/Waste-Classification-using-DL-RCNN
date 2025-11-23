"""
CNN Waste Classifier with Kaggle Dataset Integration
This script uses a custom CNN architecture instead of VGG16 transfer learning
for waste classification (organic vs recyclable).
"""

import os
import sys
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Tuple, List
import json

# GPU Setup and Detection Functions
def setup_gpu():
    """Setup and configure GPU for TensorFlow."""
    print("\n" + "="*60)
    print("GPU SETUP AND DETECTION")
    print("="*60)
    
    # Check TensorFlow version
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Check if TensorFlow was built with CUDA support
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # List all physical devices
    print("\nPhysical devices:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"  {device}")
    
    # Check GPU devices specifically
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"\nGPU devices found: {len(gpu_devices)}")
    
    if gpu_devices:
        print("✓ GPU(s) detected!")
        for i, gpu in enumerate(gpu_devices):
            print(f"  GPU {i}: {gpu}")
            
        # Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth configured")
            
            # Optional: Set memory limit (prevents OOM errors)
            try:
                tf.config.set_logical_device_configuration(
                    gpu_devices[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]  # 3GB limit
                )
                print("✓ GPU memory limit set to 3GB")
            except:
                pass  # Memory limit setting is optional
                
        except Exception as e:
            print(f"⚠ Could not configure GPU memory growth: {e}")
        
        # Skip mixed precision to avoid compatibility issues
        print("ℹ Mixed precision disabled for stability")
        
        return True
    else:
        print("❌ No GPU devices found!")
        
        return False

def check_gpu_utilization():
    """Check current GPU utilization and memory usage."""
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print("\n📊 GPU Status:")
            for i, gpu in enumerate(gpu_devices):
                try:
                    # Get GPU memory info
                    memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                    current_mb = memory_info['current'] / 1024 / 1024
                    peak_mb = memory_info['peak'] / 1024 / 1024
                    print(f"  GPU {i} Memory - Current: {current_mb:.1f}MB, Peak: {peak_mb:.1f}MB")
                except Exception as e:
                    print(f"  GPU {i}: Memory info not available ({e})")
        else:
            print("No GPU devices available for monitoring")
    except Exception as e:
        print(f"Could not check GPU utilization: {e}")

def run_gpu_test():
    """Run a simple GPU computation test."""
    print("\n🧪 Running GPU computation test...")
    try:
        with tf.device('/GPU:0'):
            # Create random matrices and multiply them
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            
        print("✓ GPU computation test successful!")
        print(f"  Result shape: {c.shape}")
        return True
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
        print("  Falling back to CPU computation")
        return False

class CNNWasteClassifier:
    """Custom CNN waste classifier with Kaggle dataset integration."""
    
    def __init__(self):
        self.model = None
        self.input_shape = (128, 128, 3)  # Smaller input for custom CNN
        self.classes = ['organic', 'recyclable']  # Binary classification
        self.dataset_path = "waste_dataset"
        self.model_path = "cnn_waste_model.h5"
        self.gpu_available = False
        
        # Setup GPU on initialization
        self.gpu_available = setup_gpu()
        
    def setup_kaggle_credentials(self):
        """Setup Kaggle API credentials."""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if kaggle_json.exists():
            print("✓ Kaggle credentials found")
            return True
        
        print("\n" + "="*60)
        print("KAGGLE API SETUP REQUIRED")
        print("="*60)
        print("\nTo download datasets from Kaggle:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. This downloads kaggle.json")
        print(f"5. Place kaggle.json in: {kaggle_dir}")
        print("\nAlternatively, enter your Kaggle credentials now:")
        
        username = input("\nKaggle Username (or 'skip'): ").strip()
        if username.lower() == 'skip':
            return False
        
        key = input("Kaggle API Key: ").strip()
        
        # Create kaggle directory
        kaggle_dir.mkdir(exist_ok=True, parents=True)
        
        # Create kaggle.json
        kaggle_config = {
            "username": username,
            "key": key
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(kaggle_config, f)
        
        # Set permissions (important for Linux/Mac)
        try:
            os.chmod(kaggle_json, 0o600)
        except:
            pass
        
        print(f"✓ Kaggle credentials saved to {kaggle_json}")
        return True
    
    def download_kaggle_dataset(self):
        """Download waste classification dataset from Kaggle."""
        print("\n" + "="*60)
        print("DOWNLOADING KAGGLE DATASET")
        print("="*60)
        
        # Try to import kaggle
        try:
            import kaggle
        except ImportError:
            print("Installing Kaggle API...")
            os.system(f"{sys.executable} -m pip install kaggle")
            import kaggle
        
        # Setup credentials
        if not self.setup_kaggle_credentials():
            print("\n⚠ Kaggle setup skipped. Cannot download dataset.")
            return False
        
        # Create dataset directory
        dataset_path = Path(self.dataset_path)
        if dataset_path.exists():
            print(f"✓ Dataset directory already exists: {dataset_path}")
            # Check if data already downloaded
            subdirs = list(dataset_path.glob('*/'))
            if subdirs:
                print(f"✓ Found existing data: {[d.name for d in subdirs[:5]]}")
                return True
        
        dataset_path.mkdir(exist_ok=True)
        
        # Popular waste classification datasets
        datasets = [
            "techsash/waste-classification-data",
            "mostafaabla/garbage-classification",
            "asdasdasasdas/garbage-classification"
        ]
        
        print("\nAvailable datasets:")
        for i, ds in enumerate(datasets, 1):
            print(f"{i}. {ds}")
        
        choice = input(f"\nSelect dataset (1-{len(datasets)}) or press Enter for default: ").strip()
        
        if not choice:
            choice = "1"
        
        try:
            dataset_name = datasets[int(choice) - 1]
        except:
            dataset_name = datasets[0]
        
        print(f"\nDownloading: {dataset_name}")
        print("This may take a few minutes...")
        
        try:
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(dataset_path),
                unzip=True
            )
            print("✓ Dataset downloaded successfully!")
            return True
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def organize_dataset(self):
        """Organize downloaded dataset into train/val structure."""
        print("\n" + "="*60)
        print("ORGANIZING DATASET")
        print("="*60)
        
        dataset_root = Path(self.dataset_path)
        
        # Find all directories with images
        all_dirs = [d for d in dataset_root.rglob('*') if d.is_dir()]
        
        # Category mapping for different dataset structures
        organic_keywords = ['organic', 'compost', 'biological', 'food', 'o']
        recyclable_keywords = ['recyclable', 'recycle', 'plastic', 'paper', 
                              'metal', 'glass', 'cardboard', 'r']
        
        # Prepare organized structure
        train_dir = dataset_root / 'organized' / 'train'
        val_dir = dataset_root / 'organized' / 'val'
        
        for split_dir in [train_dir, val_dir]:
            for class_name in self.classes:
                (split_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        image_count = {'organic': 0, 'recyclable': 0}
        
        # Process each directory
        for dir_path in all_dirs:
            if 'organized' in str(dir_path):
                continue
            
            dir_name = dir_path.name.lower()
            
            # Determine category
            category = None
            if any(keyword in dir_name for keyword in organic_keywords):
                category = 'organic'
            elif any(keyword in dir_name for keyword in recyclable_keywords):
                category = 'recyclable'
            else:
                continue  # Skip unknown categories
            
            # Find all images in directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(dir_path.glob(ext))
                image_files.extend(dir_path.glob(ext.upper()))
            
            if not image_files:
                continue
            
            # Split into train/val
            train_files, val_files = train_test_split(
                image_files, test_size=0.2, random_state=42
            )
            
            # Copy files
            for file_list, target_base in [(train_files, train_dir), (val_files, val_dir)]:
                target_class_dir = target_base / category
                
                for img_file in file_list:
                    target_file = target_class_dir / f"{dir_name}_{img_file.name}"
                    if not target_file.exists():
                        shutil.copy2(img_file, target_file)
                        image_count[category] += 1
        
        print(f"\n✓ Dataset organized:")
        print(f"  - Organic: {image_count['organic']} images")
        print(f"  - Recyclable: {image_count['recyclable']} images")
        
        # Check if we have enough data
        if image_count['organic'] == 0 or image_count['recyclable'] == 0:
            print("\n⚠ Warning: Some categories have no images!")
            print("The dataset structure might be different. Attempting alternative organization...")
            return self.organize_dataset_alternative()
        
        return True
    
    def organize_dataset_alternative(self):
        """Alternative organization for different dataset structures."""
        dataset_root = Path(self.dataset_path)
        
        # Look for TRAIN/TEST structure
        train_test_dirs = []
        for possible_name in ['TRAIN', 'TEST', 'train', 'test', 'training', 'testing']:
            possible_dir = dataset_root / possible_name
            if possible_dir.exists():
                train_test_dirs.append(possible_dir)
        
        if train_test_dirs:
            print(f"Found structured directories: {[d.name for d in train_test_dirs]}")
            
            # Create organized structure
            train_dir = dataset_root / 'organized' / 'train'
            val_dir = dataset_root / 'organized' / 'val'
            
            for split_dir in [train_dir, val_dir]:
                for class_name in self.classes:
                    (split_dir / class_name).mkdir(parents=True, exist_ok=True)
            
            # Use first dir as training source
            if train_test_dirs:
                source_dir = train_test_dirs[0]
                
                # Map O->organic, R->recyclable
                class_mapping = {'o': 'organic', 'r': 'recyclable'}
                
                for subdir in source_dir.iterdir():
                    if not subdir.is_dir():
                        continue
                    
                    dir_key = subdir.name.lower()[0]  # First letter
                    if dir_key in class_mapping:
                        category = class_mapping[dir_key]
                        
                        # Get images
                        image_files = []
                        for ext in ['*.jpg', '*.jpeg', '*.png']:
                            image_files.extend(subdir.rglob(ext))
                        
                        # Split and copy
                        train_files, val_files = train_test_split(
                            image_files, test_size=0.2, random_state=42
                        )
                        
                        for file_list, target_base in [(train_files, train_dir), (val_files, val_dir)]:
                            for img_file in file_list:
                                target = target_base / category / img_file.name
                                if not target.exists():
                                    shutil.copy2(img_file, target)
                
                print("✓ Alternative organization completed")
                return True
        
        print("✗ Could not organize dataset automatically")
        return False
    
    def build_model(self):
        """Build custom CNN model for waste classification."""
        print("\n" + "="*60)
        print("BUILDING CUSTOM CNN MODEL")
        print("="*60)
        
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Classification Head - Use GlobalAveragePooling2D instead of Flatten
            tf.keras.layers.GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(len(self.classes), activation='softmax')
        ])
        
        # Use simple Adam optimizer (no mixed precision to avoid issues)
        optimizer = Adam(learning_rate=0.001)
            
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✓ Custom CNN built with {self.model.count_params():,} parameters")
        if self.gpu_available:
            print("✓ Model configured for GPU training")
        else:
            print("⚠ Model configured for CPU training")
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=30, batch_size=32):
        """Train the CNN model on waste dataset."""
        print("\n" + "="*60)
        print("TRAINING CNN MODEL")
        print("="*60)
        
        if not self.model:
            self.build_model()
        
        train_dir = Path(self.dataset_path) / 'organized' / 'train'
        val_dir = Path(self.dataset_path) / 'organized' / 'val'
        
        if not train_dir.exists():
            print("✗ Training data not found. Run dataset download first.")
            return None
        
        # Data generators with augmentation (stronger for CNN training)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.3,
            shear_range=0.2,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.classes
        )
        
        val_generator = val_datagen.flow_from_directory(
            str(val_dir),
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.classes
        )
        
        print(f"\n✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=4,
                verbose=1,
                min_lr=1e-7
            )
        ]
        
        # Check GPU before training
        if self.gpu_available:
            print("🚀 Starting GPU-accelerated training...")
            check_gpu_utilization()
            run_gpu_test()
        else:
            print("🐌 Starting CPU training (consider installing tensorflow[and-cuda] for GPU acceleration)")
        
        # Train
        print(f"\nTraining for {epochs} epochs...")
        try:
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks
            )
        except Exception as e:
            print(f"❌ Training failed: {e}")
            if "CUDA" in str(e) or "GPU" in str(e):
                print("🔧 GPU error detected. Try:")
                print("1. Restart Python")
                print("2. pip install tensorflow[and-cuda]")
                print("3. Update NVIDIA drivers")
            raise
        
        print(f"\n✓ Training completed!")
        print(f"✓ Best model saved to: {self.model_path}")
        
        # Check final GPU utilization
        if self.gpu_available:
            check_gpu_utilization()
        
        # Plot training history
        self.plot_history(history)
        
        return history
    
    def plot_history(self, history):
        """Plot training history with enhanced visualization."""
        print("\n📊 Generating training plots...")
        
        # Normalize history to a simple dict of lists
        hist = history.history if hasattr(history, 'history') else history
        if not isinstance(hist, dict):
            print("❌ Unexpected history format; cannot plot.")
            return

        # Utility: pick best-matching keys for metrics
        def pick_key(candidates, default=None):
            for k in candidates:
                if k in hist:
                    return k
            return default

        # Resolve accuracy keys robustly
        acc_key = pick_key([
            'accuracy',
            'acc',
            'categorical_accuracy',
            'binary_accuracy',
            'sparse_categorical_accuracy'
        ])
        val_acc_key = pick_key([
            'val_accuracy',
            'val_acc',
            f"val_{acc_key}" if acc_key else None
        ])

        # Resolve loss keys
        loss_key = pick_key(['loss']) or next((k for k in hist.keys() if k.endswith('loss') and not k.startswith('val')), None)
        val_loss_key = pick_key(['val_loss', f"val_{loss_key}" if loss_key else None]) or next((k for k in hist.keys() if k.startswith('val') and k.endswith('loss')), None)

        # Resolve learning rate keys (optional)
        lr_key = pick_key(['lr', 'learning_rate'])

        # Extract series (default to empty list if missing)
        acc = hist.get(acc_key, []) if acc_key else []
        val_acc = hist.get(val_acc_key, []) if val_acc_key else []
        loss = hist.get(loss_key, []) if loss_key else []
        val_loss = hist.get(val_loss_key, []) if val_loss_key else []
        lrs = hist.get(lr_key, []) if lr_key else []

        # Debug: print detected keys and lengths to help diagnose blank plots
        print("Detected history keys:", list(hist.keys()))
        print(f"Using keys -> acc: {acc_key}, val_acc: {val_acc_key}, loss: {loss_key}, val_loss: {val_loss_key}, lr: {lr_key}")
        print(f"Lengths -> acc: {len(acc)}, val_acc: {len(val_acc)}, loss: {len(loss)}, val_loss: {len(val_loss)}, lr: {len(lrs)}")

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CNN Waste Classifier Training Results', fontsize=16, fontweight='bold')

        # Plot 1: Accuracy
        if len(acc) or len(val_acc):
            if len(acc):
                axes[0, 0].plot(acc, 'b-', label='Training Accuracy', linewidth=2)
            if len(val_acc):
                axes[0, 0].plot(val_acc, 'r-', label='Validation Accuracy', linewidth=2)
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, 'No Accuracy Metrics Found', ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12, alpha=0.7)
        axes[0, 0].set_title('Model Accuracy Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Loss
        if len(loss) or len(val_loss):
            if len(loss):
                axes[0, 1].plot(loss, 'b-', label='Training Loss', linewidth=2)
            if len(val_loss):
                axes[0, 1].plot(val_loss, 'r-', label='Validation Loss', linewidth=2)
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, 'No Loss Metrics Found', ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12, alpha=0.7)
        axes[0, 1].set_title('Model Loss Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Learning Rate (if available)
        if len(lrs):
            axes[1, 0].plot(lrs, 'g-', linewidth=2)
            axes[1, 0].set_yscale('log')
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                            ha='center', va='center', transform=axes[1, 0].transAxes,
                            fontsize=12, alpha=0.7)
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Training Summary
        def safe_last(values):
            try:
                return float(values[-1]) if len(values) else None
            except Exception:
                return None

        final_train_acc = safe_last(acc)
        final_val_acc = safe_last(val_acc)
        best_val_acc = float(max(val_acc)) if len(val_acc) else None

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "N/A"

        epochs_len = max(len(acc), len(val_acc), len(loss), len(val_loss))
        summary_text = f"""CNN Training Summary:
        
Final Training Accuracy: {fmt(final_train_acc)}
Final Validation Accuracy: {fmt(final_val_acc)}
Best Validation Accuracy: {fmt(best_val_acc)}

Total Epochs: {epochs_len}
Classes: {', '.join(self.classes)}
Input Shape: {self.input_shape}"""

        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        axes[1, 1].set_title('Training Summary', fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('cnn_training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history saved to cnn_training_history.png")
        plt.show()

        # Save standardized training history data
        history_data = {
            'accuracy': list(map(float, acc)) if len(acc) else [],
            'val_accuracy': list(map(float, val_acc)) if len(val_acc) else [],
            'loss': list(map(float, loss)) if len(loss) else [],
            'val_loss': list(map(float, val_loss)) if len(val_loss) else [],
            'lr': list(map(float, lrs)) if len(lrs) else [],
            'epochs': epochs_len,
            'model_type': 'custom_cnn',
            'input_shape': self.input_shape,
            'final_metrics': {
                'train_accuracy': float(final_train_acc) if final_train_acc is not None else None,
                'val_accuracy': float(final_val_acc) if final_val_acc is not None else None,
                'best_val_accuracy': float(best_val_acc) if best_val_acc is not None else None
            }
        }

        with open('cnn_training_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
        print("✓ Training data saved to cnn_training_history.json")
    
    def show_training_plot(self):
        """Display saved training history plot."""
        print("\n📊 Loading CNN training history...")
        
        # Try to load training history data
        if os.path.exists('cnn_training_history.json'):
            with open('cnn_training_history.json', 'r') as f:
                history_data = json.load(f)
            
            # Warn if saved arrays are empty
            try:
                acc_len = len(history_data.get('accuracy', []))
                val_acc_len = len(history_data.get('val_accuracy', []))
                loss_len = len(history_data.get('loss', []))
                val_loss_len = len(history_data.get('val_loss', []))
                print(f"Saved history lengths -> acc: {acc_len}, val_acc: {val_acc_len}, loss: {loss_len}, val_loss: {val_loss_len}")
                if acc_len == 0 and val_acc_len == 0 and loss_len == 0 and val_loss_len == 0:
                    print("⚠ Saved history arrays are empty. The previous training may not have run, or the file is stale.")
                    print("Tip: Retrain the model or delete cnn_training_history.json and run training again.")
            except Exception:
                pass

            # Create mock history object for plotting
            class MockHistory:
                def __init__(self, data):
                    self.history = data
            
            mock_history = MockHistory(history_data)
            self.plot_history(mock_history)
            return True
        
        # Fallback: try to show saved PNG
        elif os.path.exists('cnn_training_history.png'):
            print("✓ Displaying saved CNN training plot...")
            import matplotlib.image as mpimg
            
            img = mpimg.imread('cnn_training_history.png')
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Saved CNN Training History', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()
            return True
        
        else:
            print("❌ No CNN training history found!")
            print("Train the CNN model first to generate training plots.")
            return False
    
    def load_trained_model(self):
        """Load previously trained CNN model."""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✓ Loaded trained CNN model from {self.model_path}")
            return True
        return False
    
    def classify_image(self, image_path):
        """Classify a single image using CNN."""
        if not self.model:
            if not self.load_trained_model():
                print("✗ No trained CNN model available!")
                return None
        
        # Load and preprocess image (no VGG16 preprocessing needed)
        img = load_img(image_path, target_size=self.input_shape[:2])
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Simple normalization for CNN
        
        # Predict
        prediction = self.model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        predicted_class = self.classes[class_idx]
        
        # Calculate dry waste percentage
        recyclable_percentage = prediction[0][self.classes.index('recyclable')] * 100
        
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'dry_waste_percentage': float(recyclable_percentage),
            'is_dry_waste': predicted_class == 'recyclable',
            'all_probabilities': {
                self.classes[i]: float(prediction[0][i] * 100) 
                for i in range(len(self.classes))
            }
        }
        
        return result
    
    def analyze_image_with_visualization(self, image_path):
        """Analyze and visualize image classification using CNN."""
        result = self.classify_image(image_path)
        
        if not result:
            return
        
        # Display results
        print("\n" + "="*60)
        print("CNN CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Image: {image_path}")
        print(f"Class: {result['predicted_class'].upper()}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Dry Waste (Recyclable): {result['dry_waste_percentage']:.2f}%")
        print(f"Organic: {result['all_probabilities']['organic']:.2f}%")
        
        # Visualize
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Image: {Path(image_path).name}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        classes = list(result['all_probabilities'].keys())
        percentages = list(result['all_probabilities'].values())
        colors = ['green' if c == 'recyclable' else 'brown' for c in classes]
        
        plt.bar(classes, percentages, color=colors, alpha=0.7)
        plt.ylabel('Percentage (%)')
        plt.title('CNN Classification Results')
        plt.ylim(0, 100)
        
        for i, (cls, pct) in enumerate(zip(classes, percentages)):
            plt.text(i, pct + 3, f'{pct:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return result

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("CNN WASTE CLASSIFIER WITH KAGGLE DATA")
    print("="*60)
    
    classifier = CNNWasteClassifier()
    
    print("\nWhat would you like to do?")
    print("1. Download dataset and train CNN model")
    print("2. Train CNN with existing downloaded dataset")  
    print("3. Classify an image (requires trained CNN model)")
    print("4. 📊 Show CNN training graphs/plots")
    print("5. 🔧 GPU diagnostics and test")
    print("6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        # Download and train
        if classifier.download_kaggle_dataset():
            classifier.organize_dataset()
            
            epochs = input("\nNumber of training epochs (default 30): ").strip()
            epochs = int(epochs) if epochs else 30
            
            classifier.train_model(epochs=epochs)
            
            test_image = input("\nTest with an image? Enter path (or skip): ").strip()
            if test_image and os.path.exists(test_image):
                classifier.analyze_image_with_visualization(test_image)
    
    elif choice == "2":
        # Train with existing data
        if Path(classifier.dataset_path).exists():
            if not (Path(classifier.dataset_path) / 'organized').exists():
                classifier.organize_dataset()
            
            epochs = input("\nNumber of training epochs (default 30): ").strip()
            epochs = int(epochs) if epochs else 30
            
            classifier.train_model(epochs=epochs)
        else:
            print("✗ Dataset not found. Please download first (option 1)")
    
    elif choice == "3":
        # Classify image
        image_path = input("\nEnter image path: ").strip()
        
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return
        
        classifier.analyze_image_with_visualization(image_path)
        
        # Save results
        result = classifier.classify_image(image_path)
        with open('cnn_classification_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\n✓ Results saved to cnn_classification_result.json")
    
    elif choice == "4":
        # Show training plots
        print("\n📊 CNN TRAINING VISUALIZATION")
        print("="*60)
        classifier.show_training_plot()
    
    elif choice == "5":
        # GPU diagnostics
        print("\n🔧 GPU DIAGNOSTICS")
        print("="*60)
        setup_gpu()
        run_gpu_test()
        check_gpu_utilization()
        
        print("\n💡 If no GPU detected:")
        print("1. Install CUDA-enabled TensorFlow:")
        print("   pip uninstall tensorflow")
        print("   pip install 'tensorflow[and-cuda]==2.20.*'")
        print("2. Check NVIDIA driver: nvidia-smi")
        print("3. Restart Python after installation")
        print("4. Make sure you have an NVIDIA GPU")
    
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()