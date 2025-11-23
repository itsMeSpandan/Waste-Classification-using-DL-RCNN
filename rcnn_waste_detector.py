"""
R-CNN Waste Object Detector with Heatbox Visualization
This script uses Region-based CNN (R-CNN) for multi-object waste detection
and provides visual heatboxes showing detected waste objects with confidence scores.
Enhanced with DRY/WET waste classification for practical sorting applications.
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# GPU Setup Functions
def setup_gpu():
    """Setup and configure GPU for TensorFlow."""
    print("\n" + "="*60)
    print("GPU SETUP AND DETECTION")
    print("="*60)
    
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    physical_devices = tf.config.list_physical_devices()
    print("\nPhysical devices:")
    for device in physical_devices:
        print(f"  {device}")
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"\nGPU devices found: {len(gpu_devices)}")
    
    if gpu_devices:
        print("✓ GPU(s) detected!")
        for i, gpu in enumerate(gpu_devices):
            print(f"  GPU {i}: {gpu}")
            
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth configured")
        except Exception as e:
            print(f"⚠ Could not configure GPU memory growth: {e}")
        
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
                print(f"  GPU {i}: Available")
        else:
            print("No GPU devices available for monitoring")
    except Exception as e:
        print(f"Could not check GPU utilization: {e}")

class RCNNWasteDetector:
    """R-CNN based model for waste object detection with enhanced heatbox visualization."""
    
    def __init__(self):
        self.trained_cnn = None  # Will hold your pre-trained CNN
        self.use_trained_cnn = False  # Flag to use real vs mock detection
        self.cnn_input_shape = (128, 128, 3)  # Your CNN's input size
        self.classes = ['background', 'organic', 'recyclable']  # Match CNN classifier
        self.cnn_classes = ['organic', 'recyclable']  # Your actual CNN classes
        self.cnn_model_path = "cnn_waste_model.h5"  # Use the main trained model
        self.gpu_available = False
        
        # R-CNN specific parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.3
        self.region_proposals_count = 100  # Number of region proposals to generate
        self.min_box_size = 20  # Minimum bounding box size
        
        # Enhanced colors for different waste types with RGB values
        self.class_colors = {
            'background': (128, 128, 128),  # Gray
            'organic': (139, 69, 19),       # Brown
            'recyclable': (30, 144, 255)    # Dodger Blue (for all recyclables)
        }
        
        # Waste classification mapping (binary: organic vs recyclable)
        self.waste_classification = {
            'organic': {'type': 'wet', 'recyclable': False, 'category': 'Organic'},
            'recyclable': {'type': 'dry', 'recyclable': True, 'category': 'Recyclable'},
            'background': {'type': 'none', 'recyclable': False, 'category': 'Background'}
        }
        
        self.gpu_available = setup_gpu()
        
        # Load your pre-trained CNN model automatically
        self.use_trained_cnn = self.load_trained_cnn_model()
        
    def load_trained_cnn_model(self):
        """Load your pre-trained CNN model for real classification."""
        print("\n🔄 Loading your trained CNN model...")
        
        if not os.path.exists(self.cnn_model_path):
            print(f"❌ {self.cnn_model_path} not found!")
            print("   Will use mock detections for demonstration.")
            print("   Train your CNN first using cnn_waste_classifier.py")
            return False
        
        try:
            # Try multiple loading methods for compatibility
            print("   Attempting primary model loading...")
            self.trained_cnn = tf.keras.models.load_model(self.cnn_model_path)
            
        except Exception as primary_error:
            print(f"   Primary loading failed: {primary_error}")
            print("   Trying alternative loading methods...")
            
            # Try loading with compile=False
            try:
                print("   Attempting loading without compilation...")
                self.trained_cnn = tf.keras.models.load_model(self.cnn_model_path, compile=False)
                
                # Recompile manually
                self.trained_cnn.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                print("   ✓ Model loaded without compilation and recompiled")
                
            except Exception as secondary_error:
                print(f"   Alternative loading failed: {secondary_error}")
                
                # Try loading weights only (requires model recreation)
                try:
                    print("   Attempting to recreate model and load weights...")
                    weights_path = self.cnn_model_path.replace('.h5', '_weights.h5')
                    
                    if os.path.exists(weights_path):
                        # Create a simple CNN architecture to load weights into
                        self.trained_cnn = self._create_fallback_model()
                        self.trained_cnn.load_weights(weights_path)
                        print("   ✓ Model recreated and weights loaded")
                    else:
                        raise Exception("No weights file found")
                        
                except Exception as tertiary_error:
                    print(f"   All loading methods failed: {tertiary_error}")
                    print("   Recommendations:")
                    print("   1. Retrain your model with current TensorFlow version")
                    print("   2. Use a different model file (check other .h5 files)")
                    print("   3. Use mock detection for now")
                    print("   Will use mock detections for demonstration.")
                    return False
        
        # If we get here, model loaded successfully
        try:
            print(f"✅ Loaded trained CNN model: {self.cnn_model_path}")
            print(f"   Parameters: {self.trained_cnn.count_params():,}")
            print(f"   Input shape: {self.trained_cnn.input_shape}")
            print(f"   Output shape: {self.trained_cnn.output_shape}")
            print(f"   Expected classes: {self.cnn_classes}")
            
            # Validate model output shape
            expected_output = len(self.cnn_classes)
            actual_output = self.trained_cnn.output_shape[-1]
            if actual_output != expected_output:
                print(f"⚠ Warning: Model outputs {actual_output} classes, expected {expected_output}")
            
            return True
            
        except Exception as info_error:
            print(f"⚠ Model loaded but info extraction failed: {info_error}")
            print("   Model should still work for predictions")
            return True
    
    def _create_fallback_model(self):
        """Create a fallback CNN model architecture for loading weights."""
        print("   Creating fallback CNN model architecture...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.cnn_input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.cnn_classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def classify_region_with_trained_cnn(self, image_region):
        """Use YOUR trained CNN to classify each region proposal."""
        try:
            # Resize region to your CNN's input size (128x128)
            region_resized = cv2.resize(image_region, self.cnn_input_shape[:2])
            
            # Normalize exactly like cnn_waste_classifier.py does
            region_normalized = region_resized.astype(np.float32) / 255.0
            region_batch = np.expand_dims(region_normalized, axis=0)
            
            # Use YOUR trained CNN model (same as cnn_waste_classifier.py)
            prediction = self.trained_cnn.predict(region_batch, verbose=0)
            
            # CNN outputs [organic_prob, recyclable_prob] with softmax
            organic_prob = float(prediction[0][0])
            recyclable_prob = float(prediction[0][1])
            
            # Get the maximum confidence
            max_confidence = max(organic_prob, recyclable_prob)
            
            # Accept predictions with confidence > 0.5 (trust the trained model)
            if max_confidence < 0.5:
                return 'background', max_confidence
            
            # Binary classification: organic or recyclable
            if organic_prob > recyclable_prob:
                class_name = 'organic'
                confidence = organic_prob
            else:
                class_name = 'recyclable'
                confidence = recyclable_prob
            
            return class_name, confidence
            
        except Exception as e:
            print(f"⚠ CNN classification error: {e}")
            return 'background', 0.3
    
    def test_cnn_model(self):
        """Test the loaded CNN model with a dummy input."""
        if not self.use_trained_cnn:
            print("❌ No CNN model loaded to test")
            return False
            
        try:
            print("\n🧪 Testing CNN model with dummy input...")
            # Create dummy input matching the CNN's expected shape
            dummy_input = np.random.rand(1, *self.cnn_input_shape).astype(np.float32)
            
            # Test prediction
            prediction = self.trained_cnn.predict(dummy_input, verbose=0)
            organic_prob = float(prediction[0][0])
            recyclable_prob = float(prediction[0][1])
            
            print(f"✅ CNN model test successful!")
            print(f"   Dummy prediction: Organic={organic_prob:.3f}, Recyclable={recyclable_prob:.3f}")
            print(f"   Prediction sum: {organic_prob + recyclable_prob:.3f} (should be ~1.0 for softmax)")
            
            # Validate softmax output
            if abs((organic_prob + recyclable_prob) - 1.0) > 0.01:
                print(f"⚠ Warning: Prediction probabilities don't sum to 1.0")
                
            return True
            
        except Exception as e:
            print(f"❌ CNN model test failed: {e}")
            return False
        
    def generate_region_proposals(self, image):
        """Generate region proposals using selective search approach."""
        height, width = image.shape[:2]
        proposals = []
        
        # Strategy 1: Grid-based proposals with multiple scales
        scales = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        grid_sizes = [3, 4, 5, 6]
        
        for scale in scales:
            for grid_size in grid_sizes:
                step_x = width // grid_size
                step_y = height // grid_size
                box_w = int(width * scale)
                box_h = int(height * scale)
                
                for i in range(grid_size + 1):
                    for j in range(grid_size + 1):
                        x = min(i * step_x, width - box_w)
                        y = min(j * step_y, height - box_h)
                        
                        if box_w >= self.min_box_size and box_h >= self.min_box_size:
                            proposals.append([x, y, x + box_w, y + box_h])
        
        # Strategy 2: Random proposals for diversity
        for _ in range(50):
            x1 = random.randint(0, width - self.min_box_size)
            y1 = random.randint(0, height - self.min_box_size)
            
            max_w = min(width - x1, width // 2)
            max_h = min(height - y1, height // 2)
            
            w = random.randint(self.min_box_size, max_w)
            h = random.randint(self.min_box_size, max_h)
            
            x2 = x1 + w
            y2 = y1 + h
            
            proposals.append([x1, y1, x2, y2])
        
        # Strategy 3: Edge-based proposals
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours from edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    if w >= self.min_box_size and h >= self.min_box_size:
                        # Add some padding
                        padding = 10
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(width, x + w + padding)
                        y2 = min(height, y + h + padding)
                        proposals.append([x1, y1, x2, y2])
        except Exception as e:
            print(f"⚠ Edge detection failed: {e}")
        
        # Remove duplicates and limit number of proposals
        unique_proposals = []
        for prop in proposals:
            if prop not in unique_proposals:
                unique_proposals.append(prop)
        
        # Sort by area (larger first) and take top N
        unique_proposals.sort(key=lambda x: (x[2]-x[0]) * (x[3]-x[1]), reverse=True)
        
        return unique_proposals[:self.region_proposals_count]
    
    def detect_objects_with_real_cnn(self, image_path):
        """Detect objects using region proposals + YOUR trained CNN."""
        print(f"\n🔍 R-CNN Detection with YOUR trained CNN: {Path(image_path).name}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        
        print("  📍 Generating region proposals...")
        # Generate region proposals
        proposals = self.generate_region_proposals(img_rgb)
        print(f"  ✓ Generated {len(proposals)} region proposals")
        
        detections = []
        
        # Classify each region with YOUR CNN
        print(f"  🧠 Classifying regions with your trained CNN...")
        for i, proposal in enumerate(proposals):
            x1, y1, x2, y2 = proposal
            
            # Ensure valid coordinates
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            
            # Extract region
            region = img_rgb[y1:y2, x1:x2]
            if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
                continue
            
            # Use YOUR CNN to classify this region
            class_name, confidence = self.classify_region_with_trained_cnn(region)
            
            # Filter low confidence detections and background
            if confidence > self.confidence_threshold and class_name != 'background':
                waste_info = self.waste_classification[class_name]
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': confidence,
                    'object_id': len(detections) + 1,
                    'waste_type': waste_info['type'],
                    'is_recyclable': waste_info['recyclable'],
                    'waste_category': waste_info['category'],
                    'detection_method': 'R-CNN + Your Trained CNN'
                }
                
                detections.append(detection)
        
        print(f"  ✅ Found {len(detections)} objects using your trained CNN")
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections, img_rgb
    
    def detect_objects_rcnn(self, image_path):
        """Detect objects using R-CNN approach with trained CNN classifier."""
        print(f"\n🔍 R-CNN Object Detection: {Path(image_path).name}")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img_rgb.shape[:2]
        
        print("  📍 Generating region proposals...")
        # Generate region proposals
        proposals = self.generate_region_proposals(img_rgb)
        print(f"  ✓ Generated {len(proposals)} region proposals")
        
        # Use real CNN for classification if available
        if self.use_trained_cnn:
            detections = self._classify_proposals_with_cnn(img_rgb, proposals)
        else:
            print("  ⚠️ No trained CNN model - using fallback detection")
            detections = self.create_enhanced_detections(original_width, original_height, proposals)
        
        return detections, img_rgb
    
    def _classify_proposals_with_cnn(self, image_rgb, proposals):
        """Classify region proposals using trained CNN model."""
        detections = []
        height, width = image_rgb.shape[:2]
        
        print(f"  🧠 Classifying {len(proposals)} regions with trained CNN...")
        
        for i, proposal in enumerate(proposals):
            x1, y1, x2, y2 = proposal
            
            # Ensure valid coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Extract region
            region = image_rgb[y1:y2, x1:x2]
            if region.size == 0 or region.shape[0] < 20 or region.shape[1] < 20:
                continue
            
            # Use CNN to classify this region
            class_name, confidence = self.classify_region_with_trained_cnn(region)
            
            # Filter low confidence and background
            if confidence > self.confidence_threshold and class_name != 'background':
                waste_info = self.waste_classification[class_name]
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'class_id': self.classes.index(class_name),
                    'confidence': confidence,
                    'object_id': len(detections) + 1,
                    'waste_type': waste_info['type'],
                    'is_recyclable': waste_info['recyclable'],
                    'waste_category': waste_info['category'],
                    'detection_method': 'R-CNN + Trained CNN'
                }
                
                detections.append(detection)
        
        print(f"  ✅ Found {len(detections)} objects using trained CNN")
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def create_enhanced_detections(self, width, height, proposals):
        """Create enhanced mock detections for demonstration."""
        detections = []
        
        # Generate 10-15 high-quality detections
        num_objects = random.randint(10, 15)
        print(f"  🎯 Generating {num_objects} enhanced object detections...")
        
        # Use some proposals and generate some custom ones
        selected_proposals = random.sample(proposals, min(len(proposals), num_objects // 2))
        
        for i in range(num_objects):
            if i < len(selected_proposals):
                # Use region proposal
                bbox = selected_proposals[i]
                x1, y1, x2, y2 = bbox
                
                # Add some randomness to the bbox
                noise_x = random.randint(-10, 10)
                noise_y = random.randint(-10, 10)
                noise_w = random.randint(-5, 15)
                noise_h = random.randint(-5, 15)
                
                x1 = max(0, x1 + noise_x)
                y1 = max(0, y1 + noise_y)
                x2 = min(width, x2 + noise_w)
                y2 = min(height, y2 + noise_h)
            else:
                # Generate new detection
                scale = random.choice([0.08, 0.12, 0.18, 0.25, 0.35])
                x1 = random.randint(0, int(width * 0.7))
                y1 = random.randint(0, int(height * 0.7))
                
                box_w = int(width * scale)
                box_h = int(height * scale)
                
                x2 = min(width, x1 + box_w)
                y2 = min(height, y1 + box_h)
            
            # Ensure valid bounding box
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Assign class with realistic distribution
            if i % 5 == 0:
                class_idx = 1  # organic
            elif i % 5 == 1 or i % 5 == 2:
                class_idx = 2  # plastic (more common)
            elif i % 5 == 3:
                class_idx = random.choice([3, 4, 5])  # metal, paper, glass
            else:
                class_idx = random.randint(1, len(self.classes) - 1)
            
            class_name = self.classes[class_idx]
            
            # Realistic confidence scores with R-CNN characteristics
            if i < 3:  # High confidence detections
                confidence = random.uniform(0.88, 0.97)
            elif i < 6:  # Medium-high confidence
                confidence = random.uniform(0.75, 0.88)
            elif i < 9:  # Medium confidence
                confidence = random.uniform(0.62, 0.75)
            else:  # Lower but valid confidence
                confidence = random.uniform(0.51, 0.68)
            
            # Get waste classification
            waste_info = self.waste_classification[class_name]
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'class': class_name,
                'class_id': class_idx,
                'confidence': confidence,
                'object_id': i + 1,
                'waste_type': waste_info['type'],
                'is_recyclable': waste_info['recyclable'],
                'waste_category': waste_info['category'],
                'detection_method': 'R-CNN'
            }
            
            detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def non_max_suppression_rcnn(self, detections, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to R-CNN detections."""
        if len(detections) == 0:
            return detections
        
        # Convert to format suitable for NMS
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Apply TensorFlow NMS
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=15, iou_threshold=iou_threshold
        ).numpy()
        
        # Return filtered detections
        filtered_detections = [detections[i] for i in selected_indices]
        
        print(f"  🔧 NMS: {len(detections)} → {len(filtered_detections)} detections")
        return filtered_detections
    
    def visualize_rcnn_detections(self, image_path, detections, save_path=None):
        """Visualize R-CNN detections with enhanced DRY/WET classification."""
        print("\n🎨 Generating R-CNN detection overview with DRY/WET classification...")
        
        # Load original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        image_name = Path(image_path).stem
        
        # Prepare bounding box image with enhanced labels
        bbox_img = img.copy()
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            object_id = detection.get('object_id', 0)
            waste_type = detection['waste_type']
            is_recyclable = detection['is_recyclable']
            
            color = self.class_colors.get(class_name, (255, 0, 0))
            
            # Enhanced border color based on waste type
            if waste_type == 'wet':
                border_color = (139, 69, 19)  # Brown for organic
                label_bg_color = (139, 69, 19)
            else:  # dry/recyclable
                border_color = (0, 191, 255)  # Deep sky blue for recyclable
                label_bg_color = (0, 191, 255)
            
            # Draw thick border
            cv2.rectangle(bbox_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), border_color, 5)
            
            # Enhanced label with waste classification
            recyclable_text = "[R]" if is_recyclable else "[W]"
            waste_status = "DRY" if waste_type == 'dry' else "WET"
            label = f"{recyclable_text} {waste_status}({class_name}) {confidence:.2f}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Background rectangle for label
            cv2.rectangle(bbox_img, 
                         (bbox[0], bbox[1] - label_size[1] - 15),
                         (bbox[0] + label_size[0] + 12, bbox[1]), 
                         label_bg_color, -1)
            
            # White text for good contrast
            cv2.putText(bbox_img, label, (bbox[0] + 6, bbox[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create advanced heatmap with multiple layers
        heatmap = np.zeros(img.shape[:2], dtype=np.float32)
        confidence_map = np.zeros(img.shape[:2], dtype=np.float32)
        
        # Multi-layer heatmap generation
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            waste_type = detection['waste_type']
            
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Create confidence-based heat with larger radius
            for y in range(max(0, y1-20), min(heatmap.shape[0], y2+20)):
                for x in range(max(0, x1-20), min(heatmap.shape[1], x2+20)):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    # Different heat patterns for dry vs wet waste
                    if waste_type == 'wet':
                        heat_value = confidence * np.exp(-distance / 60)  # Wider spread for organic
                    else:
                        heat_value = confidence * np.exp(-distance / 45)  # Focused spread for recyclable
                    
                    heatmap[y, x] += heat_value
                    confidence_map[y, x] = max(confidence_map[y, x], confidence)
        
        # Normalize heatmaps
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        if confidence_map.max() > 0:
            confidence_map = confidence_map / confidence_map.max()
        
        # Calculate comprehensive statistics
        if detections:
            dry_waste_count = sum(1 for d in detections if d['waste_type'] == 'dry')
            wet_waste_count = sum(1 for d in detections if d['waste_type'] == 'wet')
            recyclable_count = sum(1 for d in detections if d['is_recyclable'])
            total_objects = len(detections)
            
            dry_percentage = (dry_waste_count / total_objects) * 100 if total_objects > 0 else 0
            wet_percentage = (wet_waste_count / total_objects) * 100 if total_objects > 0 else 0
            recyclable_percentage = (recyclable_count / total_objects) * 100 if total_objects > 0 else 0
            
            # Class and confidence statistics
            class_counts = {}
            confidence_stats = []
            for detection in detections:
                class_name = detection['class']
                confidence = detection['confidence']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidence_stats.append(confidence)
            
            avg_confidence = np.mean(confidence_stats)
            max_confidence = max(confidence_stats)
            min_confidence = min(confidence_stats)
        
        # Generate comprehensive overview image
        overview_save_path = f"{image_name}_rcnn_complete_overview.jpg"
        
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(f'R-CNN Waste Detection & Classification Overview: {Path(image_path).name}', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Original image
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=14)
        axes[0, 0].axis('off')
        
        # 2. Enhanced detections with waste classification
        axes[0, 1].imshow(bbox_img)
        axes[0, 1].set_title('R-CNN Detections (DRY/WET Classification)', fontweight='bold', fontsize=14)
        axes[0, 1].axis('off')
        
        # 3. Advanced confidence heatmap
        axes[0, 2].imshow(img, alpha=0.5)
        heat_overlay = axes[0, 2].imshow(heatmap, cmap='hot', alpha=0.9, interpolation='gaussian')
        axes[0, 2].set_title('Enhanced Confidence Heatmap', fontweight='bold', fontsize=14)
        axes[0, 2].axis('off')
        plt.colorbar(heat_overlay, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 4. Waste type distribution chart
        if detections:
            waste_categories = []
            waste_counts = []
            waste_colors = []
            
            if dry_waste_count > 0:
                waste_categories.append(f'DRY\n(Recyclable)')
                waste_counts.append(dry_waste_count)
                waste_colors.append('#4CAF50')  # Green
            
            if wet_waste_count > 0:
                waste_categories.append(f'WET\n(Organic)')
                waste_counts.append(wet_waste_count)
                waste_colors.append('#FF5722')  # Red-orange
            
            bars = axes[1, 0].bar(waste_categories, waste_counts, color=waste_colors, alpha=0.8, width=0.6)
            axes[1, 0].set_title('Waste Classification Distribution', fontweight='bold', fontsize=14)
            axes[1, 0].set_ylabel('Object Count', fontsize=12)
            
            # Add count labels on bars
            for bar, count in zip(bars, waste_counts):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{count}\n({count/total_objects*100:.1f}%)', 
                               ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 5. Detailed class breakdown pie chart
        if detections and class_counts:
            class_names = list(class_counts.keys())
            counts = list(class_counts.values())
            colors = [np.array(self.class_colors.get(cls, (128, 128, 128))) / 255.0 for cls in class_names]
            
            wedges, texts, autotexts = axes[1, 1].pie(counts, labels=class_names, autopct='%1.1f%%', 
                                                     colors=colors, startangle=90, textprops={'fontsize': 10})
            axes[1, 1].set_title('Object Class Distribution', fontweight='bold', fontsize=14)
            
            # Enhance pie chart text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 6. Comprehensive summary
        if detections:
            summary_text = f"""R-CNN DETECTION SUMMARY
            
DETECTION STATISTICS:
   Total Objects: {total_objects}
   Avg Confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)
   Best Detection: {max_confidence:.3f} ({max_confidence*100:.1f}%)
   Lowest Detection: {min_confidence:.3f} ({min_confidence*100:.1f}%)

WASTE CLASSIFICATION:
   DRY (Recyclable): {dry_waste_count} objects ({dry_percentage:.1f}%)
   WET (Organic): {wet_waste_count} objects ({wet_percentage:.1f}%)
   Recyclability: {recyclable_percentage:.1f}%

CLASS BREAKDOWN:"""
            
            for class_name, count in class_counts.items():
                waste_info = self.waste_classification[class_name]
                waste_type = waste_info['type'].upper()
                recyclable = "[R]" if waste_info['recyclable'] else "[W]"
                percentage = (count / total_objects) * 100
                summary_text += f"\n   {recyclable} {class_name.upper()} ({waste_type}): {count} ({percentage:.1f}%)"
            
            summary_text += f"\n\nDETECTION METHOD: R-CNN"
            summary_text += f"\nRegion Proposals: {self.region_proposals_count}"
            summary_text += f"\nConfidence Threshold: {self.confidence_threshold}"
            
            axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, pad=1))
            axes[1, 2].set_title('Detection Summary', fontweight='bold', fontsize=14)
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(overview_save_path, dpi=300, bbox_inches='tight', format='jpeg')
        plt.close()
        print(f"✅ R-CNN overview saved: {overview_save_path}")
        
        return {
            'files_saved': 1,
            'image_name': image_name,
            'overview_file': overview_save_path,
            'dry_waste_count': dry_waste_count if detections else 0,
            'wet_waste_count': wet_waste_count if detections else 0,
            'recyclable_count': recyclable_count if detections else 0,
            'total_objects': total_objects if detections else 0
        }
    
    def analyze_image_with_rcnn(self, image_path, save_results=True):
        """Complete R-CNN analysis with enhanced detection and visualization."""
        print("\n" + "="*60)
        print("R-CNN ENHANCED WASTE OBJECT DETECTION")
        print("="*60)
        print(f"Analyzing: {Path(image_path).name}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        try:
            # Use R-CNN with trained CNN classifier (same approach as cnn_waste_classifier.py)
            if self.use_trained_cnn:
                print("🧠 Using R-CNN with YOUR trained CNN for real classification!")
            else:
                print("⚠️ CNN model not loaded - using fallback detection")
            
            detections, original_img = self.detect_objects_rcnn(image_path)
            
            if not detections:
                print("❌ No objects detected")
                return None
            
            # Apply Non-Maximum Suppression
            detections = self.non_max_suppression_rcnn(detections)
            
            # Print detailed detection results
            print(f"\n🎯 R-CNN Detection Results:")
            print("-" * 55)
            
            for i, detection in enumerate(detections, 1):
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                waste_type = detection['waste_type']
                is_recyclable = detection['is_recyclable']
                waste_category = detection['waste_category']
                object_id = detection['object_id']
                
                box_width = bbox[2] - bbox[0]
                box_height = bbox[3] - bbox[1]
                box_area = box_width * box_height
                
                recyclable_icon = "[R]" if is_recyclable else "[W]"
                waste_type_icon = "[WET]" if waste_type == 'wet' else "[DRY]"
                
                print(f"{recyclable_icon} Object {object_id}: {class_name.upper()} ({waste_type.upper()})")
                print(f"   Category: {waste_category}")
                print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                print(f"   Location: ({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})")
                print(f"   Size: {box_width}×{box_height} pixels ({box_area:,} px²)")
                print(f"   Recyclable: {'Yes' if is_recyclable else 'No'}")
                
                # Confidence level assessment
                if confidence > 0.9:
                    conf_level = "🟢 Excellent"
                elif confidence > 0.8:
                    conf_level = "🔵 Very Good"
                elif confidence > 0.7:
                    conf_level = "🟡 Good"
                elif confidence > 0.6:
                    conf_level = "🟠 Fair"
                else:
                    conf_level = "🔴 Low"
                
                print(f"   Quality: {conf_level}")
                print()
            
            # Generate comprehensive statistics
            total_objects = len(detections)
            dry_waste_count = sum(1 for d in detections if d['waste_type'] == 'dry')
            wet_waste_count = sum(1 for d in detections if d['waste_type'] == 'wet')
            recyclable_count = sum(1 for d in detections if d['is_recyclable'])
            
            confidences = [d['confidence'] for d in detections]
            avg_confidence = np.mean(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)
            
            # Class statistics
            class_stats = {}
            for detection in detections:
                class_name = detection['class']
                if class_name not in class_stats:
                    class_stats[class_name] = {
                        'count': 0, 
                        'confidences': [], 
                        'waste_type': detection['waste_type'],
                        'recyclable': detection['is_recyclable']
                    }
                class_stats[class_name]['count'] += 1
                class_stats[class_name]['confidences'].append(detection['confidence'])
            
            print(f"📊 R-CNN COMPREHENSIVE ANALYSIS:")
            print("=" * 55)
            print(f"🎯 Detection Summary:")
            print(f"   Total Objects: {total_objects}")
            print(f"   Average Confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
            print(f"   Best Detection: {max_confidence:.3f} ({max_confidence*100:.1f}%)")
            print(f"   Lowest Detection: {min_confidence:.3f} ({min_confidence*100:.1f}%)")
            print()
            
            print(f"♻️ Waste Classification:")
            dry_percentage = (dry_waste_count / total_objects) * 100
            wet_percentage = (wet_waste_count / total_objects) * 100
            recyclable_percentage = (recyclable_count / total_objects) * 100
            
            print(f"   [DRY] DRY (Recyclable): {dry_waste_count} objects ({dry_percentage:.1f}%)")
            print(f"   [WET] WET (Organic): {wet_waste_count} objects ({wet_percentage:.1f}%)")
            print(f"   [R] Total Recyclable: {recyclable_count} objects ({recyclable_percentage:.1f}%)")
            print()
            
            print(f"📋 Class-wise Analysis:")
            for class_name, stats in class_stats.items():
                count = stats['count']
                avg_conf = np.mean(stats['confidences'])
                waste_type = stats['waste_type']
                recyclable = "[R]" if stats['recyclable'] else "[W]"
                
                print(f"   {recyclable} {class_name.upper()} ({waste_type.upper()}):")
                print(f"      Count: {count} objects ({count/total_objects*100:.1f}%)")
                print(f"      Avg Confidence: {avg_conf:.3f} ({avg_conf*100:.1f}%)")
            print()
            
            # Generate visualization
            print(f"🎨 Generating R-CNN visualization with waste classification...")
            viz_results = self.visualize_rcnn_detections(image_path, detections)
            
            # Save comprehensive results
            if save_results:
                image_name = Path(image_path).stem
                results = {
                    'detection_info': {
                        'method': 'R-CNN',
                        'image_path': image_path,
                        'image_name': Path(image_path).name,
                        'analysis_timestamp': datetime.now().isoformat(),
                        'total_region_proposals': self.region_proposals_count
                    },
                    'detection_summary': {
                        'total_objects': total_objects,
                        'average_confidence': float(avg_confidence),
                        'max_confidence': float(max_confidence),
                        'min_confidence': float(min_confidence),
                        'unique_classes': len(class_stats),
                        'confidence_std': float(np.std(confidences))
                    },
                    'waste_classification': {
                        'dry_waste_count': dry_waste_count,
                        'wet_waste_count': wet_waste_count,
                        'recyclable_count': recyclable_count,
                        'dry_percentage': float(dry_percentage),
                        'wet_percentage': float(wet_percentage),
                        'recyclable_percentage': float(recyclable_percentage)
                    },
                    'class_statistics': {
                        class_name: {
                            'count': stats['count'],
                            'percentage': float(stats['count'] / total_objects * 100),
                            'average_confidence': float(np.mean(stats['confidences'])),
                            'confidence_std': float(np.std(stats['confidences'])),
                            'waste_type': stats['waste_type'],
                            'is_recyclable': stats['recyclable'],
                            'waste_category': self.waste_classification[class_name]['category']
                        }
                        for class_name, stats in class_stats.items()
                    },
                    'individual_detections': [
                        {
                            'object_id': detection['object_id'],
                            'class': detection['class'],
                            'confidence': float(detection['confidence']),
                            'waste_type': detection['waste_type'],
                            'is_recyclable': detection['is_recyclable'],
                            'waste_category': detection['waste_category'],
                            'bounding_box': detection['bbox'],
                            'detection_method': detection['detection_method']
                        }
                        for detection in detections
                    ],
                    'model_info': {
                        'type': 'R-CNN with Your Trained CNN' if self.use_trained_cnn else 'R-CNN with Mock Detection',
                        'cnn_model_used': self.cnn_model_path if self.use_trained_cnn else 'None',
                        'cnn_input_shape': self.cnn_input_shape,
                        'classes': self.classes,
                        'confidence_threshold': self.confidence_threshold,
                        'nms_threshold': self.nms_threshold,
                        'region_proposals_count': self.region_proposals_count
                    },
                    'visualization_file': viz_results['overview_file']
                }
                
                results_path = f"rcnn_detection_results_{image_name}.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"✅ R-CNN results saved to: {results_path}")
            
            print(f"\n🎉 R-CNN analysis complete!")
            print(f"📁 Generated: {viz_results['overview_file']}")
            print(f"🔍 Detected {total_objects} objects with {avg_confidence*100:.1f}% average confidence")
            print(f"♻️ Classification: {dry_waste_count} DRY + {wet_waste_count} WET ({recyclable_count} recyclable)")
            
            if self.use_trained_cnn:
                print(f"� Detection Method: R-CNN + Your Trained CNN ({self.cnn_model_path})")
            else:
                print(f"🎭 Detection Method: R-CNN with Mock Detection")
            
            return detections
            
        except Exception as e:
            print(f"❌ R-CNN analysis failed: {e}")
            return None
    
    def batch_detect_folder_rcnn(self, folder_path, save_results=True):
        """Batch R-CNN detection on folder of images."""
        print("\n" + "="*60)
        print("BATCH R-CNN OBJECT DETECTION")
        print("="*60)
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.glob(ext))
            image_files.extend(folder.glob(ext.upper()))
        
        if not image_files:
            print(f"❌ No image files found in: {folder_path}")
            return
        
        print(f"Found {len(image_files)} images for R-CNN processing...")
        
        all_results = []
        total_objects = 0
        total_dry_waste = 0
        total_wet_waste = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] R-CNN Processing: {image_file.name}")
            
            try:
                detections = self.analyze_image_with_rcnn(str(image_file), save_results=False)
                
                if detections:
                    dry_count = sum(1 for d in detections if d['waste_type'] == 'dry')
                    wet_count = sum(1 for d in detections if d['waste_type'] == 'wet')
                    
                    result = {
                        'image': str(image_file),
                        'objects_found': len(detections),
                        'dry_waste': dry_count,
                        'wet_waste': wet_count,
                        'average_confidence': np.mean([d['confidence'] for d in detections]),
                        'detections': detections
                    }
                    all_results.append(result)
                    total_objects += len(detections)
                    total_dry_waste += dry_count
                    total_wet_waste += wet_count
                    
                    print(f"  ✓ Found {len(detections)} objects ({dry_count} DRY, {wet_count} WET)")
                else:
                    print(f"  ❌ No objects detected")
            
            except Exception as e:
                print(f"  ❌ Error processing {image_file.name}: {e}")
        
        # Save comprehensive batch results
        if save_results and all_results:
            batch_results = {
                'batch_info': {
                    'method': 'R-CNN Batch Processing',
                    'folder': str(folder),
                    'processing_timestamp': datetime.now().isoformat(),
                    'total_images_processed': len(image_files),
                    'images_with_detections': len(all_results)
                },
                'overall_statistics': {
                    'total_objects_detected': total_objects,
                    'total_dry_waste': total_dry_waste,
                    'total_wet_waste': total_wet_waste,
                    'dry_percentage': (total_dry_waste / total_objects * 100) if total_objects > 0 else 0,
                    'wet_percentage': (total_wet_waste / total_objects * 100) if total_objects > 0 else 0,
                    'average_objects_per_image': total_objects / len(all_results) if all_results else 0
                },
                'detailed_results': all_results
            }
            
            batch_results_path = folder / 'rcnn_batch_detection_results.json'
            with open(batch_results_path, 'w') as f:
                json.dump(batch_results, f, indent=2)
            
            print(f"\n✅ R-CNN batch results saved to: {batch_results_path}")
            print(f"📊 Summary: {len(all_results)}/{len(image_files)} images had detections")
            print(f"🎯 Total objects detected: {total_objects}")
            print(f"♻️ Waste breakdown: {total_dry_waste} DRY, {total_wet_waste} WET")

def main():
    """Main execution function for R-CNN waste detector."""
    print("\n" + "="*60)
    print("R-CNN WASTE OBJECT DETECTOR")
    print("Enhanced with DRY/WET Classification")
    print("="*60)
    
    detector = RCNNWasteDetector()
    
    print("\nWhat would you like to do?")
    print("1. 🔍 R-CNN detection with trained CNN classifier")
    print("2. 📁 Batch R-CNN detection on image folder")
    print("3. 🔄 Reload CNN model")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        # Single image R-CNN detection
        image_path = input("\nEnter image path for R-CNN analysis: ").strip()
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        if detector.use_trained_cnn:
            print(f"✅ Using your trained CNN: {detector.cnn_model_path}")
        else:
            print("⚠ CNN model not loaded - will use mock detection")
            
        detector.analyze_image_with_rcnn(image_path)
    
    elif choice == "2":
        # Batch folder R-CNN detection
        folder_path = input("\nEnter folder path for batch R-CNN processing: ").strip()
        detector.batch_detect_folder_rcnn(folder_path)
    
    elif choice == "3":
        # Reload CNN model
        print("\n🔄 RELOADING CNN MODEL")
        print("="*60)
        
        # Show available models
        available_models = []
        model_files = ['cnn_waste_model.h5', 'cnn_waste_model_compatible.h5', 'trained_waste_model.h5', 'trained_waste_modelfufu.h5', 'mobilenetv2_waste_model.h5']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                available_models.append(model_file)
        
        if available_models:
            print("Available model files:")
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
            
            choice_model = input(f"\nSelect model to load (1-{len(available_models)}) or press Enter for default: ").strip()
            
            if choice_model and choice_model.isdigit():
                model_idx = int(choice_model) - 1
                if 0 <= model_idx < len(available_models):
                    detector.cnn_model_path = available_models[model_idx]
                    print(f"Selected model: {detector.cnn_model_path}")
        
        detector.use_trained_cnn = detector.load_trained_cnn_model()
        
        if detector.use_trained_cnn:
            print("✅ CNN model successfully reloaded!")
            # Also test the model after reloading
            detector.test_cnn_model()
        else:
            print("❌ Failed to reload CNN model")
    
    elif choice == "4":
        print("Goodbye! Happy waste detection! ♻️")
    else:
        print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()