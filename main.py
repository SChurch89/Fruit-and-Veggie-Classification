import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define dataset paths
train_dir = "path_to_dataset/Train"
test_dir = "path_to_dataset/Test"
val_dir = "path_to_dataset/Validation"

# Image dimensions
IMG_SIZE = (224, 224)  # Standard size for pretrained models like VGG16 or ResNet
BATCH_SIZE = 32

# Data Augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    rotation_range=30,  # Rotate images up to 30 degrees
    width_shift_range=0.2,  # Horizontal shifts
    height_shift_range=0.2,  # Vertical shifts
    shear_range=0.2,  # Shear distortions
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Flip images
    fill_mode="nearest"
)

# No augmentation for test & validation sets, just rescaling
test_val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Print class labels
print("Class labels:", train_generator.class_indices)
