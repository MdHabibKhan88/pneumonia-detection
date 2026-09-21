# ============================================
# Pneumonia Detection - Dataset Loading
# ============================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset paths
TRAIN_DIR = r"D:\KYAU\Semester 3.2\Project Devolopment\Poject\chest_xray\train"
VAL_DIR = r"D:\KYAU\Semester 3.2\Project Devolopment\Poject\chest_xray\val"
TEST_DIR = r"D:\KYAU\Semester 3.2\Project Devolopment\Poject\chest_xray\test"

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Training image generator
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True
)

# Validation/Test generator
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# Load training images
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load validation images
val_data = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load test images
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nDataset loading completed!")
print("Class names:", train_data.class_indices)


# ============================================
# CNN Model - Pneumonia Detection
# ============================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    # Convolution Layer 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Convolution Layer 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Convolution Layer 3
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Output: 0 = NORMAL, 1 = PNEUMONIA
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Show model structure
model.summary()

# ============================================
# Train the CNN Model
# ============================================

# ============================================
# Train the CNN Model with Class Weights
# ============================================

EPOCHS = 10

class_weight = {
    0: 2.0,  # NORMAL
    1: 1.0   # PNEUMONIA
}

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight
)
# ============================================
# Save the Trained Model
# ============================================

model.save("pneumonia_model.keras")

print("\nModel saved successfully!")

# ============================================
# Test the Trained Model
# ============================================

test_loss, test_accuracy = model.evaluate(test_data)

print("\nTest Results:")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ============================================
# Confusion Matrix - Model Evaluation
# ============================================

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Predict test images
predictions = model.predict(test_data)

# Convert probabilities to class labels
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Actual labels
true_classes = test_data.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=["NORMAL", "PNEUMONIA"]
    )
)
