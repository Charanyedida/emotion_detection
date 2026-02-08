import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# --- 1. GPU CONFIGURATION ---
# This prevents "Physical devices cannot be modified" errors if run early
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)

# --- 2. DATA LOADERS ---
IMG_SIZE = 48
BATCH_SIZE = 64
DATA_PATH = '/kaggle/input/fer2013/train'  # Double check this path!

# Agumentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

print("⏳ Loading Data...")
# Load Train
train_gen = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    subset='training',
    shuffle=True
)

# Load Val
val_gen = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation',
    shuffle=False
)

# --- AUTO-DETECT CLASSES (Crucial Fix) ---
num_classes = len(train_gen.class_indices)
print(f"✅ Detected {num_classes} classes: {list(train_gen.class_indices.keys())}")

# --- 3. MODEL BUILDING (DenseNet121) ---
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Fine-Tuning Strategy: Unfreeze last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
# Dynamic output layer based on detected classes
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 4. CALLBACKS ---
checkpoint = ModelCheckpoint(
    'best_emotion_model.keras',  # .keras is the new standard format
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint, reduce_lr, early_stop]

# --- 5. TRAINING (Keras 3 Compatible) ---
print("🚀 Starting Training...")
history = model.fit(
    train_gen,
    epochs=40,
    validation_data=val_gen,
    callbacks=callbacks
)

print("🎉 Training Complete. Model saved as 'best_emotion_model.keras'")