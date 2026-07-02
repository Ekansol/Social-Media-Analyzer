# ============================================================
# PART 4 — DEEPFAKE FORENSICS (ResNet50 Binary Classifier)
# ============================================================
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

print("=" * 55)
print("PART 4: Deepfake Forensics (ResNet50)")
print("=" * 55)

# ── STEP 1: Handle the Nested Folders ────────────────────────
dataset_path = 'vision_dataset'

# ResNet50 has a specific preprocessing requirement for pixels
data_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2 
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("Loading and mapping nested folders...")
# Keras will automatically see 'ai_generated' and 'real' as the two classes,
# and will recursively pull the images from the animal/city/food subfolders!
train_generator = data_gen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # 0 = AI Generated, 1 = Real
    subset='training'
)

val_generator = data_gen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ── STEP 2: Download ResNet50 Base ───────────────────────────
print("\nDownloading ResNet50 Feature Extractor...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

# ── STEP 3: Attach the Binary Detective Head ─────────────────
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

# 1 Neuron with Sigmoid (Result < 0.5 = AI, Result > 0.5 = Real)
predictions = Dense(1, activation='sigmoid')(x)

vision_model = Model(inputs=base_model.input, outputs=predictions)

vision_model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# ── STEP 4: Train the Detective ──────────────────────────────
print("\nTraining the Forensics Classifier...")
vision_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,  # Transfer learning is fast; 5 epochs is enough
    verbose=1
)

# ── STEP 5: Save Model ───────────────────────────────────────
vision_model.save('forensics_resnet50.h5')
print("\n✅ Deepfake Detective Saved: forensics_resnet50.h5")