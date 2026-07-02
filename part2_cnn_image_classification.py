# ============================================================
# PART 2 — IMAGE CLASSIFICATION USING CNN + TRANSFER LEARNING
# Covers: Unit III (CNN), Lab 1, Lab 2, Lab 10
# ============================================================

# ── STEP 1: Import Libraries ─────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                      Dropout, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# ── STEP 2: Dataset Setup ────────────────────────────────────
# Expected folder structure (create these folders & add images):
#   dataset/
#     train/
#       meme/         ← put meme images here
#       advertisement/ ← put ad images here
#       news/         ← put news images here
#       personal/     ← put personal photos here
#     val/
#       meme/
#       advertisement/
#       news/
#       personal/
#
# For testing without real images, we generate random dummy data:

IMG_SIZE    = 224    # ResNet50 expects 224×224
BATCH_SIZE  = 16
NUM_CLASSES = 4      # meme, advertisement, news, personal
EPOCHS      = 15

def create_dummy_dataset(base_path, classes, n_per_class=20):
    """Creates dummy image files for testing when no real dataset exists."""
    from PIL import Image
    import random
    for split in ['train', 'val']:
        for cls in classes:
            folder = os.path.join(base_path, split, cls)
            os.makedirs(folder, exist_ok=True)
            count = n_per_class if split == 'train' else 5
            for i in range(count):
                # Random coloured image simulates different image classes
                arr = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3),
                                        dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(os.path.join(folder, f'{cls}_{i}.jpg'))

DATASET_PATH = 'social_media_images'
CLASSES      = ['meme', 'advertisement', 'news', 'personal']

# Only create dummy data if dataset folder doesn't exist
if not os.path.exists(DATASET_PATH):
    print("Creating dummy dataset for demonstration...")
    try:
        create_dummy_dataset(DATASET_PATH, CLASSES)
        print("Dummy dataset created!")
    except ImportError:
        print("PIL not found. Please install: pip install Pillow")

# ── STEP 3: Data Augmentation (prevents overfitting) ─────────
# Training data: randomly flip, rotate, zoom to create variety
train_datagen = ImageDataGenerator(
    rescale=1./255,           # normalize pixel values 0-1
    rotation_range=20,        # randomly rotate ±20 degrees
    width_shift_range=0.2,    # randomly shift horizontally
    height_shift_range=0.2,   # randomly shift vertically
    horizontal_flip=True,     # randomly flip left-right
    zoom_range=0.2            # randomly zoom in/out
)
# Validation data: only normalize, no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'   # one-hot encoded labels
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
print("\nClass mapping:", train_gen.class_indices)

# ── STEP 4: Load Pre-trained ResNet50 ────────────────────────
# include_top=False → remove ResNet's original 1000-class head
# weights='imagenet' → use weights learned from 1M images
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# FREEZE all ResNet layers — don't change their weights
# They already know edges, textures, shapes from ImageNet
base_model.trainable = False
print(f"\nResNet50 loaded: {len(base_model.layers)} layers (all frozen)")

# ── STEP 5: Add Our Custom Classification Head ───────────────
x = base_model.output
x = GlobalAveragePooling2D()(x)   # flatten feature maps to 1D vector
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)               # prevent overfitting
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)  # 4 classes

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),   # small LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f"Total params   : {model.count_params():,}")
print(f"Trainable params: {sum([np.prod(v.shape) for v in model.trainable_variables]):,}")

# ── STEP 6: Train (Phase 1 — only our custom layers) ─────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

print("\n── Phase 1: Training custom head (ResNet frozen) ──")
history1 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ── STEP 7: Fine-tuning (Phase 2 — unfreeze last few layers) ──
print("\n── Phase 2: Fine-tuning last 20 ResNet layers ──")
# Unfreeze the last 20 layers of ResNet for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Use even smaller learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ── STEP 8: Evaluate & Visualize ─────────────────────────────
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")

# Combine both training histories
all_acc  = history1.history['accuracy']  + history2.history['accuracy']
all_val  = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss = history1.history['loss']      + history2.history['loss']
all_vloss= history1.history['val_loss']  + history2.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(all_acc,  label='Train Acc')
plt.plot(all_val,  label='Val Acc')
plt.axvline(x=len(history1.history['accuracy'])-1,
            color='red', linestyle='--', label='Fine-tune starts')
plt.title('CNN Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(all_loss,  label='Train Loss')
plt.plot(all_vloss, label='Val Loss')
plt.axvline(x=len(history1.history['loss'])-1,
            color='red', linestyle='--', label='Fine-tune starts')
plt.title('CNN Loss'); plt.legend()
plt.tight_layout()
plt.savefig('cnn_training_curves.png')
plt.show()

# ── STEP 9: Predict a Single Image ───────────────────────────
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(image_path):
    """Classify a single social media image."""
    img   = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr   = img_to_array(img) / 255.0
    arr   = np.expand_dims(arr, axis=0)        # add batch dimension
    preds = model.predict(arr, verbose=0)[0]
    idx   = np.argmax(preds)
    label = CLASSES[idx]
    conf  = preds[idx] * 100
    print(f"\nImage     : {image_path}")
    print(f"Category  : {label}  ({conf:.1f}% confidence)")
    for i, (cls, prob) in enumerate(zip(CLASSES, preds)):
        bar = '█' * int(prob * 20)
        print(f"  {cls:<15} {bar} {prob*100:.1f}%")

# ── STEP 10: Visualize Feature Maps (Grad-CAM style) ─────────
def show_feature_maps(image_path, n_filters=8):
    """Show what the first conv layer is detecting."""
    from tensorflow.keras.models import Model as KModel
    # Get output of first conv layer
    first_conv = [l for l in model.layers if 'conv' in l.name][0]
    feat_model = KModel(inputs=model.input, outputs=first_conv.output)

    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    features = feat_model.predict(arr, verbose=0)[0]

    plt.figure(figsize=(12, 4))
    for i in range(min(n_filters, features.shape[-1])):
        plt.subplot(2, 4, i+1)
        plt.imshow(features[:, :, i], cmap='viridis')
        plt.title(f'Filter {i+1}')
        plt.axis('off')
    plt.suptitle('First Conv Layer Feature Maps')
    plt.tight_layout()
    plt.savefig('feature_maps.png')
    plt.show()

# ── STEP 11: Save the Model ───────────────────────────────────
model.save('cnn_social_image_model.h5')
print("\nModel saved as 'cnn_social_image_model.h5'")
