import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

print("============================================")
print("ASL A-Z MOBILENETV2 TRAINING")
print("============================================")

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = "processed_dataset"
IMG_SIZE = 128
BATCH_SIZE = 32

# CHANGED: 20 + 20 = up to 40 epochs
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 20

MODEL_NAME = "asl_mobilenetv2.keras"


# ============================================
# CHECK DATASET
# ============================================

if not os.path.exists(DATASET_PATH):
    print("ERROR: processed_dataset folder not found!")
    exit()

print("\nDataset found:")
print(DATASET_PATH)


# ============================================
# LOAD TRAINING DATASET
# ============================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)


# ============================================
# LOAD VALIDATION DATASET
# ============================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)


# ============================================
# CLASS NAMES
# ============================================

class_names = train_dataset.class_names

print("\n============================================")
print("CLASSES")
print("============================================")

print(class_names)
print("Number of classes:", len(class_names))

if len(class_names) != 26:
    print("\nWARNING: Expected 26 classes!")
    print("Please check your A-Z folders.")


# ============================================
# PERFORMANCE SETTINGS
# ============================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)


# ============================================
# DATA AUGMENTATION
# ============================================

data_augmentation = tf.keras.Sequential([

    layers.RandomRotation(0.08),

    layers.RandomZoom(0.10),

    layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    ),

    layers.RandomContrast(0.15),

    layers.RandomBrightness(0.15)

])

# No horizontal flipping because
# hand orientation matters for ASL.


# ============================================
# LOAD MOBILENETV2
# ============================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze MobileNetV2 initially
base_model.trainable = False

print("MobileNetV2 loaded successfully!")


# ============================================
# BUILD MODEL
# ============================================

inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Data augmentation
x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(x)

# MobileNetV2 feature extraction
x = base_model(
    x,
    training=False
)

# Global Average Pooling
x = layers.GlobalAveragePooling2D()(x)

# Dense layer
x = layers.Dense(
    256,
    activation="relu"
)(x)

# Dropout
x = layers.Dropout(0.4)(x)

# Output layer - 26 ASL letters
outputs = layers.Dense(
    26,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ============================================
# COMPILE MODEL
# ============================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================
# MODEL SUMMARY
# ============================================

print("\n============================================")
print("MODEL SUMMARY")
print("============================================")

model.summary()


# ============================================
# CALLBACKS
# ============================================

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=4,

    restore_best_weights=True
)


checkpoint = ModelCheckpoint(

    MODEL_NAME,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ============================================
# PHASE 1
# INITIAL TRAINING
# ============================================

print("\n============================================")
print("PHASE 1: INITIAL TRAINING")
print("============================================")

print("Maximum epochs:", INITIAL_EPOCHS)

history1 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=INITIAL_EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================
# PHASE 2
# FINE-TUNING
# ============================================

print("\n============================================")
print("PHASE 2: FINE-TUNING")
print("============================================")

print("Maximum epochs:", FINE_TUNE_EPOCHS)

# Unfreeze MobileNetV2
base_model.trainable = True


# Freeze earlier layers
for layer in base_model.layers[:-30]:

    layer.trainable = False


# Recompile with lower learning rate
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


history2 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================
# FINAL MODEL EVALUATION
# ============================================

print("\n============================================")
print("FINAL MODEL EVALUATION")
print("============================================")

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)


# ============================================
# PRINT RESULTS
# ============================================

print("\nValidation Accuracy:")

print(
    f"{accuracy * 100:.2f}%"
)

print("\nValidation Loss:")

print(
    f"{loss:.4f}"
)


# ============================================
# SAVE MODEL
# ============================================

model.save(MODEL_NAME)

print("\n============================================")
print("MODEL SAVED SUCCESSFULLY")
print("============================================")

print(
    f"Model: {MODEL_NAME}"
)


# ============================================
# SAVE CLASS NAMES
# ============================================

with open(
    "class_names_mobilenetv2.txt",
    "w"
) as file:

    for name in class_names:

        file.write(
            name + "\n"
        )


print("Classes saved:")

print(
    "class_names_mobilenetv2.txt"
)


# ============================================
# TRAINING COMPLETED
# ============================================

print("\n============================================")
print("TRAINING COMPLETED!")
print("============================================")

print(
    "Initial training: up to 20 epochs"
)

print(
    "Fine-tuning: up to 20 epochs"
)

print(
    "Maximum total: up to 40 epochs"
)

print("============================================")