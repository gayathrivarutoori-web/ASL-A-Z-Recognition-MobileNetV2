import os
import random
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "processed_dataset"
MODEL_PATH = "asl_mobilenetv2.keras"
IMG_SIZE = 128
VALIDATION_PERCENTAGE = 0.20
SEED = 123

random.seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ==============================
# LOAD MODEL
# ==============================

print("============================================")
print("ASL A-Z STRATIFIED MODEL EVALUATION")
print("============================================")

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

print("\nClasses:")
print(CLASS_NAMES)
print("Number of classes:", len(CLASS_NAMES))

# ==============================
# CREATE STRATIFIED VALIDATION SET
# ==============================

print("\nCreating validation set...")

validation_images = []
validation_labels = []

for label, letter in enumerate(CLASS_NAMES):

    folder = os.path.join(DATASET_PATH, letter)

    if not os.path.exists(folder):
        print(f"WARNING: Folder not found: {folder}")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files.sort()

    random.shuffle(files)

    total = len(files)

    validation_count = max(
        1,
        int(total * VALIDATION_PERCENTAGE)
    )

    selected_files = files[:validation_count]

    print(
        f"{letter}: {total} total -> "
        f"{validation_count} validation"
    )

    for filename in selected_files:

        image_path = os.path.join(folder, filename)

        image = cv2.imread(image_path)

        if image is None:
            print("Could not read:", image_path)
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(np.float32)

        validation_images.append(image)
        validation_labels.append(label)

print("\nTotal validation images:", len(validation_images))

# ==============================
# CONVERT TO NUMPY ARRAYS
# ==============================

X_validation = np.array(
    validation_images,
    dtype=np.float32
)

y_true = np.array(
    validation_labels,
    dtype=np.int32
)

print("\nImages loaded successfully!")

# ==============================
# GENERATE PREDICTIONS
# ==============================

print("\nGenerating predictions...")
print("Please wait...")

predictions = model.predict(
    X_validation,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# ==============================
# CALCULATE METRICS
# ==============================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

correct = np.sum(y_true == y_pred)
incorrect = np.sum(y_true != y_pred)

# ==============================
# DISPLAY RESULTS
# ==============================

print("\n============================================")
print("FINAL MODEL EVALUATION")
print("============================================")

print(
    f"Total validation images : {len(y_true)}"
)

print(
    f"Correct predictions     : {correct}"
)

print(
    f"Incorrect predictions   : {incorrect}"
)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

# ==============================
# CONFUSION MATRIX
# ==============================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(26))
)

plt.figure(figsize=(14, 12))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title("ASL A-Z Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(26)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Add numbers inside cells
for i in range(26):
    for j in range(26):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

# ==============================
# SAVE CONFUSION MATRIX
# ==============================

output_file = os.path.join(
    os.getcwd(),
    "confusion_matrix_final.png"
)

try:

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nConfusion matrix saved successfully:"
    )

    print(output_file)

except Exception as e:

    print(
        "\nCould not save confusion matrix."
    )

    print("Reason:", e)

plt.close()

print("\n============================================")
print("EVALUATION COMPLETED")
print("============================================")