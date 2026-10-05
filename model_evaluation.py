import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Settings
# -----------------------------
DATASET_PATH = "processed_dataset"
MODEL_PATH = "asl_mobilenetv2.keras"

IMG_SIZE = 128
BATCH_SIZE = 32

# -----------------------------
# Load the trained model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# -----------------------------
# Load validation dataset
# -----------------------------
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = validation_dataset.class_names

print("\nClasses:")
print(class_names)

# -----------------------------
# Get actual and predicted labels
# -----------------------------
y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# Calculate evaluation metrics
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)

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

# -----------------------------
# Display results
# -----------------------------
print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

# -----------------------------
# Classification report
# -----------------------------
print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("ASL A-Z Confusion Matrix")

plt.tight_layout()

plt.savefig("asl_confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved as:")
print("asl_confusion_matrix.png")