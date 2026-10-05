import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("============================================")
print("ASL A-Z CONFUSION MATRIX")
print("============================================")

# Settings
DATASET_PATH = "processed_dataset"
IMG_SIZE = 128
BATCH_SIZE = 32
MODEL_PATH = "asl_mobilenetv2.keras"

# Load the trained model
print("\nLoading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load validation dataset
print("\nLoading validation dataset...")

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

# Get actual labels and predictions
true_labels = []
predicted_labels = []

print("\nGenerating predictions...")

for images, labels in validation_dataset:

    predictions = model.predict(images, verbose=0)

    predicted = np.argmax(predictions, axis=1)

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Calculate confusion matrix
cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=range(len(class_names))
)

# Display confusion matrix
print("\n============================================")
print("CONFUSION MATRIX")
print("============================================")

print(cm)

# Calculate accuracy
accuracy = np.trace(cm) / np.sum(cm)

print("\nValidation Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Plot confusion matrix
plt.figure(figsize=(14, 14))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45
)

plt.title("ASL A-Z Confusion Matrix")
plt.tight_layout()

# Save image
plt.savefig("asl_confusion_matrix.png", dpi=300)

print("\nConfusion matrix saved as:")
print("asl_confusion_matrix.png")

plt.show()