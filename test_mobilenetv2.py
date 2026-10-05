import tensorflow as tf
import numpy as np
import os

print("============================================")
print("ASL MOBILENETV2 MODEL TEST")
print("============================================")

# Load MobileNetV2 model
print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model(
    "asl_mobilenetv2.keras"
)

print("Model loaded successfully!")

# Load class names
with open(
    "class_names_mobilenetv2.txt",
    "r"
) as file:
    class_names = [
        line.strip()
        for line in file
    ]

print("\nClasses:")
print(class_names)

print("\nNumber of classes:", len(class_names))

# Dataset location
dataset_path = "processed_dataset"

# Letters to test
test_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("\n============================================")
print("TESTING A-Z")
print("============================================")

correct = 0
total = 0

for letter in test_letters:

    folder = os.path.join(
        dataset_path,
        letter
    )

    if not os.path.exists(folder):
        print(f"{letter}: folder not found")
        continue

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if len(images) == 0:
        print(f"{letter}: NO IMAGES")
        continue

    # Test first image from each folder
    image_path = os.path.join(
        folder,
        images[0]
    )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(128, 128)
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # Do NOT divide by 255 here.
    # The model already performs MobileNetV2 scaling.

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_letter = class_names[
        predicted_index
    ]

    confidence = (
        prediction[0][predicted_index] * 100
    )

    if predicted_letter == letter:
        correct += 1

    total += 1

    print(
        f"Actual: {letter}  |  "
        f"Predicted: {predicted_letter}  |  "
        f"Confidence: {confidence:.2f}%"
    )

# Final result
accuracy = (
    correct / total * 100
    if total > 0
    else 0
)

print("\n============================================")
print("TEST RESULT")
print("============================================")

print(f"Correct: {correct}/{total}")
print(f"Test Accuracy: {accuracy:.2f}%")

print("\n============================================")
print("TEST COMPLETED!")
print("============================================")