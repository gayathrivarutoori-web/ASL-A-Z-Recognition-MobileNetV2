import cv2
import os

# Input and output folders
input_folder = "."
output_folder = "processed_dataset"

# Image size required by the model
IMG_SIZE = 128

# ASL letters
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

total_images = 0

for letter in letters:

    input_path = os.path.join(input_folder, letter)
    output_path = os.path.join(output_folder, letter)

    print(f"\nProcessing {letter}...")

    if not os.path.exists(input_path):
        print(f"Folder not found: {input_path}")
        continue

    os.makedirs(output_path, exist_ok=True)

    count = 0

    for filename in os.listdir(input_path):

        file_path = os.path.join(input_path, filename)

        # Only process image files
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image = cv2.imread(file_path)

        if image is None:
            print(f"Could not read: {filename}")
            continue

        # Crop the hand region
        cropped = image[105:395, 105:495]

        # Resize to 128 x 128
        resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

        # Save processed image
        output_file = os.path.join(output_path, filename)
        cv2.imwrite(output_file, resized)

        count += 1
        total_images += 1

    print(f"Processed {count} images for {letter}")

print("\n================================")
print("PREPROCESSING COMPLETED!")
print("================================")
print(f"Total images processed: {total_images}")
print(f"Images saved in: {output_folder}")