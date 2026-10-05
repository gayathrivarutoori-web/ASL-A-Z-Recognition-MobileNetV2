import cv2
import tensorflow as tf
import numpy as np
from collections import deque, Counter

print("============================================")
print("REAL-TIME ASL - MOBILENETV2")
print("============================================")

# --------------------------------------------
# LOAD MODEL
# --------------------------------------------

print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model("asl_mobilenetv2.keras")

print("Model loaded successfully!")

# --------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------

with open("class_names_mobilenetv2.txt", "r") as file:
    class_names = [line.strip() for line in file]

print("\nClasses loaded:")
print(class_names)

# --------------------------------------------
# CAMERA
# --------------------------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("\nERROR: Camera could not be opened.")
    exit()

print("\nCamera started!")
print("Place your hand inside the blue box.")
print("Press Q to quit.")

# --------------------------------------------
# PREDICTION SETTINGS
# --------------------------------------------

prediction_history = deque(maxlen=15)

CONFIDENCE_THRESHOLD = 70

stable_letter = "Waiting..."

# --------------------------------------------
# REAL-TIME LOOP
# --------------------------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read camera.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # ----------------------------------------
    # CROP SAME AREA USED DURING PREPROCESSING
    # ----------------------------------------

    cropped = frame[105:395, 105:495]

    # Resize
    resized = cv2.resize(cropped, (128, 128))

    # Convert BGR → RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Convert to float32
    resized = resized.astype(np.float32)

    # Add batch dimension
    resized = np.expand_dims(resized, axis=0)

    # ----------------------------------------
    # MODEL PREDICTION
    # ----------------------------------------

    predictions = model.predict(resized, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_letter = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    # ----------------------------------------
    # CONFIDENCE FILTER
    # ----------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        prediction_history.append(predicted_letter)

    # ----------------------------------------
    # MAJORITY VOTING
    # ----------------------------------------

    if len(prediction_history) >= 5:

        counter = Counter(prediction_history)

        most_common_letter, count = counter.most_common(1)[0]

        # Require the same letter several times
        if count >= 8:

            stable_letter = most_common_letter

    # ----------------------------------------
    # DRAW BLUE BOX
    # ----------------------------------------

    cv2.rectangle(
        frame,
        (105, 105),
        (495, 395),
        (255, 0, 0),
        2
    )

    # ----------------------------------------
    # DISPLAY STABLE LETTER
    # ----------------------------------------

    cv2.putText(
        frame,
        f"Letter: {stable_letter}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # ----------------------------------------
    # DISPLAY CURRENT CONFIDENCE
    # ----------------------------------------

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ----------------------------------------
    # DISPLAY CURRENT MODEL PREDICTION
    # ----------------------------------------

    cv2.putText(
        frame,
        f"Current: {predicted_letter}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ----------------------------------------
    # DISPLAY TOP 3 PREDICTIONS
    # ----------------------------------------

    top_indices = np.argsort(predictions[0])[-3:][::-1]

    y_position = 430

    for rank, index in enumerate(top_indices):

        letter = class_names[index]

        probability = predictions[0][index] * 100

        text = f"{rank + 1}. {letter}: {probability:.2f}%"

        cv2.putText(
            frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        y_position += 30

    # ----------------------------------------
    # SHOW WINDOW
    # ----------------------------------------

    cv2.imshow(
        "Real-Time ASL Recognition - MobileNetV2",
        frame
    )

    # ----------------------------------------
    # QUIT
    # ----------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

# --------------------------------------------
# CLOSE CAMERA
# --------------------------------------------

camera.release()
cv2.destroyAllWindows()

print("\n============================================")
print("REAL-TIME RECOGNITION STOPPED")
print("============================================")