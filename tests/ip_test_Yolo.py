from pathlib import Path

import cv2
from ultralytics import YOLO

# Load YOLO model
model_path = Path("./models/landing_pad.pt")  # for landing_pad image
# model_path = Path("./models/LandingPad.pt") # for LandingPad image


# Check if model file exists before loading
if not model_path.exists():
    raise FileNotFoundError(
        f"Model file '{model_path}' not found! Ensure the path is correct."
    )

model = YOLO(model_path)
# Open webcam
cap = cv2.VideoCapture(0)

# Optional: Set webcam resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam. Check camera connection.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Warning: Failed to capture frame. Retrying...")
            continue  # Skip iteration if frame capture fails

        # Run YOLO detection
        results = model(frame)

        # Annotate frame with detections
        annotated_frame = results[0].plot()

        # Display the processed frame
        cv2.imshow("YOLO Webcam", annotated_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting YOLO Webcam...")
            break

except KeyboardInterrupt:
    print("\nKeyboard Interrupt detected. Exiting...")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam and OpenCV windows closed.")
