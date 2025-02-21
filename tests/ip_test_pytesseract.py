import cv2
import pytesseract

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Unable to access the webcam. Check camera connection.")

print("Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # Get the image dimensions
        height, width = frame.shape[:2]
        image_center = (width // 2, height // 2)

        # Convert to grayscale for OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to enhance text detection
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Run OCR on the processed image
        extracted_data = pytesseract.image_to_boxes(thresh, config="--psm 6")

        detected_H = []  # Store all 'H' detections

        # Parse OCR results for character 'H'
        for line in extracted_data.splitlines():
            char, x_min, y_min, x_max, y_max = line.split()[:5]
            if char == "H":
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                # Convert OCR coordinates to match OpenCV's coordinate system
                y_min, y_max = height - y_min, height - y_max
                x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2

                detected_H.append((x_center, y_center, x_min, y_min, x_max, y_max))

                # Draw bounding box and center for each detected 'H'
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(frame, (x_center, y_center), 10, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    "'H' Detected",
                    (x_center - 20, y_center - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Print detected 'H' coordinates (useful for debugging)
        if detected_H:
            print(f"Detected 'H' at: {detected_H}")

        # Display the processed frame
        cv2.imshow("Webcam - Detect 'H'", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("\nKeyboard Interrupt detected. Exiting...")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam and OpenCV windows closed.")
