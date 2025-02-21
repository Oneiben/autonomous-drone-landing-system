import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract

class image_processing:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO("./models/landing_pad.pt")

    def cv_detection(self, frame):
        """
        Detect an "H" marker in the given image using color segmentation and contour analysis.

        Args:
            frame (numpy.ndarray): Input image.

        Returns:
            tuple: 
                - image_center (list): Coordinates [x, y] of the image center.
                - center_state (list or None): Coordinates [x, y] of the "H" marker's center or None if not detected.
                - frame (numpy.ndarray): Annotated image with the "H" marker highlighted.
        """
        # Image center
        height, width = frame.shape[:2]
        image_center = [width // 2, height // 2]

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 5000]
        
        # Draw the image center
        cv2.circle(frame, (image_center[0], image_center[1]), 10, (0, 255, 255), -1)
        cv2.putText(frame, "Image Center", (image_center[0] + 15, image_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Process contours if available
        if contours:
            for contour in contours:
                epsilon = 0.08 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # Check if the polygon has 4 points (potential landing pad)
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

                    # Calculate the center of the rectangle
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        x_center = int(M["m10"] / M["m00"])
                        y_center = int(M["m01"] / M["m00"])
                        center_state = [x_center, y_center]
                        
                        # Draw the rectangle center
                        cv2.circle(frame, (x_center, y_center), 10, (0, 0, 255), -1)
                        cv2.putText(frame, "Center", (x_center - 20, y_center - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        return image_center, center_state, frame

        # If no "Landing pad" is detected, return None
        return image_center, None, frame

    def Yolo_detection(self, frame):
        """
        Perform object detection using YOLO, extract bounding box corners, and annotate the frame.

        Args:
            frame (numpy.ndarray): Input image.

        Returns:
            tuple:
                - image_center (list): [x, y] center of the image.
                - center_state (list or None): [x, y] center of the detected object (or None if no detection).
                - frame (numpy.ndarray): Annotated frame with bounding boxes and centers.
        """
        # Get image dimensions
        height, width = frame.shape[:2]
        image_center = [width // 2, height // 2]
        
        # Draw the image center
        cv2.circle(frame, (image_center[0], image_center[1]), 10, (0, 255, 255), -1)
        cv2.putText(frame, "Image Center", (image_center[0] + 15, image_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Run YOLO inference
        results = self.model(frame)

        center_state = None  # Default: No detection

        # Check if any results are found
        if results:
            for result in results:
                for box in result.boxes.data:
                    x_center, y_center, width, height = box[:4]  # Extract YOLO bbox values
                    confidence = float(box[4])  # Confidence score

                    if confidence > 0.5:  # Only consider detections with confidence greater than 0.5
                        center_state = [x_center, y_center]
                        annotated_frame = results[0].plot()
                        return image_center, center_state, annotated_frame

        return image_center, None, frame

    def tes_detection(self, frame):
        """
        Detect an "H" marker in the given image using Tesseract OCR.

        Args:
            frame (numpy.ndarray): Input image.

        Returns:
            tuple:
                - image_center (list): Coordinates [x, y] of the image center.
                - center_state (list or None): Coordinates [x, y] of the "H" marker's center or None if not detected.
                - frame (numpy.ndarray): Annotated frame with the detection result.
        """
        # Get image dimensions
        height, width = frame.shape[:2]
        image_center = [width // 2, height // 2]

        # Draw the image center
        cv2.circle(frame, (image_center[0], image_center[1]), 10, (0, 255, 255), -1)
        cv2.putText(frame, "Image Center", (image_center[0] + 15, image_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to preprocess for OCR
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        try:
            # Run OCR on the processed image
            extracted_data = pytesseract.image_to_boxes(thresh, config="--psm 6")
        except pytesseract.TesseractError as e:
            print(f"Tesseract OCR error: {e}")
            return image_center, None, frame

        center_state = None  # Default: No detection

        # Parse OCR results for the character 'H'
        for line in extracted_data.splitlines():
            char, x_min, y_min, x_max, y_max, _ = line.split()
            if char == "H":
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                # Calculate the center of the detected "H"
                x_center = (x_min + x_max) // 2
                y_center = -(y_min + y_max) // 2
                center_state = [x_center, y_center]

                # Draw bounding box and center
                cv2.rectangle(frame, (x_min, frame.shape[0] - y_min), (x_max, frame.shape[0] - y_max), (0, 255, 0), 2)
                cv2.circle(frame, (x_center, frame.shape[0] - y_center), 10, (0, 0, 255), -1)
                cv2.putText(frame, "'H' Detected", (x_center - 20, frame.shape[0] - y_center - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Stop after finding the first 'H'
                break

        return image_center, center_state, frame
