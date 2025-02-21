import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Unable to access the webcam. Check camera connection.")

print("Press 'q' to quit.")

# Create a window for trackbars
cv2.namedWindow("Processed Image")

# --- Set Default Values for Trackbars ---
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

cv2.createTrackbar("Lower R", "Processed Image", lower_blue[0], 255, lambda x: None)
cv2.createTrackbar("Lower G", "Processed Image", lower_blue[1], 255, lambda x: None)
cv2.createTrackbar("Lower B", "Processed Image", lower_blue[2], 255, lambda x: None)

cv2.createTrackbar("Upper R", "Processed Image", upper_blue[0], 255, lambda x: None)
cv2.createTrackbar("Upper G", "Processed Image", upper_blue[1], 255, lambda x: None)
cv2.createTrackbar("Upper B", "Processed Image", upper_blue[2], 255, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from the webcam.")
        break

    # --- Apply RGB Mask Based on Trackbar Values ---
    lower_r = cv2.getTrackbarPos("Lower R", "Processed Image")
    lower_g = cv2.getTrackbarPos("Lower G", "Processed Image")
    lower_b = cv2.getTrackbarPos("Lower B", "Processed Image")

    upper_r = cv2.getTrackbarPos("Upper R", "Processed Image")
    upper_g = cv2.getTrackbarPos("Upper G", "Processed Image")
    upper_b = cv2.getTrackbarPos("Upper B", "Processed Image")


    lower_bgr = np.array([lower_b, lower_g, lower_r])
    upper_bgr = np.array([upper_b, upper_g, upper_r])

    mask_rgb = cv2.inRange(frame, lower_bgr, upper_bgr)
    masked_rgb_image = cv2.bitwise_and(frame, frame, mask=mask_rgb)

    # --- Landing Pad Detection using HSV ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_hsv = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 5000]

    center_state = None

    if contours:
        for contour in contours:
            epsilon = 0.08 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:  # Check if the polygon has 4 points (potential landing pad)
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

    # Resize the window Webcam
    cv2.namedWindow("Webcam - Detect Landing Pad", 0)
    cv2.resizeWindow("Webcam - Detect Landing Pad", 500,300)
    
    # Display outputs
    cv2.imshow("Webcam - Detect Landing Pad", frame)
    cv2.imshow("Processed Image", masked_rgb_image)
    
    cv2.moveWindow("Webcam - Detect Landing Pad", 0, 50)
    cv2.moveWindow("Processed Image", 500, 50)
    

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
