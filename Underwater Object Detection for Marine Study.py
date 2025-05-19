import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('bheem.mp4')

# Set HSV range based on clicked color
lower_color = np.array([90, 50, 30])
upper_color = np.array([105, 255, 255])

# Morphological kernel
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()

    # Loop video if it ends
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Morphological operations
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # Display results
    cv2.imshow("Underwater View", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(100) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()