import cv2
import numpy as np

def get_hand_gesture(cnt, defects):
    if defects is None or len(defects) < 2:
        return "Fist"

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = np.linalg.norm(np.array(start) - np.array(end))
        b = np.linalg.norm(np.array(start) - np.array(far))
        c = np.linalg.norm(np.array(end) - np.array(far))

        # Apply cosine rule
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        if angle <= 90:
            finger_count += 1

    # Simple rules based on number of defects
    if finger_count == 0:
        return "Fist"
    elif finger_count == 1:
        return "Thumbs Up"
    elif finger_count >= 4:
        return "Palm"
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Region of interest
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range (can be adjusted)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gesture = "No Hand"

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 2000:  # avoid noise
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)

            gesture = get_hand_gesture(cnt, defects)
            cv2.drawContours(roi, [cnt], -1, (255, 0, 0), 2)

    # Display the gesture name
    cv2.putText(frame, f"Gesture: {gesture}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

