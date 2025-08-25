import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model('model/asl_model_2.h5')  # Path to your saved model

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe drawing utils (for drawing the landmarks)
mp_drawing = mp.solutions.drawing_utils

alphabet = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y'
}

intvl = 2
last_prediction_time = time.time()

# Function to preprocess the image
def preprocess_image(img, target_size=(28, 28)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize while maintaining aspect ratio and padding
    h, w = img_gray.shape
    tw, th = target_size
    if w > h:
        new_w, new_h = tw, int(h * tw / w)
    else:
        new_w, new_h = int(w * th / h), th
    resized_img = cv2.resize(img_gray, (new_w, new_h))
    padded_img = np.zeros(target_size, dtype=np.uint8)
    x_offset = (tw - new_w) // 2
    y_offset = (th - new_h) // 2
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    img_normalized = padded_img.astype('float32') / 255.0
    img_reshaped = img_normalized.reshape(1, *target_size, 1)
    return img_reshaped

# Function to predict the sign
def predict_sign(img):
    # Preprocess the image
    preprocessed_image = preprocess_image(img)
    
    # Make prediction using the trained model
    prediction = model.predict(preprocessed_image)
    
    # Get the predicted class label (the index of the highest probability)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return predicted_class

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Flip the frame horizontally (optional, for better visual display)
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hands
    results = hands.process(rgb_frame)

    # If hands are detected, process the landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks
            landmark_points = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in hand_landmarks.landmark])

            # Calculate a more stable ROI (example: based on hand centroid and spread)
            center_x, center_y = np.mean(landmark_points, axis=0)
            hand_width = np.max(landmark_points[:, 0]) - np.min(landmark_points[:, 0])
            hand_height = np.max(landmark_points[:, 1]) - np.min(landmark_points[:, 1])
            roi_size = int(max(hand_width, hand_height) * 1.5) # Adjust multiplier as needed
            x1 = int(center_x - roi_size / 2)
            y1 = int(center_y - roi_size / 2)
            x2 = int(center_x + roi_size / 2)
            y2 = int(center_y + roi_size / 2)

            # Ensure ROI is within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            hand_roi = frame[y1:y2, x1:x2]

            if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0 and (time.time()-last_prediction_time) > intvl:
                processed_roi = preprocess_image(hand_roi)
                prediction = model.predict(processed_roi)
                predicted_class = np.argmax(prediction, axis=1)[0]

                cv2.putText(frame, f'Predicted Sign: {alphabet[predicted_class]}',
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with hand landmarks and prediction
    cv2.imshow('Real-Time ASL Prediction', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
