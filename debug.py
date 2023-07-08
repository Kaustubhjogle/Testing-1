import pickle
import cv2
import mediapipe as mp
import numpy as np
import cv2
import mediapipe as mp
import numpy as np
import pickle



# Load the pre-trained model
model_dict = pickle.load(open('modelsfinal.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hands module
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'Hi', 1: 'Hello', 2: 'Good Morning', 3: 'Good Afternoon', 4: 'How are you?', 5: 'I am Good', 6: 'I cannot hear you',
               7: 'Can you repeat?', 8: 'Yes', 9: 'No', 10: 'A', 11: 'Hello', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
               20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
               35: 'Z', 36: 'Backspace', 37: 'Space', 38: 'Done'}

# Video capture object
cap = cv2.VideoCapture(0)
# Rest of the code...



# Debugging code
while True:
    # Rest of the code...
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Extract hand landmarks and normalize
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Make prediction
        x1 = int(min(x_) * frame.shape[1]) - 10
        y1 = int(min(y_) * frame.shape[0]) - 10
        x2 = int(max(x_) * frame.shape[1]) - 10
        y2 = int(max(y_) * frame.shape[0]) - 10

        data_aux = data_aux[:42]  # truncate values for second hand
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw rectangle and display predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        # Print the input data
        print('Input Data:', data_aux)

        # Make predictions
        prediction = model.predict([np.asarray(data_aux)])

        # Print the raw predicted values
        print('Raw Predictions:', prediction)

        # Convert the predicted value to label
        predicted_character = labels_dict[int(prediction[0])]
        print('Predicted Label:', predicted_character)

    # Rest of the code...
