from flask import Flask, render_template, Response
import cv2
import pickle
import mediapipe as mp
import numpy as np
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

model_dict = pickle.load(open('./modelsfinal.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
con = []
count = 0
labels_dict = {
    0: 'Hi', 1: 'Hello', 2: 'Good Morning', 3: 'Good Afternoon', 4: 'How are you?', 5: 'I am Good',
    6: 'I cannot hear you', 7: 'Can you repeat?', 8: 'Yes', 9: 'No', 10: 'A', 11: 'Hello', 12: 'C',
    13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N',
    24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z', 36: 'Backspace', 37: 'Space', 38: 'Done'
}

cap = None


def gen_frames():
    global cap
    previous_character_added = ''
    previous_letter = ''
    while True:
        if cap is None:
            cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        if not ret:
            continue

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # truncate values for second hand
            data_aux = data_aux[:42]

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]
            if predicted_character == previous_letter:
                count += 1
            else:
                previous_letter = predicted_character
                count = 0
            if count == 15:
                if predicted_character == 'Backspace' and con:
                    con.pop()
                elif predicted_character == 'Space':
                    con.append(' ')
                elif predicted_character == 'Done':
                    print('Predicted Text:', ''.join(con))
                    con.clear()
                else:
                    con.append(predicted_character)

                previous_character_added = predicted_character

                socketio.emit('character_info', {'predicted_character': predicted_character, 'previous_letter': previous_letter}, namespace='/test')
                print(f"Predicted Character: {predicted_character}, Previous Letter: {previous_letter}")

            cv2.putText(frame, 'Previous character Detected: ' + previous_character_added, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 6)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpeg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect', namespace='/test')
def test_connect():
    print('Client connected')


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
