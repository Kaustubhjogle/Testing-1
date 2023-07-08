from flask import Flask, render_template, Response
import cv2

import pickle
# import cv2
import mediapipe as mp
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
databaseURL = {
     'databaseURL': "https://text-7aaad-default-rtdb.asia-southeast1.firebasedatabase.app"
}
# Use a service account.
cred = credentials.Certificate(r"C:\Users\daksh\Downloads\text-7aaad-firebase-adminsdk-lwdcn-2882b41cac.json")

app = firebase_admin.initialize_app(cred, databaseURL)

db = firestore.client()

app = Flask(__name__)

model_dict = pickle.load(open('./modelsfinal.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
con=[]
count=0
labels_dict = {0: 'Hi', 1: 'Hello', 2: 'Good Morning', 3: 'Good Afternoon', 4: 'How are you?', 5: 'I am Good', 6: 'I cannot hear you',
               7: 'Can you repeat?', 8: 'Yes', 9: 'No', 10: 'A', 11: 'Hello', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
               20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
               35: 'Z', 36: 'Backspace', 37: 'Space', 38: 'Done'}
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    previous_character_added = ''
    previous_letter = ''
    temp_text = ''
    while True:

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

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

            # trucate values for second hand
            data_aux = data_aux[:42]
            # print('check',[np.asarray(data_aux)])
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
                    temp_text = db.collection(u'text').document(u'final_predicted_text')
                    doc = temp_text.get()
                    text = str(doc.to_dict()['innertext'])
                    data = {
                        u'innertext': text
                    }
                    db.collection(u'text').document(u'2USg7lFk04dOe2JEQOEG').set(data, merge=True)
                    con.clear()
                else:
                    con.append(predicted_character)
                data = {
                    u'innertext': array_to_string(con)
                }
                db.collection(u'text').document(u'predicted_text').set(data, merge=True)
                previous_character_added = predicted_character

            cv2.putText(frame, 'Previos charatcter Detected: '+previous_character_added, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 6)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def array_to_string(array):
    text_temp = ''
    for i in array:
        text_temp = text_temp+i
    return text_temp


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)