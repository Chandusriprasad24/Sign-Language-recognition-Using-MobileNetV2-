

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os

app = Flask(__name__)

# Load model and class names
model = load_model('model/mobilenetv2_final_model.keras')
classes = sorted(os.listdir('C:/Users/Chandu Sri Prasad/OneDrive/Desktop/Major_Project  YoLo/New Model/split/train'))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
word = ""
current_letter = ""  # <- store latest prediction

def gen_frames():
    global word, current_letter
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        letter = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x, x_min), min(y, y_min)
                    x_max, y_max = max(x, x_max), max(y, y_max)

                margin = 30
                x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
                x_max, y_max = min(x_max + margin, w), min(y_max + margin, h)

                hand_roi = frame[y_min:y_max, x_min:x_max]
                hand_roi = cv2.GaussianBlur(hand_roi, (5, 5), 0)
                hand_roi = cv2.resize(cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB), (128, 128)) / 255.0
                hand_roi = np.expand_dims(hand_roi, axis=0)

                preds = model.predict(hand_roi, verbose=0)
                idx = np.argmax(preds)
                letter = classes[idx]
                current_letter = letter  # update the latest predicted letter

                cv2.putText(frame, f"{letter}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the word
        cv2.putText(frame, f"Word: {word}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_word')
def get_word():
    return jsonify({'word': word})

@app.route('/reset_word')
def reset_word():
    global word
    word = ""
    return jsonify({'status': 'reset'})

@app.route('/append_letter')
def append_letter():
    global word, current_letter
    if current_letter:
        if current_letter.lower() == 'space':
            word += ' '
        elif current_letter.lower() == 'del':
            word = word[:-1]
        else:
            word += current_letter
    return jsonify({'word': word})


if __name__ == '__main__':
    app.run(debug=True)


    
#http://127.0.0.1:5000

