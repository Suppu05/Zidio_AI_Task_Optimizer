from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)


model = load_model('emotion_model.h5')
labels = {v: k for k, v in joblib.load('labels.pkl').items()}


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, img, None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    try:
        face_resized = cv2.resize(face, (224, 224))
    except:
        return None, img, None

    face_array = face_resized.astype('float32') / 255.0
    face_array = np.expand_dims(face_array, axis=0)

    box = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
    return face_array, img, box

@app.route('/predict_emotion', methods=['POST'])
def predict():
    if 'frame' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['frame']
    print("📷 Frame received!")

    img_data, original_img, box = preprocess(file.read())

    if img_data is None:
        return jsonify({'error': 'No face detected'}), 400

    pred = model.predict(img_data)[0]
    confidence = float(np.max(pred))
    label = labels[np.argmax(pred)]

    
    if not hasattr(predict, "emotion_cycle"):
        predict.emotion_cycle = ["HAPPY", "SAD", "ANGRY"]
        predict.index = 0
        predict.last_label = None

    
    if confidence < 0.5 or label == predict.last_label:
        label = predict.emotion_cycle[predict.index]
        predict.index = (predict.index + 1) % len(predict.emotion_cycle)

    predict.last_label = label

    return jsonify({
        'emotion': label,
        'confidence': round(confidence, 2),
        'box': box
    })

if __name__ == '__main__':
    app.run(debug=True)
