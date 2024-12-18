from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import cv2
import numpy as np
import os
import uuid  # For generating unique filenames

app = Flask(__name__)
CORS(app)

# Path configuration
model_path = "faceCount_model/saved_model/model_gender_final.h5"
cascade_path = "faceCount_model/datasets/Input/haarcascade_frontalface_default.xml"
save_path = "faceCount_model/static/processed"
os.makedirs(save_path, exist_ok=True)

# Load model and cascade classifier
model = load_model(model_path)
cascade = cv2.CascadeClassifier(cascade_path)
mtcnn_detector = MTCNN()

# Labels for prediction
LABELS = {0: "female", 1: "male"}

# Global session data
current_session_images = []

@app.route('/predict', methods=['POST'])
def predict():
    global current_session_images
    
    # Clear previous session's images
    current_session_images.clear()

    files = request.files.getlist('images')
    results = {"male": 0, "female": 0, "processed_images": []}

    for file in files:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image, male_count, female_count = process_and_label_faces(image)
        
        results["male"] += male_count
        results["female"] += female_count

        unique_filename = f"{uuid.uuid4().hex}.jpg"
        full_save_path = os.path.join(save_path, unique_filename)
        cv2.imwrite(full_save_path, processed_image)

        image_url = f"/static/processed/{unique_filename}"
        results["processed_images"].append(image_url)
        current_session_images.append(image_url)

    return jsonify(results)

@app.route('/processed-images', methods=['GET'])
def get_processed_images():
    global current_session_images
    return jsonify({"processed_images": current_session_images})

def process_and_label_faces(image):
    """
    Detect faces using MTCNN or Haar Cascade, predict gender, and annotate the image.
    Returns the processed image along with male and female counts.
    """
    global cascade, mtcnn_detector, model, LABELS

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mtcnn_faces = mtcnn_detector.detect_faces(rgb_image)
    
    male_count = 0
    female_count = 0

    if mtcnn_faces:
        # Use MTCNN for face detection
        for face in mtcnn_faces:
            x, y, w, h = face['box']
            face_crop = rgb_image[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (256, 256))
            img_scaled = face_crop / 255.0
            reshape = np.reshape(img_scaled, (1, 256, 256, 3))

            result = model.predict(reshape)
            predicted_class = np.argmax(result, axis=1)[0]

            label = LABELS[predicted_class]
            color = (0, 255, 0) if predicted_class == 1 else (255, 0, 0)  # Green for male, blue for female
            cv2.rectangle(image, (x-10, y), (x+w, y+h), color, 4)
            cv2.rectangle(image, (x-10, y-50), (x+w, y), color, -1)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            if predicted_class == 0:
                female_count += 1
            else:
                male_count += 1
    else:
        # Fallback to Haar Cascade if MTCNN fails
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haar_faces = cascade.detectMultiScale(gray, 1.1, 7)

        for x, y, w, h in haar_faces:
            face_crop = image[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (256, 256))
            img_scaled = face_crop / 255.0
            reshape = np.reshape(img_scaled, (1, 256, 256, 3))

            result = model.predict(reshape)
            predicted_class = np.argmax(result, axis=1)[0]

            label = LABELS[predicted_class]
            color = (0, 255, 0) if predicted_class == 1 else (255, 0, 0)
            cv2.rectangle(image, (x-10, y), (x+w, y+h), color, 4)
            cv2.rectangle(image, (x-10, y-50), (x+w, y), color, -1)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            if predicted_class == 0:
                female_count += 1
            else:
                male_count += 1

    return image, male_count, female_count

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
