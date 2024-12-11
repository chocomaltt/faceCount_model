from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt  # Required for plt.imshow (if needed for debugging)

app = Flask(__name__)
CORS(app)
path = "/home/chocomalt/kuliah/Machine Learning/faceCount_model/"
model = load_model(os.path.join(path, 'saved_model/model_gender_final.h5'))
cascade = cv2.CascadeClassifier(os.path.join(path, 'datasets/Input/haarcascade_frontalface_default.xml'))
save_path = os.path.join(path, 'static/processed')

# Labels for prediction
LABELS = {0: "female", 1: "male"}

import uuid  # Import the uuid library for generating unique filenames

current_session_images = []

@app.route('/predict', methods=['POST'])
def predict():
    global current_session_images
    
    # Clear previous session's images
    current_session_images.clear()
    
    files = request.files.getlist('images')
    results = {"male": 0, "female": 0, "processed_images": []}

    save_directory = os.path.join(os.path.dirname(__file__), 'static', 'processed')
    os.makedirs(save_directory, exist_ok=True)

    for file in files:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image, male_count, female_count = process_and_label_faces(image)
        
        results["male"] += male_count
        results["female"] += female_count
        
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        full_save_path = os.path.join(save_directory, unique_filename)
        cv2.imwrite(full_save_path, processed_image)
        
        # Store only the current session's image URLs
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
    Detect faces, predict gender, and draw rectangles and labels on the image.
    Returns the processed image along with male and female counts.
    """
    global cascade, model, LABELS

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 7)  # Detect faces
    male_count = 0
    female_count = 0

    for x, y, w, h in faces:
        # Extract face region
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (256, 256))
        img_scaled = face / 255.0
        reshape = np.reshape(img_scaled, (1, 256, 256, 3))
        
        # Predict gender
        result = model.predict(reshape)
        predicted_class = np.argmax(result, axis=1)[0]

        # Draw rectangle and add label
        label = LABELS[predicted_class]
        color = (0, 255, 0) if predicted_class == 1 else (255, 0, 0)  # Green for male, blue for female
        cv2.rectangle(image, (x-10, y), (x+w, y+h), color, 4)
        cv2.rectangle(image, (x-10, y-50), (x+w, y), color, -1)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        # Update counts
        if predicted_class == 0:
            female_count += 1
        else:
            male_count += 1

    return image, male_count, female_count

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
