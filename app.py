from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import cv2
import numpy as np
import os
import uuid  # For generating unique filenames

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Path configuration
    model_path = "/app/saved_model/model_gender_final.h5"
    cascade_path = "/app/datasets/Input/haarcascade_frontalface_default.xml"
    save_path = "/app/static/processed"
    os.makedirs(save_path, exist_ok=True)

    # Load model and cascade classifier
    model = load_model(model_path)
    cascade = cv2.CascadeClassifier(cascade_path)
    mtcnn_detector = MTCNN()

    # Labels for prediction
    LABELS = {0: "female", 1: "male"}

    @app.route('/predict', methods=['POST'])
    def predict():
        files = request.files.getlist('images')
        male_count, female_count = 0, 0
        processed_images = []

        for file in files:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            result, annotated_image = process_and_label_faces(image, model, cascade, mtcnn_detector, LABELS)

            # Accumulate counts
            male_count += result['male']
            female_count += result['female']

            # Save processed image
            unique_filename = f"{uuid.uuid4().hex}.jpg"
            full_save_path = os.path.join(save_path, unique_filename)
            cv2.imwrite(full_save_path, annotated_image)

            image_url = f"/static/processed/{unique_filename}"
            processed_images.append(image_url)

        return jsonify({
            "male": male_count,
            "female": female_count,
            "processed_images": processed_images
        })

    def process_and_label_faces(image, model, cascade, mtcnn_detector, labels):
        """
        Detect faces using MTCNN or Haar Cascade, predict gender, and annotate the image.
        Returns a dictionary with counts and the annotated image.
        """
        male_count, female_count = 0, 0
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mtcnn_faces = mtcnn_detector.detect_faces(rgb_image)

        if mtcnn_faces:
            # Use MTCNN for face detection
            for face in mtcnn_faces:
                x, y, w, h = face['box']
                face_crop = preprocess_face(rgb_image, x, y, w, h)
                if face_crop is not None:
                    predicted_class = predict_gender(face_crop, model)
                    label, color = annotate_image(image, x, y, w, h, predicted_class, labels)
                    male_count, female_count = update_counts(predicted_class, male_count, female_count)
        else:
            # Fallback to Haar Cascade if MTCNN fails
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            haar_faces = cascade.detectMultiScale(gray, 1.1, 7)
            for x, y, w, h in haar_faces:
                face_crop = preprocess_face(image, x, y, w, h)
                if face_crop is not None:
                    predicted_class = predict_gender(face_crop, model)
                    label, color = annotate_image(image, x, y, w, h, predicted_class, labels)
                    male_count, female_count = update_counts(predicted_class, male_count, female_count)

        return {"male": male_count, "female": female_count}, image

    def preprocess_face(image, x, y, w, h):
        """Extract and preprocess face for model prediction."""
        try:
            face_crop = image[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (150, 150))
            face_crop = face_crop / 255.0
            return np.reshape(face_crop, (1, 150, 150, 3))
        except Exception:
            return None

    def predict_gender(face_crop, model):
        """Predict gender of the given face crop."""
        result = model.predict(face_crop)
        return np.argmax(result, axis=1)[0]

    def annotate_image(image, x, y, w, h, predicted_class, labels):
        """Annotate the image with the predicted gender label."""
        label = labels[predicted_class]
        color = (0, 255, 0) if predicted_class == 1 else (255, 0, 0)
        cv2.rectangle(image, (x-10, y), (x+w, y+h), color, 4)
        cv2.rectangle(image, (x-10, y-50), (x+w, y), color, -1)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        return label, color

    def update_counts(predicted_class, male_count, female_count):
        """Update gender counts based on prediction."""
        if predicted_class == 0:
            female_count += 1
        else:
            male_count += 1
        return male_count, female_count

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
