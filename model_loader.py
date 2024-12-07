from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)
model = load_model('saved_model/gender_classifier_final.h5')
cascade = cv2.CascadeClassifier('datasets/Input/haarcascade_frontalface_default.xml')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('images')  # Ambil banyak file
    results = {"male": 0, "female": 0, "processed_images": []}

    for file in files:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 7)

        for x, y, w, h in faces:
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (256, 256))
            img_scaled = face / 255.0
            reshape = np.reshape(img_scaled, (1, 256, 256, 3))
            prediction = model.predict(reshape)
            predicted_class = np.argmax(prediction, axis=1)[0]

            if predicted_class == 0:
                results["female"] += 1
            else:
                results["male"] += 1

        # Simpan gambar yang diproses ke folder statis
        save_path = os.path.join("static", "processed", file.filename)
        cv2.imwrite(save_path, image)
        results["processed_images"].append(request.host_url + save_path)

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
