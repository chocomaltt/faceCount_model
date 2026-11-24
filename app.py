from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from ai_edge_litert.interpreter import Interpreter
import base64
from pydantic import BaseModel
import cv2
import numpy as np
import os

app = FastAPI()

# Define paths
path = os.path.join(os.path.abspath(__file__))
model_path = "saved_model/model_gender_dynamic.tflite"
cascade_path = 'datasets/Input/haarcascade_frontalface_default.xml'
save_path = '/app/static/processed'

# Load model and cascade
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

in_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']

cascade = cv2.CascadeClassifier(cascade_path)

# Labels for prediction
LABELS = {0: "female", 1: "male"}

current_session_images = []

class PredictResponse(BaseModel):
    male: int
    female: int
    image: str

@app.post('/predict', response_model=PredictResponse)
async def predict(files: UploadFile):
    results = {"male": 0, "female": 0, "processed_images": []}

    data = await files.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail=f"{files.filename} is not valid")
    processed_image, male_count, female_count = process_and_label_faces(image)
    
    results["male"] += male_count
    results["female"] += female_count
    
    _, buffer = cv2. imencode('.jpg', processed_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    image_string = f"data:image/jpeg;base64,{img_base64}"
    
    predict_result = {"male": male_count, "female": female_count, "image": image_string}
    return PredictResponse(**predict_result)

@app.get('/processed-images')
def get_processed_images():
    global current_session_images
    return jsonable_encoder({"processed_images": current_session_images})

def process_and_label_faces(image):
    """
    Detect faces, predict gender, and draw rectangles and labels on the image.
    Returns the processed image along with male and female counts.
    """
    global cascade, model, LABELS

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 7)  # Detect faces
    male_count = female_count = 0

    for (x, y, w, h) in faces:
        # Extract face region
        face = image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (150, 150)).astype(np.float32) / 255.0
        face = np.reshape(face, (1, 150, 150, 3))

        interpreter.set_tensor(in_idx, face)
        interpreter.invoke()
        preds = interpreter.get_tensor(out_idx)
        cls = int(np.argmax(preds[0]))

        
        label = LABELS[cls]
        color = (0, 255, 0) if cls == 1 else (255, 0, 0)  # Green for male, blue for female
        cv2.rectangle(image, (x-10, y), (x+w, y+h), color, 4)
        cv2.rectangle(image, (x-10, y-50), (x+w, y), color, -1)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        if cls == 0: female_count += 1
        else: male_count += 1

    return image, male_count, female_count

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

