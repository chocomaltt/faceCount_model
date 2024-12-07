
# Face Count Model Repository

Welcome to the **Face Count Model** repository! This repository contains the core deep learning model for the **Face Count** application, designed to detect gender from facial images. The model utilizes a **Convolutional Neural Network (CNN)** built with TensorFlow's Sequential API.

---

## 🚀 Features

- **Gender Detection**: Classifies gender based on facial images with high accuracy.
- **Saved Model**: Pretrained models available in `.h5` and `.keras` formats for direct use.
- **Notebook Workflow**: Includes Jupyter Notebook for training and evaluation.

---

## 📂 Repository Structure

```plaintext
├── saved_models/
│   ├── gender_classifier_final.h5        # Saved model in H5 format
│   ├── gender_classifier_final2.tflite   # Saved model in tflite format
│   ├── model_gender_final.h5        # Saved model in H5 format
│   ├── model_gender_final.keras     # Saved model in Keras format
│
├── notebooks/
│   ├── model_gender.ipynb  # latest notebook, more efficient and lighter
│   ├── gender_classification.ipynb  # Old notebook
│
├── README.md           # Project overview (this file)
```

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Jupyter Notebook

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chocomaltt/faceCount_model.git
   cd faceCount_model
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow notebook
   ```

3. **Download Dataset**:
   Since the dataset is not included due to size constraints, download your dataset and ensure it is organized into training, validation, and test sets.

---

## 🧠 Model Overview

The model is a **Convolutional Neural Network (CNN)** built for binary classification (gender detection). 

### Architecture Highlights:
- **Convolutional Layers**: Extract spatial features from facial images.
- **MaxPooling**: Reduces spatial dimensions while retaining key features.
- **Fully Connected Layers**: Aggregate features for classification.
- **Output Layer**: Binary classification using Softmax activation.

### Training Details:
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

---

## 🚀 Usage

1. **Training the Model**:
   Open the `gender_classification.ipynb` or `model_gender.ipynb` notebook in Jupyter and follow the steps to preprocess the data, train the model, and save it.

2. **Using the Pretrained Model**:
   Load the saved model in your application:
   ```python
   from tensorflow.keras.models import load_model

   model = load_model("saved_model/model_gender_final.h5") # use model_gender_final for the most optimal model
   ```

---

## 📊 Results

The model achieves high accuracy on gender detection tasks. Refer to the evaluation results in the notebook for detailed metrics.

---


## 🌟 Acknowledgments

Special thanks to:
- TensorFlow for providing the framework for this project.
- The community for inspiring the creation of this model.

---

Feel free to reach out by creating an issue for any questions or suggestions. Happy coding! 😊
