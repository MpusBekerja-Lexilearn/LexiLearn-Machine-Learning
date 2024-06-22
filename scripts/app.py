from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load models
az_model_path = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/az-models/az_handwritten_model.h5'
emnist_model_path = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/processed/emnist_letters_model.h5'
az_model = tf.keras.models.load_model(az_model_path)
emnist_model = tf.keras.models.load_model(emnist_model_path)

# Define label dictionaries
az_labels = {i: chr(65 + i) for i in range(26)}
emnist_labels = {i: chr(65 + i) for i in range(26)}

def preprocess_image(image, target_size):
    image = image.convert("L")
    image = image.resize(target_size)
    image = np.array(image)
    image = image.reshape((1, target_size[0], target_size[1], 1))
    image = image / 255.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        img = Image.open(file.stream)
        preprocessed_img = preprocess_image(img, target_size=(28, 28))
        model_type = request.form['model_type']
        if model_type == 'AZ':
            prediction = az_model.predict(preprocessed_img)
            predicted_label = az_labels[np.argmax(prediction)]
        else:
            prediction = emnist_model.predict(preprocessed_img)
            predicted_label = emnist_labels[np.argmax(prediction)]
        return jsonify({"predicted_label": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)