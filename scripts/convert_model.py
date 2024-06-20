import tensorflow as tf
import os
import json

# Define paths
project_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning'
processed_data_dir = os.path.join(project_dir, 'data', 'processed')
model_path = os.path.join(processed_data_dir, 'emnist_letters_model.h5')
tflite_model_path = os.path.join(processed_data_dir, 'converted_model.tflite')
json_model_path = os.path.join(processed_data_dir, 'emnist_letters_model.json')

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# Save the model architecture as JSON
model_json = model.to_json()
with open(json_model_path, 'w') as json_file:
    json_file.write(model_json)

print("Conversion and saving to TFLite and JSON complete.")