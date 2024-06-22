import tensorflow as tf
import tensorflowjs as tfjs
import os

# Load the trained model
models_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/az-models'
model_path = os.path.join(models_dir, 'az_handwritten_model.h5')
model = tf.keras.models.load_model(model_path)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_models_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/az-models'
os.makedirs(tflite_models_dir, exist_ok=True)
with open(os.path.join(tflite_models_dir, 'az_handwritten_model.tflite'), 'wb') as f:
    f.write(tflite_model)

print("Model conversion to TensorFlow Lite complete.")

# Convert the model to TensorFlow.js format
tfjs_models_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/az-models/tfjs_model'
os.makedirs(tfjs_models_dir, exist_ok=True)
tfjs.converters.save_keras_model(model, tfjs_models_dir)

print("Model conversion to TensorFlow.js complete.")