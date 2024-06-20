import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# Define paths
project_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning'
data_dir = os.path.join(project_dir, 'data')
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')

# Load A-Z Handwritten Data.csv
az_dataset_path = os.path.join(raw_data_dir, 'A_Z Handwritten Data.csv')
az_dataset = pd.read_csv(az_dataset_path, header=None)

# Split into features (X_az) and labels (y_az)
X_az = az_dataset.iloc[:, 1:].values  # Features (pixel values)
y_az = az_dataset.iloc[:, 0].values   # Labels (letters)

# Normalize and reshape for A-Z Handwritten Data
def preprocess_images(images):
    processed_images = []
    for img in images:
        img = img.reshape(28, 28).astype('uint8')  # Reshape to 28x28 and convert to uint8
        img = Image.fromarray(img)  # Convert to PIL image
        img = img.resize((28, 28), Image.LANCZOS)  # Resize to 28x28
        img = np.array(img).astype('float32') / 255.0  # Normalize
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        processed_images.append(img)
    return np.array(processed_images)

X_az_processed = preprocess_images(X_az)

# One-hot encode labels for A-Z Handwritten Data
num_classes_az = 26  # A-Z
y_az_processed = tf.keras.utils.to_categorical(y_az, num_classes_az)

# Save preprocessed data
np.savez(os.path.join(processed_data_dir, 'az_preprocessed.npz'),
         images=X_az_processed, labels=y_az_processed)

print("Data preprocessing complete and saved.")