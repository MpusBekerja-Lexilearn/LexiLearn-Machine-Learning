import os
import gzip
import struct
import numpy as np
import tensorflow as tf

# Define paths
project_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning'
data_dir = os.path.join(project_dir, 'data')
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')

train_images_gz = os.path.join(raw_data_dir, 'emnist-letters-train-images-idx3-ubyte.gz')
train_labels_gz = os.path.join(raw_data_dir, 'emnist-letters-train-labels-idx1-ubyte.gz')
test_images_gz = os.path.join(raw_data_dir, 'emnist-letters-test-images-idx3-ubyte.gz')
test_labels_gz = os.path.join(raw_data_dir, 'emnist-letters-test-labels-idx1-ubyte.gz')

# Extract files
def extract_gz(file_path, output_path):
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())

# Paths to extracted files
extracted_dir = os.path.join(processed_data_dir, 'extracted_emnist')
os.makedirs(extracted_dir, exist_ok=True)
train_images_path = os.path.join(extracted_dir, 'train-images-idx3-ubyte')
train_labels_path = os.path.join(extracted_dir, 'train-labels-idx1-ubyte')
test_images_path = os.path.join(extracted_dir, 'test-images-idx3-ubyte')
test_labels_path = os.path.join(extracted_dir, 'test-labels-idx1-ubyte')

# Extract all .gz files
extract_gz(train_images_gz, train_images_path)
extract_gz(train_labels_gz, train_labels_path)
extract_gz(test_images_gz, test_images_path)
extract_gz(test_labels_gz, test_labels_path)

# Load data
def load_images(file_path):
    with open(file_path, 'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        _, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

train_images = load_images(train_images_path)
train_labels = load_labels(train_labels_path)
test_images = load_images(test_images_path)
test_labels = load_labels(test_labels_path)

# Normalize and reshape data
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# One-hot encode labels
num_classes = 26
train_labels = tf.keras.utils.to_categorical(train_labels - 1, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels - 1, num_classes)

# Save preprocessed data
np.savez(os.path.join(processed_data_dir, 'emnist_preprocessed.npz'),
         train_images=train_images, train_labels=train_labels,
         test_images=test_images, test_labels=test_labels)

print("Data preprocessing complete and saved.")