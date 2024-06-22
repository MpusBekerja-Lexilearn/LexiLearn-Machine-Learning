import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os

# Load the dataset
data_path = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/raw/A_Z Handwritten Data.csv'
data = pd.read_csv(data_path).astype('float32')

# Separate features and labels
X = data.drop('0', axis=1)
y = data['0']

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

# Reshape the data into 28x28 images
train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28, 1))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28, 1))

# Convert labels to categorical (one-hot encoding)
train_yOHE = to_categorical(train_y, num_classes=26, dtype='int')
test_yOHE = to_categorical(test_y, num_classes=26, dtype='int')

# Define the directory to save processed data
processed_data_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/processed/az-processed'

# Create the directory if it doesn't exist
os.makedirs(processed_data_dir, exist_ok=True)

# Save processed data as .npy files
np.save(os.path.join(processed_data_dir, 'train_x.npy'), train_x)
np.save(os.path.join(processed_data_dir, 'test_x.npy'), test_x)
np.save(os.path.join(processed_data_dir, 'train_y.npy'), train_yOHE)
np.save(os.path.join(processed_data_dir, 'test_y.npy'), test_yOHE)

print("Data preprocessing complete.")
