import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import os

# Load processed data
processed_data_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/processed/az-processed'
train_x = np.load(os.path.join(processed_data_dir, 'train_x.npy'))
test_x = np.load(os.path.join(processed_data_dir, 'test_x.npy'))
train_y = np.load(os.path.join(processed_data_dir, 'train_y.npy'))
test_y = np.load(os.path.join(processed_data_dir, 'test_y.npy'))

# Define the model architecture
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y), callbacks=[reduce_lr, early_stop])

# Save the model
models_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/az-models'
os.makedirs(models_dir, exist_ok=True)
model.save(os.path.join(models_dir, 'az_handwritten_model.h5'))

print("Model training complete.")