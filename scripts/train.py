import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
processed_data_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/processed'
preprocessed_data_path = os.path.join(processed_data_dir, 'az_preprocessed.npz')
model_save_path = os.path.join(processed_data_dir, 'az_handwritten_model.h5')

def load_preprocessed_data():
    """Load preprocessed A-Z Handwritten data."""
    data = np.load(preprocessed_data_path)
    images = data['images']
    labels = data['labels']
    return images, labels

def build_model(input_shape=(28, 28, 1), num_classes=26):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), axis=-1)
        val_targ = np.argmax(self.validation_data[1], axis=-1)
        f1 = f1_score(val_targ, val_predict, average='weighted')
        print(f" — val_f1: {f1:.4f}")


# Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

def train_and_save_model_with_augmentation(model, images, labels, epochs=10, batch_size=128):
    """Train the CNN model with data augmentation and save it."""
    validation_split = 0.2
    val_size = int(validation_split * len(images))
    val_data = (images[:val_size], labels[:val_size])

    f1_callback = F1ScoreCallback(validation_data=val_data)

    datagen.fit(images[val_size:])
    train_generator = datagen.flow(images[val_size:], labels[val_size:], batch_size=batch_size)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[f1_callback]
    )

    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    # Load preprocessed data
    images, labels = load_preprocessed_data()

    # Build the model
    model = build_model()

    # Display model summary
    model.summary()

    # Train and save the model with augmentation
    train_and_save_model_with_augmentation(model, images, labels)
