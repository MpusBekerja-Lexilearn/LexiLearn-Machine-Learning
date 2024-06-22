import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
from sklearn.metrics import f1_score

# Define paths
processed_data_dir = '/Users/amryzulfaalhusna/Documents/GitHub/LexiLearn-Machine-Learning/data/processed'
preprocessed_data_path = os.path.join(processed_data_dir, 'emnist_preprocessed.npz')
model_save_path = os.path.join(processed_data_dir, 'emnist_letters_model.h5')

def load_preprocessed_data():
    """Load preprocessed EMNIST data."""
    data = np.load(preprocessed_data_path)
    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    return train_images, train_labels, test_images, test_labels

def build_model(input_shape=(28, 28, 1), num_classes=26):
    """Build CNN model for EMNIST dataset."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, x_val, y_val):
        super().__init__()
        self.model = model
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.x_val), axis=1)
        y_true = np.argmax(self.y_val, axis=1)
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f' - F1-score: {f1:.4f}')


def train_and_save_model(model, train_images, train_labels, test_images, test_labels, epochs=10, batch_size=128):
    """Train the CNN model and save it."""
    f1score_callback = F1ScoreCallback(model, test_images, test_labels)
    history = model.fit(train_images, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(test_images, test_labels),
                        callbacks=[f1score_callback])

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    # Load preprocessed data
    train_images, train_labels, test_images, test_labels = load_preprocessed_data()

    # Build the model
    model = build_model()

    # Display model summary
    model.summary()

    # Train and save the model
    train_and_save_model(model, train_images, train_labels, test_images, test_labels)
