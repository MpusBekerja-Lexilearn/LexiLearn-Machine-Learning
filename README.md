# LexiLearn-Machine-Learning
Machine Learning models and code for LexiLearn project

## Project Structure
The project is structured as follows:

```LexiLearn-Machine-Learning/
│
├── data/
│   ├── raw/              # Raw data directory
│   │   ├── emnist-letters-train-images-idx3-ubyte.gz
│   │   ├── emnist-letters-train-labels-idx1-ubyte.gz
│   │   ├── emnist-letters-test-images-idx3-ubyte.gz
│   │   └── emnist-letters-test-labels-idx1-ubyte.gz
│   └── processed/        # Processed data directory
│       ├── emnist_preprocessed.npz       # Preprocessed data
│       ├── emnist_letters_model.h5       # Trained model
│       └── converted_model.tflite        # Converted model (TFLite format)
│
├── colab-notebooks/            # notebooks for data preprocessing and model training
│   └── Dataset_Preparation_and_Preprocessing_EMNIST_Letters.ipynb
│
└── scripts/              # Python scripts for data preprocessing, model training, and model conversion
    ├── preprocess_data.py
    ├── train_model.py
    └── convert_model.py
```

## Environment
This project utilizes the following Python libraries and dependencies:

```
numpy==1.24.3
pandas==2.0.3
Pillow==10.0.0
scipy==1.10.1
tensorflow==2.13.0
tensorflow-datasets==4.9.2
```

## Scripts
This section contains Python scripts for different tasks in the machine learning pipeline.

- `preprocess_data.py`: Script for preprocessing raw data, including extracting, loading, and preprocessing images and labels.
- `train_model.py`: Script for training machine learning models using preprocessed data.
- `convert_model.py`: Script for converting trained models to the TensorFlow Lite format for deployment on mobile devices.

## Models
This folder contains trained machine learning models.
- `emnist_letters_model.h5`: Trained TensorFlow/Keras model for handwritten text recognition using the EMNIST letters dataset.

## Processed Data
This directory contains preprocessed data used for training and evaluation.
- `emnist_preprocessed.npz`: Preprocessed data for training and testing machine learning models, including images and labels.

## Converted Models
This directory stores models converted to TensorFlow Lite format for deployment on mobile devices.
- `converted_model.tflite`: TensorFlow Lite model converted from the original TensorFlow/Keras model.

## Model Accuracy
The model trained on the EMNIST dataset achieved an accuracy of approximately `94.54%` on the test set.
