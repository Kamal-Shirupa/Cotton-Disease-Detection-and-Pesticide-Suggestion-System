# (Src) src/

This folder contains the core source code modules for the crop disease detection and recommendation system. Each script is modular and focused on a single part of the ML pipeline, making the project scalable, testable, and easy to maintain.

## 📄 Modules Overview
- `data_preprocessing.py`  
  Loads and preprocesses image data for training and testing using Keras' `ImageDataGenerator`.
- `model_training.py`  
  Builds and trains a Convolutional Neural Network (CNN) using TensorFlow/Keras. Also handles saving the best-performing model.
- `recommendation.py`  
  Loads pesticide recommendation data from a CSV file and provides treatment suggestions based on predicted disease classes.
- `inference.py`  
  Performs prediction on input images and returns the most likely disease class using the trained model.
- `utils.py`  
  Contains helper functions, like training history plots, that support the overall pipeline.
