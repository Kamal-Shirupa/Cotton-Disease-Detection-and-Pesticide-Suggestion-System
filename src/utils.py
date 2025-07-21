# Example: add visualization utilities, logging, or helper functions here.
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Training History')
    plt.legend()
    plt.show()
