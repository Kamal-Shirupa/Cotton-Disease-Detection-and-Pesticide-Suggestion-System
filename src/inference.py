from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_image(model, image_path, img_height=128, img_width=128, class_labels=None):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    if class_labels:
        predicted_class = class_labels[predicted_index]
        return predicted_class
    return predicted_index
