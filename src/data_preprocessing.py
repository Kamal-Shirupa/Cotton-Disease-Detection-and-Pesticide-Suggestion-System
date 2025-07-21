import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, test_dir, img_height=128, img_width=128, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical"
    )
    val_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical"
    )
    return train_gen, val_gen, list(train_gen.class_indices.keys())
