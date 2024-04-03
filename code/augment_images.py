from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def data_generator():
  datagen= ImageDataGenerator(
    zoom_range = 0.1, # apply random zoom transformations
    rotation_range = 15, # randomly rotate images by 15 degrees
    width_shift_range = .05, # randomly shift images horizontally by 5% of the width
    height_shift_range = .05 # randomly shift images vertically by 5% of the height)
    return datagen

def augment_images(img, datagen):
    img = np.expand_dims(img, axis=0)
    # Generate augmented images using ImageDataGenerator
    augmented_img = next(datagen.flow(img, batch_size=1))
    # Remove the batch dimension
    augmented_img = np.squeeze(augmented_img, axis=0)
    return augmented_img
