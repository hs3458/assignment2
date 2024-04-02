from tensorflow.keras.preprocessing.image import ImageDataGenerator

def image_data_generator():
  return ImageDataGenerator(
    zoom_range = 0.1, # apply random zoom transformations
    rotation_range = 15, # randomly rotate images by 15 degrees
    width_shift_range = .05, # randomly shift images horizontally by 5% of the width
    height_shift_range = .05 # randomly shift images vertically by 5% of the height
  )
