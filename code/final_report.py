import tensorflow as tf
from predict_images import predict_image_categories
from preprocess_images import preprocessor
from augment_images import data_generator, augmented_images


# Define paths
image_paths = []  # input/import image_paths
model_path = '' # chosen model

# Preprocess images
preprocessed_images = preprocessor(image_paths)
datagen = data_generator()
augmented_images = (preprocessed_images, datagen)

# Make predictions
predictions = predict_image_categories(augmented_images, model_path)

# Display predictions
for image_path, prediction in zip(image_paths, predictions):
    print(f'Image: {image_path}, Prediction: {prediction}')
