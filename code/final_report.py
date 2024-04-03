import tensorflow as tf
from predict_images import predict_image_categories
from preprocess_images import preprocessor


# Define paths
image_paths = []  # input/import image_paths
model_path = '' # chosen model

# Preprocess images
preprocessed_images = preprocessor(image_paths)

# Make predictions
predictions = predict_image_categories(image_paths, model_path)

# Display predictions
for image_path, prediction in zip(image_paths, predictions):
    print(f'Image: {image_path}, Prediction: {prediction}')
