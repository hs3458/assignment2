import numpy as np
from tensorflow.keras.models import load_model

def load_image(image_path, target_size=(192, 192)):
    """Load and preprocess an image."""
    img = Image.open(img_path).convert("RGB").resize((192,192))
    img = (np.float32(img)-1.)/(255-1.)
    img=img.reshape((192,192,3))
    return img

def predict_image_categories(image_paths, model_path):
    """Predict categories for a batch of images."""
    # Load the trained model
    model = tf.keras.models.load_model(model_path,compile=False)

    # Initialize an empty list to store the predictions
    predictions = []

    # Iterate over the image paths and make predictions
    for image_path in image_paths:
        # Load and preprocess the image
        img = load_image(image_path)

        # Expand dimensions to match the model's input shape
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)

        # Append the prediction to the list
        predictions.append(prediction)

    return predictions
