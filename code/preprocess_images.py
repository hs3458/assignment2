from keras.preprocessing import image
import numpy as np
from PIL import Image

def preprocessor(img_path, target_size=(192, 192)):
        """Load and preprocess an image."""
        img = Image.open(img_path).convert("RGB").resize((192,192))
        img = (np.float32(img)-1.)/(255-1.)
        img=img.reshape((192,192,3))
        return img
