from keras.preprocessing import image
import numpy as np
from PIL import Image

def preprocessor(img_path):
        img = Image.open(img_path).convert("RGB").resize((224,224))
        img = (np.float32(img)-1.)/(255-1.)
        img=img.reshape((224,224,3))
        return img
