import os
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.preprocessing import image
from PIL import Image

# 1. get the list of categories
categories = os.listdir("/Users/kshitija/Desktop/AI/transfer/Fish_Dataset/dataset_for_model/train/")
categories.sort()
# print(categories)

# 2. Load Saved Model
modelSavedPath = "/Users/kshitija/Desktop/AI/transfer/Fish_Dataset/dataset_for_model/FishV3.keras"
model = tf.keras.models.load_model(modelSavedPath)

# 3. Predict the image
def classify_image(imageFile):
    x = []
    
    img = Image.open(imageFile)
    img.load()
    img = img.resize((320,320), Image.Resampling.LANCZOS) # Image.Resampling.LANCZOS is a resampling filter used to reduce aliasing (smoother image).
    
    x = image.img_to_array(img) # (320, 320, 3) for a 3-channel RGB image. Deep learning models work with NumPy arrays or Tesnors
    x = np.expand_dims(x,axis=0)
    print(x.shape)
    # Adds an extra dimension at position 0 to make the shape (1, 320, 320, 3).
    # # This is because models expect a batch of images, even if you're passing just one.
    
    pred = model.predict(x)
    # print(pred)
    
    categoryValue = np.argmax(pred,axis=1)[0]
    result = categories[categoryValue]
    
    return result
    
    
img_path = "test.jpeg"
resultText = classify_image(img_path)
print(resultText)
