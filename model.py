# Transfer Learning using MobileNet-V3 large
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator

trainPath = "/Users/kshitija/Desktop/AI/transfer/Fish_Dataset/dataset_for_model/train/"
validPath = "/Users/kshitija/Desktop/AI/transfer/Fish_Dataset/dataset_for_model/validate/"

# 1. Data augmentation and loading using Keras
trainGenerator = ImageDataGenerator(
    rotation_range=15, # Rotates images randomly by up to 15 degrees.
    width_shift_range=0.1, # Randomly shifts the image horizontally by up to 10% of the width.
    height_shift_range=0.1, # Randomly shifts the image vertically by up to 10% of the height.
    brightness_range=(0,0.2) # Adjusts brightness of images randomly. (0, 0.2) makes images darker
    ).flow_from_directory(
        trainPath, target_size=(320,320),batch_size=32) # Resizes all images to 320x320 pixels. Required because CNNs need fixed-size inputs.

validGenerator = ImageDataGenerator(
    rotation_range=15, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0,0.2)).flow_from_directory(validPath, target_size=(320,320),batch_size=32)

# 2. Build the Model
# This loads pre-trained weights from the ImageNet dataset (a massive dataset with over 1M labeled images and 1000 classes).
# include_top=False removes the classification head, leaving only the convolutional base.
baseModel = MobileNetV3Large(weights="imagenet",include_top=False)

x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
predictionLayer = tf.keras.layers.Dense(9, activation='softmax')(x)

model = tf.keras.Model(inputs=baseModel.input,outputs=predictionLayer)
# print(model.summary())

# Freeze the layers of the MobileNetV3 (already trained)
for layers in model.layers[:-5]:  # Exclude the last 5 layers
    layers.trainable = False
    
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

# 3. Train the Model
model.fit(trainGenerator,validation_data=validGenerator,epochs=5)
modelSavedPath = "/Users/kshitija/Desktop/AI/transfer/Fish_Dataset/dataset_for_model/FishV3.keras"
model.save(modelSavedPath)