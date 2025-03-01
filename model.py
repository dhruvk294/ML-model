import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
from google.colab import drive
drive.mount('/content/drive')
# Define dataset path
dataset_path = "/content/drive/MyDrive/Waste Segregation_dataset/train"
from PIL import Image
import os

dataset_path = "/content/drive/MyDrive/Waste Segregation_dataset/train"  # Change this to your dataset directory

def check_and_remove_corrupt_images(directory):
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                img_path = os.path.join(category_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Check if the image is valid
                except (OSError, IOError):
                    print(f"Corrupt image removed: {img_path}")
                    os.remove(img_path)  # Delete the corrupt image

check_and_remove_corrupt_images(dataset_path)
# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Fix truncated image loading

# Now define ImageDataGenerator and continue your training
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Data augmentation and loading
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
# Load pre-trained MobileNetV2 model as feature extractor
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers
# Custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(len(train_data.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output_layer)
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
epochs = 10
history = model.fit(train_data, validation_data=val_data, epochs=epochs)
model.save("waste_classifier.h5")
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("waste_classifier.h5")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("waste_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved!")
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
# Load the trained model
model = tf.keras.models.load_model('waste_classifier.h5')

# Define waste categories as per your labels
categories = [
    'e_waste',
    'food_waste',
    'leaf_waste',
    'metal_cans',
    'paper_waste',
    'plastic_bags',
    'plastic_bottles',
    'wood_waste'
]
# Load a test image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Use the input size your model expects
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array
# Predict waste category
def predict_waste_category(img_path):
    img, img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_category = categories[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Display the image
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_category} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    print(f"Category: {predicted_category}")
    print(f"Confidence: {confidence:.2f}%")
# Test the model with a sample image
test_image_path = 'test1.jpg'  # Replace with your test image path
predict_waste_category(test_image_path)
