import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("waste_classifier.h5")

# Define image path (replace with your actual test image)
test_image_path = "img2.jpg"

# Load and preprocess the image
img = image.load_img(test_image_path, target_size=(224, 224))  # Resize to model input size
img_array = image.img_to_array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)  # Get the class index

# Print results
print("Class Probabilities:", predictions)
print("Predicted Class Index:", predicted_class)
